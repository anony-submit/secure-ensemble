package server

import (
	"context"
	"encoding/gob"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"time"

	"cnn/svhn/go_inference/common"
	pb "cnn/svhn/go_inference/proto"
	ser "secure-ensemble/pkg/serialization"

	"github.com/anony-submit/snu-mghe/mkckks"
	"github.com/anony-submit/snu-mghe/mkrlwe"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type CSPTiming struct {
	InferenceStats        common.TimingStats
	EnsembleStats         common.TimingStats
	TotalComputeStats     common.TimingStats
	ClientTransferStats   common.TimingStats
	DataOwnerTransferTime time.Duration
}

type CSPServer struct {
	pb.UnimplementedCSPServiceServer
	mu sync.RWMutex

	mkParams  mkckks.Parameters
	evaluator *mkckks.Evaluator
	encryptor *mkckks.Encryptor

	timing          CSPTiming
	encryptedModels map[string]*common.EncryptedModel // cache

	pkSet  *mkrlwe.PublicKeySet
	rlkSet *mkrlwe.RelinearizationKeySet
	rtkSet *mkrlwe.RotationKeySet

	dataOwnerClients map[string]pb.DataOwnerServiceClient
	dataOwnerConns   map[string]*grpc.ClientConn

	storageDir string
}

type ownerKeySet struct {
	pkSet  *mkrlwe.PublicKeySet
	rlkSet *mkrlwe.RelinearizationKeySet
	rtkSet *mkrlwe.RotationKeySet
}

func newOwnerKeySet(params mkckks.Parameters) *ownerKeySet {
	return &ownerKeySet{
		pkSet:  mkrlwe.NewPublicKeyKeySet(),
		rlkSet: mkrlwe.NewRelinearizationKeySet(params.Parameters),
		rtkSet: mkrlwe.NewRotationKeySet(),
	}
}

func NewCSPServer(params mkckks.Parameters) *CSPServer {
	storageDir := "csp_disk"
	if err := os.MkdirAll(storageDir, 0755); err != nil {
		panic(fmt.Sprintf("Failed to create storage directory: %v", err))
	}

	return &CSPServer{
		mkParams:         params,
		evaluator:        mkckks.NewEvaluator(params),
		encryptor:        mkckks.NewEncryptor(params),
		encryptedModels:  make(map[string]*common.EncryptedModel),
		pkSet:            mkrlwe.NewPublicKeyKeySet(),
		rlkSet:           mkrlwe.NewRelinearizationKeySet(params.Parameters),
		rtkSet:           mkrlwe.NewRotationKeySet(),
		dataOwnerClients: make(map[string]pb.DataOwnerServiceClient),
		dataOwnerConns:   make(map[string]*grpc.ClientConn),
		storageDir:       storageDir,
	}
}

func (s *CSPServer) Start(address string) error {
	maxSize := 1024 * 1024 * 1024 * 4
	server := grpc.NewServer(
		grpc.MaxRecvMsgSize(maxSize),
		grpc.MaxSendMsgSize(maxSize),
	)
	pb.RegisterCSPServiceServer(server, s)

	listener, err := net.Listen("tcp", address)
	if err != nil {
		return fmt.Errorf("failed to start listener: %v", err)
	}

	go func() {
		if err := server.Serve(listener); err != nil {
			fmt.Printf("CSP server error: %v\n", err)
		}
	}()

	for i := 0; i < 5; i++ {
		conn, err := net.DialTimeout("tcp", address, time.Second)
		if err == nil {
			conn.Close()
			return nil
		}
		time.Sleep(time.Second)
	}

	return fmt.Errorf("server failed to start within timeout")
}

func (s *CSPServer) GetTiming() CSPTiming {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.timing
}

func (s *CSPServer) EnrollModel(ctx context.Context, req *pb.EnrollModelRequest) (*pb.EnrollModelResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	transferLatency := time.Duration(time.Now().UnixNano() - req.RequestStartTime)
	s.timing.DataOwnerTransferTime = transferLatency

	var conv1Weights [6]*mkckks.Ciphertext
	for i := 0; i < 6; i++ {
		ct, err := ser.DeserializeCiphertext(req.Conv1Weights[i], s.mkParams)
		if err != nil {
			return nil, fmt.Errorf("failed to deserialize Conv1 weight %d: %v", i, err)
		}
		conv1Weights[i] = ct
	}

	conv1Bias, err := ser.DeserializeCiphertext(req.Conv1Bias, s.mkParams)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize Conv1 bias: %v", err)
	}

	var conv2Weights [64]*mkckks.Ciphertext
	for i := 0; i < 64; i++ {
		ct, err := ser.DeserializeCiphertext(req.Conv2Weights[i], s.mkParams)
		if err != nil {
			return nil, fmt.Errorf("failed to deserialize Conv2 weight %d: %v", i, err)
		}
		conv2Weights[i] = ct
	}

	conv2Bias, err := ser.DeserializeCiphertext(req.Conv2Bias, s.mkParams)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize Conv2 bias: %v", err)
	}

	var fc1Weights [16]*mkckks.Ciphertext
	for i := 0; i < 16; i++ {
		ct, err := ser.DeserializeCiphertext(req.Fc1Weights[i], s.mkParams)
		if err != nil {
			return nil, fmt.Errorf("failed to deserialize FC1 weight %d: %v", i, err)
		}
		fc1Weights[i] = ct
	}

	fc1Bias, err := ser.DeserializeCiphertext(req.Fc1Bias, s.mkParams)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize FC1 bias: %v", err)
	}

	fc2Weights, err := ser.DeserializeCiphertext(req.Fc2Weights, s.mkParams)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize FC2 weights: %v", err)
	}

	fc2Bias, err := ser.DeserializeCiphertext(req.Fc2Bias, s.mkParams)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize FC2 bias: %v", err)
	}

	if err := ser.AddPublicKeyFromBytes(s.pkSet, req.PublicKey, s.mkParams.Parameters); err != nil {
		return nil, fmt.Errorf("failed to add public key: %v", err)
	}

	if err := ser.AddRelinKeyFromBytes(s.rlkSet, req.RelinearizationKey, s.mkParams.Parameters); err != nil {
		return nil, fmt.Errorf("failed to add relinearization key: %v", err)
	}

	for _, rtkBytes := range req.RotationKeys {
		if err := ser.AddRotationKeyFromBytes(s.rtkSet, rtkBytes, s.mkParams.Parameters); err != nil {
			return nil, fmt.Errorf("failed to add rotation key: %v", err)
		}
	}

	model := &common.EncryptedModel{
		Conv1Weights: conv1Weights,
		Conv1Bias:    conv1Bias,
		Conv2Weights: conv2Weights,
		Conv2Bias:    conv2Bias,
		FC1Weights:   fc1Weights,
		FC1Bias:      fc1Bias,
		FC2Weights:   fc2Weights,
		FC2Bias:      fc2Bias,
	}

	if err := s.saveModelToDisk(req.OwnerId, model); err != nil {
		return nil, fmt.Errorf("failed to save model to disk: %v", err)
	}

	if err := s.saveKeysToDisk(req.OwnerId, req.PublicKey, req.RelinearizationKey, req.RotationKeys); err != nil {
		return nil, fmt.Errorf("failed to save keys to disk: %v", err)
	}

	return &pb.EnrollModelResponse{
		Success: true,
		Message: fmt.Sprintf("Successfully enrolled model for %s", req.OwnerId),
	}, nil
}

func (s *CSPServer) RequestInference(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	transferLatency := time.Duration(time.Now().UnixNano() - req.RequestStartTime)
	s.timing.ClientTransferStats.AddSample(transferLatency)

	clientKeys := newOwnerKeySet(s.mkParams)
	defer func() {
		clientKeys.pkSet = nil
		clientKeys.rlkSet = nil
		clientKeys.rtkSet = nil
	}()

	if err := ser.AddPublicKeyFromBytes(clientKeys.pkSet, req.PublicKey, s.mkParams.Parameters); err != nil {
		return nil, fmt.Errorf("failed to add client public key: %v", err)
	}

	if err := ser.AddRelinKeyFromBytes(clientKeys.rlkSet, req.RelinearizationKey, s.mkParams.Parameters); err != nil {
		return nil, fmt.Errorf("failed to add client relinearization key: %v", err)
	}

	for _, rtkBytes := range req.RotationKeys {
		if err := ser.AddRotationKeyFromBytes(clientKeys.rtkSet, rtkBytes, s.mkParams.Parameters); err != nil {
			return nil, fmt.Errorf("failed to add client rotation key: %v", err)
		}
	}

	var encInput [6]*mkckks.Ciphertext
	for i := 0; i < 6; i++ {
		ct, err := ser.DeserializeCiphertext(req.EncryptedInput[i], s.mkParams)
		if err != nil {
			return nil, fmt.Errorf("failed to deserialize input %d: %v", i, err)
		}
		encInput[i] = ct
	}

	computeStart := time.Now()
	inferenceStart := time.Now()

	ownerIDs := make([]string, 0, len(s.dataOwnerClients))
	for ownerID := range s.dataOwnerClients {
		ownerIDs = append(ownerIDs, ownerID)
	}
	sort.Strings(ownerIDs)

	firstOwnerID := ownerIDs[0]
	ownerKeys := newOwnerKeySet(s.mkParams)
	if clientPK := clientKeys.pkSet.GetPublicKey("client"); clientPK != nil {
		ownerKeys.pkSet.AddPublicKey(clientPK)
	}
	ownerKeys.rlkSet = clientKeys.rlkSet
	ownerKeys.rtkSet = clientKeys.rtkSet

	model, err := s.loadModelFromDisk(firstOwnerID)
	if err != nil {
		return nil, fmt.Errorf("failed to load model for owner %s: %v", firstOwnerID, err)
	}

	keys, err := s.loadKeysFromDisk(firstOwnerID)
	if err != nil {
		return nil, fmt.Errorf("failed to load keys for owner %s: %v", firstOwnerID, err)
	}

	if err := ser.AddPublicKeyFromBytes(ownerKeys.pkSet, keys.PublicKey, s.mkParams.Parameters); err != nil {
		return nil, fmt.Errorf("failed to add owner public key: %v", err)
	}

	if err := ser.AddRelinKeyFromBytes(ownerKeys.rlkSet, keys.RelinearizationKey, s.mkParams.Parameters); err != nil {
		return nil, fmt.Errorf("failed to add owner relinearization key: %v", err)
	}

	for _, rtkBytes := range keys.RotationKeys {
		if err := ser.AddRotationKeyFromBytes(ownerKeys.rtkSet, rtkBytes, s.mkParams.Parameters); err != nil {
			return nil, fmt.Errorf("failed to add owner rotation key: %v", err)
		}
	}

	ensembleResult := s.performInferenceWithKeys(encInput, model, ownerKeys)
	model = nil
	ownerKeys = nil
	runtime.GC()

	for _, ownerID := range ownerIDs[1:] {
		ownerKeys = newOwnerKeySet(s.mkParams)
		if clientPK := clientKeys.pkSet.GetPublicKey("client"); clientPK != nil {
			ownerKeys.pkSet.AddPublicKey(clientPK)
		}
		ownerKeys.rlkSet = clientKeys.rlkSet
		ownerKeys.rtkSet = clientKeys.rtkSet

		model, err := s.loadModelFromDisk(ownerID)
		if err != nil {
			return nil, fmt.Errorf("failed to load model for owner %s: %v", ownerID, err)
		}

		keys, err := s.loadKeysFromDisk(ownerID)
		if err != nil {
			return nil, fmt.Errorf("failed to load keys for owner %s: %v", ownerID, err)
		}

		if err := ser.AddPublicKeyFromBytes(ownerKeys.pkSet, keys.PublicKey, s.mkParams.Parameters); err != nil {
			return nil, fmt.Errorf("failed to add owner public key: %v", err)
		}

		if err := ser.AddRelinKeyFromBytes(ownerKeys.rlkSet, keys.RelinearizationKey, s.mkParams.Parameters); err != nil {
			return nil, fmt.Errorf("failed to add owner relinearization key: %v", err)
		}

		for _, rtkBytes := range keys.RotationKeys {
			if err := ser.AddRotationKeyFromBytes(ownerKeys.rtkSet, rtkBytes, s.mkParams.Parameters); err != nil {
				return nil, fmt.Errorf("failed to add owner rotation key: %v", err)
			}
		}

		result := s.performInferenceWithKeys(encInput, model, ownerKeys)
		ensembleResult = s.evaluator.AddNew(ensembleResult, result)

		result = nil
		model = nil
		ownerKeys = nil
		runtime.GC()
	}

	for i := range encInput {
		encInput[i] = nil
	}
	runtime.GC()

	s.timing.InferenceStats.AddSample(time.Since(inferenceStart))
	s.timing.TotalComputeStats.AddSample(time.Since(computeStart))

	decryptionStart := time.Now().UnixNano()
	ensembleDecrypted := ensembleResult.CopyNew()
	ensembleResult = nil
	runtime.GC()

	for _, ownerID := range ownerIDs {
		resultBytes, err := ser.SerializeCiphertext(ensembleDecrypted)
		if err != nil {
			return nil, fmt.Errorf("failed to serialize result: %v", err)
		}

		partialDecBytes, err := s.dataOwnerClients[ownerID].PerformPartialDecryption(ctx, &pb.PartialDecryptionRequest{
			PartyId:          ownerID,
			EncryptedResult:  resultBytes,
			RequestStartTime: time.Now().UnixNano(),
		})
		if err != nil {
			return nil, fmt.Errorf("failed to get partial decryption from owner %s: %v", ownerID, err)
		}

		partialDec, err := ser.DeserializeCiphertext(partialDecBytes.PartialDecryption, s.mkParams)
		if err != nil {
			return nil, fmt.Errorf("failed to deserialize partial decryption: %v", err)
		}
		ensembleDecrypted.Value = partialDec.Value
		partialDec = nil
	}

	resultBytes, err := ser.SerializeCiphertext(ensembleDecrypted)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize results: %v", err)
	}

	ensembleDecrypted = nil
	runtime.GC()

	return &pb.InferenceResponse{
		EncryptedResult:             resultBytes,
		DecryptionProtocolStartTime: decryptionStart,
	}, nil
}

func (s *CSPServer) ConnectToDataOwner(ownerID string, address string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	maxSize := 1024 * 1024 * 1024 * 4
	conn, err := grpc.Dial(
		address,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(maxSize),
			grpc.MaxCallSendMsgSize(maxSize),
		),
	)
	if err != nil {
		return fmt.Errorf("failed to connect to data owner %s: %v", ownerID, err)
	}

	client := pb.NewDataOwnerServiceClient(conn)
	s.dataOwnerConns[ownerID] = conn
	s.dataOwnerClients[ownerID] = client

	return nil
}

func (s *CSPServer) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	var errs []error
	for ownerID, conn := range s.dataOwnerConns {
		if err := conn.Close(); err != nil {
			errs = append(errs, fmt.Errorf("failed to close connection to data owner %s: %v", ownerID, err))
		}
	}

	s.dataOwnerConns = make(map[string]*grpc.ClientConn)
	s.dataOwnerClients = make(map[string]pb.DataOwnerServiceClient)
	s.encryptedModels = make(map[string]*common.EncryptedModel)

	if len(errs) > 0 {
		return fmt.Errorf("errors while closing CSP server: %v", errs)
	}
	return nil
}

func (s *CSPServer) DisconnectFromDataOwner(ownerID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if conn, exists := s.dataOwnerConns[ownerID]; exists {
		if err := conn.Close(); err != nil {
			return fmt.Errorf("failed to close connection to data owner %s: %v", ownerID, err)
		}
		delete(s.dataOwnerConns, ownerID)
		delete(s.dataOwnerClients, ownerID)
	}
	return nil
}

func (s *CSPServer) DisconnectAllDataOwners() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for ownerID, conn := range s.dataOwnerConns {
		if err := conn.Close(); err != nil {
			return fmt.Errorf("failed to close connection to data owner %s: %v", ownerID, err)
		}
	}

	s.dataOwnerConns = make(map[string]*grpc.ClientConn)
	s.dataOwnerClients = make(map[string]pb.DataOwnerServiceClient)

	return nil
}

func conv1Operation(input [6]*mkckks.Ciphertext,
	weightEncodings [6]*mkckks.Ciphertext,
	biasEncoding *mkckks.Ciphertext,
	evaluator *mkckks.Evaluator,
	rlkSet *mkrlwe.RelinearizationKeySet,
	rtkSet *mkrlwe.RotationKeySet) (*mkckks.Ciphertext, error) {

	if len(input) != 6 || len(weightEncodings) != 6 {
		return nil, fmt.Errorf("invalid input dimensions: expected 6 encodings")
	}

	multiplicationResults := make([]*mkckks.Ciphertext, 6)
	for i := 0; i < 6; i++ {
		if input[i] == nil || weightEncodings[i] == nil {
			return nil, fmt.Errorf("nil encoding found at index %d", i)
		}
		multiplicationResults[i] = evaluator.MulRelinNew(input[i], weightEncodings[i], rlkSet)
	}

	result := multiplicationResults[0]
	for i := 1; i < 6; i++ {
		result = evaluator.AddNew(result, multiplicationResults[i])
	}

	rotated := evaluator.RotateNew(result, 16384, rtkSet)
	result = evaluator.AddNew(result, rotated)

	if biasEncoding == nil {
		return nil, fmt.Errorf("bias encoding is nil")
	}
	return evaluator.AddNew(result, biasEncoding), nil
}

func conv2Operation(input *mkckks.Ciphertext,
	weightEncodings [64]*mkckks.Ciphertext,
	biasEncoding *mkckks.Ciphertext,
	evaluator *mkckks.Evaluator,
	encryptor *mkckks.Encryptor,
	pkSet *mkrlwe.PublicKeySet,
	rlkSet *mkrlwe.RelinearizationKeySet,
	rtkSet *mkrlwe.RotationKeySet,
	params mkckks.Parameters) (*mkckks.Ciphertext, error) {

	if input == nil || biasEncoding == nil {
		return nil, fmt.Errorf("input or bias is nil")
	}

	result := encryptZero(params, encryptor, pkSet, "client")

	rotatedInputs := make([]*mkckks.Ciphertext, 8)
	inputHoisted := evaluator.HoistedForm(input)
	for i := 0; i < 8; i++ {
		rotatedInputs[i] = evaluator.RotateHoistedNew(input, 256*i, inputHoisted, rtkSet)
	}

	for i := 0; i < 8; i++ {
		intermediate := encryptZero(params, encryptor, pkSet, "client")

		for j := 0; j < 8; j++ {
			weightIdx := 8*i + j
			temp := evaluator.MulRelinNew(rotatedInputs[j], weightEncodings[weightIdx], rlkSet)
			intermediate = evaluator.AddNew(intermediate, temp)
		}

		rotated := evaluator.RotateNew(intermediate, 256*8*i, rtkSet)
		result = evaluator.AddNew(result, rotated)
	}

	rotated128 := evaluator.RotateNew(result, 128, rtkSet)
	result = evaluator.AddNew(result, rotated128)
	rotated64 := evaluator.RotateNew(result, 64, rtkSet)
	result = evaluator.AddNew(result, rotated64)
	result = evaluator.AddNew(result, biasEncoding)

	maskMsg := mkckks.NewMessage(params)
	for i := 0; i < 128; i++ {
		for j := 0; j < 64; j++ {
			maskMsg.Value[i*256+j] = complex(1.0, 0)
		}
	}
	mask := encryptor.EncodeMsgNew(maskMsg)
	result = evaluator.MulPtxtNew(result, mask)

	rotatedNeg64 := evaluator.RotateNew(result, -64, rtkSet)
	result = evaluator.AddNew(result, rotatedNeg64)
	rotatedNeg128 := evaluator.RotateNew(result, -128, rtkSet)
	result = evaluator.AddNew(result, rotatedNeg128)

	return result, nil
}

func fc1Operation(input *mkckks.Ciphertext,
	weightEncodings [16]*mkckks.Ciphertext,
	biasEncoding *mkckks.Ciphertext,
	evaluator *mkckks.Evaluator,
	encryptor *mkckks.Encryptor,
	pkSet *mkrlwe.PublicKeySet,
	rlkSet *mkrlwe.RelinearizationKeySet,
	rtkSet *mkrlwe.RotationKeySet,
	params mkckks.Parameters) (*mkckks.Ciphertext, error) {

	if input == nil || biasEncoding == nil {
		return nil, fmt.Errorf("input or bias is nil")
	}

	result := encryptZero(params, encryptor, pkSet, "client")

	rotatedInputs := make([]*mkckks.Ciphertext, 4)
	inputHoisted := evaluator.HoistedForm(input)
	for i := 0; i < 4; i++ {
		rotatedInputs[i] = evaluator.RotateHoistedNew(input, 256*i, inputHoisted, rtkSet)
	}

	for i := 0; i < 4; i++ {
		intermediate := encryptZero(params, encryptor, pkSet, "client")

		for j := 0; j < 4; j++ {
			weightIdx := i*4 + j
			temp := evaluator.MulRelinNew(rotatedInputs[j], weightEncodings[weightIdx], rlkSet)
			intermediate = evaluator.AddNew(intermediate, temp)
		}

		rotated := evaluator.RotateNew(intermediate, 256*4*i, rtkSet)
		result = evaluator.AddNew(result, rotated)
	}

	for _, shift := range []int{16384, 8192, 4096} {
		rotated := evaluator.RotateNew(result, shift, rtkSet)
		result = evaluator.AddNew(result, rotated)
	}

	for _, shift := range []int{32, 16, 8, 4, 2, 1} {
		rotated := evaluator.RotateNew(result, shift, rtkSet)
		result = evaluator.AddNew(result, rotated)
	}

	result = evaluator.AddNew(result, biasEncoding)

	maskMsg := mkckks.NewMessage(params)
	for i := 0; i < 512; i++ {
		maskMsg.Value[i*64] = complex(1.0, 0)
	}
	mask := encryptor.EncodeMsgNew(maskMsg)
	result = evaluator.MulPtxtNew(result, mask)

	rotatedNeg1 := evaluator.RotateNew(result, -1, rtkSet)
	result = evaluator.AddNew(result, rotatedNeg1)

	return result, nil
}

func fc2Operation(input *mkckks.Ciphertext,
	weightEncoding *mkckks.Ciphertext,
	biasEncoding *mkckks.Ciphertext,
	evaluator *mkckks.Evaluator,
	encryptor *mkckks.Encryptor,
	rlkSet *mkrlwe.RelinearizationKeySet,
	rtkSet *mkrlwe.RotationKeySet,
	params mkckks.Parameters) (*mkckks.Ciphertext, error) {

	if input == nil || weightEncoding == nil || biasEncoding == nil {
		return nil, fmt.Errorf("one or more inputs are nil")
	}

	result := evaluator.MulRelinNew(input, weightEncoding, rlkSet)

	for shift := 64 * 32; shift >= 64; shift = shift / 2 {
		rotated := evaluator.RotateNew(result, shift, rtkSet)
		result = evaluator.AddNew(result, rotated)
	}

	maskMsg := mkckks.NewMessage(params)
	for i := 0; i < 5; i++ {
		maskMsg.Value[i*4096] = complex(1.0, 0)
		maskMsg.Value[i*4096+1] = complex(1.0, 0)
	}
	mask := encryptor.EncodeMsgNew(maskMsg)
	result = evaluator.MulPtxtNew(result, mask)

	result = evaluator.AddNew(result, biasEncoding)

	return result, nil
}

func encryptZero(params mkckks.Parameters, encryptor *mkckks.Encryptor, pkSet *mkrlwe.PublicKeySet, id string) *mkckks.Ciphertext {
	zeroMsg := mkckks.NewMessage(params)
	for i := 0; i < zeroMsg.Slots(); i++ {
		zeroMsg.Value[i] = complex(0.0, 0)
	}
	return encryptor.EncryptMsgNew(zeroMsg, pkSet.GetPublicKey(id))
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// Memory management ////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

func (s *CSPServer) saveModelToDisk(ownerID string, model *common.EncryptedModel) error {
	storedModel := &common.StoredModel{
		Conv1Weights: make([][]byte, 6),
		Conv2Weights: make([][]byte, 64),
		FC1Weights:   make([][]byte, 16),
	}

	for i := 0; i < 6; i++ {
		bytes, err := ser.SerializeCiphertext(model.Conv1Weights[i])
		if err != nil {
			return fmt.Errorf("failed to serialize Conv1 weight %d: %v", i, err)
		}
		storedModel.Conv1Weights[i] = bytes
	}

	var err error
	if storedModel.Conv1Bias, err = ser.SerializeCiphertext(model.Conv1Bias); err != nil {
		return fmt.Errorf("failed to serialize Conv1 bias: %v", err)
	}

	for i := 0; i < 64; i++ {
		bytes, err := ser.SerializeCiphertext(model.Conv2Weights[i])
		if err != nil {
			return fmt.Errorf("failed to serialize Conv2 weight %d: %v", i, err)
		}
		storedModel.Conv2Weights[i] = bytes
	}

	if storedModel.Conv2Bias, err = ser.SerializeCiphertext(model.Conv2Bias); err != nil {
		return fmt.Errorf("failed to serialize Conv2 bias: %v", err)
	}

	for i := 0; i < 16; i++ {
		bytes, err := ser.SerializeCiphertext(model.FC1Weights[i])
		if err != nil {
			return fmt.Errorf("failed to serialize FC1 weight %d: %v", i, err)
		}
		storedModel.FC1Weights[i] = bytes
	}

	if storedModel.FC1Bias, err = ser.SerializeCiphertext(model.FC1Bias); err != nil {
		return fmt.Errorf("failed to serialize FC1 bias: %v", err)
	}

	if storedModel.FC2Weights, err = ser.SerializeCiphertext(model.FC2Weights); err != nil {
		return fmt.Errorf("failed to serialize FC2 weights: %v", err)
	}

	if storedModel.FC2Bias, err = ser.SerializeCiphertext(model.FC2Bias); err != nil {
		return fmt.Errorf("failed to serialize FC2 bias: %v", err)
	}

	modelPath := filepath.Join(s.storageDir, fmt.Sprintf("%s_model.gob", ownerID))
	file, err := os.Create(modelPath)
	if err != nil {
		return fmt.Errorf("failed to create model file: %v", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(storedModel); err != nil {
		return fmt.Errorf("failed to encode model: %v", err)
	}

	return nil
}

func (s *CSPServer) loadModelFromDisk(ownerID string) (*common.EncryptedModel, error) {
	modelPath := filepath.Join(s.storageDir, fmt.Sprintf("%s_model.gob", ownerID))
	file, err := os.Open(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open model file: %v", err)
	}
	defer file.Close()

	var storedModel common.StoredModel
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&storedModel); err != nil {
		return nil, fmt.Errorf("failed to decode model: %v", err)
	}

	model := &common.EncryptedModel{
		Conv1Weights: [6]*mkckks.Ciphertext{},
		Conv2Weights: [64]*mkckks.Ciphertext{},
		FC1Weights:   [16]*mkckks.Ciphertext{},
	}

	for i := 0; i < 6; i++ {
		ct, err := ser.DeserializeCiphertext(storedModel.Conv1Weights[i], s.mkParams)
		if err != nil {
			return nil, fmt.Errorf("failed to deserialize Conv1 weight %d: %v", i, err)
		}
		model.Conv1Weights[i] = ct
	}

	if model.Conv1Bias, err = ser.DeserializeCiphertext(storedModel.Conv1Bias, s.mkParams); err != nil {
		return nil, fmt.Errorf("failed to deserialize Conv1 bias: %v", err)
	}

	for i := 0; i < 64; i++ {
		ct, err := ser.DeserializeCiphertext(storedModel.Conv2Weights[i], s.mkParams)
		if err != nil {
			return nil, fmt.Errorf("failed to deserialize Conv2 weight %d: %v", i, err)
		}
		model.Conv2Weights[i] = ct
	}

	if model.Conv2Bias, err = ser.DeserializeCiphertext(storedModel.Conv2Bias, s.mkParams); err != nil {
		return nil, fmt.Errorf("failed to deserialize Conv2 bias: %v", err)
	}

	for i := 0; i < 16; i++ {
		ct, err := ser.DeserializeCiphertext(storedModel.FC1Weights[i], s.mkParams)
		if err != nil {
			return nil, fmt.Errorf("failed to deserialize FC1 weight %d: %v", i, err)
		}
		model.FC1Weights[i] = ct
	}

	if model.FC1Bias, err = ser.DeserializeCiphertext(storedModel.FC1Bias, s.mkParams); err != nil {
		return nil, fmt.Errorf("failed to deserialize FC1 bias: %v", err)
	}

	if model.FC2Weights, err = ser.DeserializeCiphertext(storedModel.FC2Weights, s.mkParams); err != nil {
		return nil, fmt.Errorf("failed to deserialize FC2 weights: %v", err)
	}

	if model.FC2Bias, err = ser.DeserializeCiphertext(storedModel.FC2Bias, s.mkParams); err != nil {
		return nil, fmt.Errorf("failed to deserialize FC2 bias: %v", err)
	}

	return model, nil
}

func (s *CSPServer) saveKeysToDisk(ownerID string, pk []byte, rlk []byte, rtks [][]byte) error {
	keys := &common.StoredKeys{
		PublicKey:          pk,
		RelinearizationKey: rlk,
		RotationKeys:       rtks,
	}

	keysPath := filepath.Join(s.storageDir, fmt.Sprintf("%s_keys.gob", ownerID))
	file, err := os.Create(keysPath)
	if err != nil {
		return fmt.Errorf("failed to create keys file: %v", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(keys); err != nil {
		return fmt.Errorf("failed to encode keys: %v", err)
	}

	return nil
}

func (s *CSPServer) loadKeysFromDisk(ownerID string) (*common.StoredKeys, error) {
	keysPath := filepath.Join(s.storageDir, fmt.Sprintf("%s_keys.gob", ownerID))
	file, err := os.Open(keysPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open keys file: %v", err)
	}
	defer file.Close()

	var keys common.StoredKeys
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&keys); err != nil {
		return nil, fmt.Errorf("failed to decode keys: %v", err)
	}

	return &keys, nil
}

func (s *CSPServer) performInferenceWithKeys(input [6]*mkckks.Ciphertext, model *common.EncryptedModel, keys *ownerKeySet) *mkckks.Ciphertext {
	conv1Out, _ := conv1Operation(input, model.Conv1Weights, model.Conv1Bias, s.evaluator, keys.rlkSet, keys.rtkSet)
	quad1Out := s.evaluator.MulRelinNew(conv1Out, conv1Out, keys.rlkSet)

	conv2Out, _ := conv2Operation(quad1Out, model.Conv2Weights, model.Conv2Bias, s.evaluator, s.encryptor, keys.pkSet, keys.rlkSet, keys.rtkSet, s.mkParams)
	quad2Out := s.evaluator.MulRelinNew(conv2Out, conv2Out, keys.rlkSet)

	fc1Out, _ := fc1Operation(quad2Out, model.FC1Weights, model.FC1Bias, s.evaluator, s.encryptor, keys.pkSet, keys.rlkSet, keys.rtkSet, s.mkParams)
	fc1Activated := s.evaluator.MulRelinNew(fc1Out, fc1Out, keys.rlkSet)

	result, _ := fc2Operation(fc1Activated, model.FC2Weights, model.FC2Bias, s.evaluator, s.encryptor, keys.rlkSet, keys.rtkSet, s.mkParams)

	return result
}
