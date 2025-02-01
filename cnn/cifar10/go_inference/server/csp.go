package server

import (
	"context"
	"fmt"
	"net"
	"sort"
	"sync"
	"time"

	"cnn/cifar10/go_inference/common"
	pb "cnn/cifar10/go_inference/proto"
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
	encryptedModels map[string]*common.EncryptedModel

	pkSet  *mkrlwe.PublicKeySet
	rlkSet *mkrlwe.RelinearizationKeySet
	rtkSet *mkrlwe.RotationKeySet

	dataOwnerClients map[string]pb.DataOwnerServiceClient
	dataOwnerConns   map[string]*grpc.ClientConn
}

func NewCSPServer(params mkckks.Parameters) *CSPServer {
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

	// Deserialize Conv1 weights
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

	// Deserialize Conv2 weights
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

	// Deserialize FC1 weights
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

	s.encryptedModels[req.OwnerId] = &common.EncryptedModel{
		Conv1Weights: conv1Weights,
		Conv1Bias:    conv1Bias,
		Conv2Weights: conv2Weights,
		Conv2Bias:    conv2Bias,
		FC1Weights:   fc1Weights,
		FC1Bias:      fc1Bias,
		FC2Weights:   fc2Weights,
		FC2Bias:      fc2Bias,
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

	if err := ser.AddPublicKeyFromBytes(s.pkSet, req.PublicKey, s.mkParams.Parameters); err != nil {
		return nil, fmt.Errorf("failed to add client public key: %v", err)
	}

	if err := ser.AddRelinKeyFromBytes(s.rlkSet, req.RelinearizationKey, s.mkParams.Parameters); err != nil {
		return nil, fmt.Errorf("failed to add client relinearization key: %v", err)
	}

	for _, rtkBytes := range req.RotationKeys {
		if err := ser.AddRotationKeyFromBytes(s.rtkSet, rtkBytes, s.mkParams.Parameters); err != nil {
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

	predictions := make([]*mkckks.Ciphertext, 0, len(s.encryptedModels))
	for _, model := range s.encryptedModels {
		result := s.performInference(encInput, model)
		predictions = append(predictions, result)
	}
	s.timing.InferenceStats.AddSample(time.Since(inferenceStart))

	ensembleStart := time.Now()
	ensembleResult := s.performEnsemble(predictions)
	s.timing.EnsembleStats.AddSample(time.Since(ensembleStart))
	s.timing.TotalComputeStats.AddSample(time.Since(computeStart))

	decryptionStart := time.Now().UnixNano()
	ensembleDecrypted := ensembleResult.CopyNew()

	ownerIDs := make([]string, 0, len(s.encryptedModels))
	for ownerID := range s.encryptedModels {
		ownerIDs = append(ownerIDs, ownerID)
	}
	sort.Strings(ownerIDs)

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
	}

	resultBytes, err := ser.SerializeCiphertext(ensembleDecrypted)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize results: %v", err)
	}

	return &pb.InferenceResponse{
		EncryptedResult:             resultBytes,
		DecryptionProtocolStartTime: decryptionStart,
	}, nil
}

func (s *CSPServer) performInference(input [6]*mkckks.Ciphertext, model *common.EncryptedModel) *mkckks.Ciphertext {
	// Existing inference logic from cifar10 implementation
	conv1Out, _ := conv1Operation(input, model.Conv1Weights, model.Conv1Bias, s.evaluator, s.rlkSet, s.rtkSet)
	quad1Out := s.evaluator.MulRelinNew(conv1Out, conv1Out, s.rlkSet)

	conv2Out, _ := conv2Operation(quad1Out, model.Conv2Weights, model.Conv2Bias, s.evaluator, s.encryptor, s.pkSet, s.rlkSet, s.rtkSet, s.mkParams)
	quad2Out := s.evaluator.MulRelinNew(conv2Out, conv2Out, s.rlkSet)

	fc1Out, _ := fc1Operation(quad2Out, model.FC1Weights, model.FC1Bias, s.evaluator, s.encryptor, s.pkSet, s.rlkSet, s.rtkSet, s.mkParams)
	fc1Activated := s.evaluator.MulRelinNew(fc1Out, fc1Out, s.rlkSet)

	result, _ := fc2Operation(fc1Activated, model.FC2Weights, model.FC2Bias, s.evaluator, s.encryptor, s.rlkSet, s.rtkSet, s.mkParams)

	return result
}

func (s *CSPServer) performEnsemble(predictions []*mkckks.Ciphertext) *mkckks.Ciphertext {
	result := predictions[0]
	for i := 1; i < len(predictions); i++ {
		result = s.evaluator.AddNew(result, predictions[i])
	}
	return result
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
