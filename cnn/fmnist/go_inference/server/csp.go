package server

import (
	"context"
	"fmt"
	"net"
	"sort"
	"sync"
	"time"

	"cnn/fmnist/go_inference/common"
	pb "cnn/fmnist/go_inference/proto"
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

type EncryptedFMNISTModel struct {
	ConvWeights *mkckks.Ciphertext
	ConvBias    *mkckks.Ciphertext
	FC1Weights  [8]*mkckks.Ciphertext
	FC1Bias     *mkckks.Ciphertext
	FC2Weights  *mkckks.Ciphertext
	FC2Bias     *mkckks.Ciphertext
}

type CSPServer struct {
	pb.UnimplementedCSPServiceServer
	mu sync.RWMutex

	mkParams  mkckks.Parameters
	evaluator *mkckks.Evaluator
	encryptor *mkckks.Encryptor

	timing          CSPTiming
	encryptedModels map[string]*EncryptedFMNISTModel

	pkSet  *mkrlwe.PublicKeySet
	rlkSet *mkrlwe.RelinearizationKeySet
	rtkSet *mkrlwe.RotationKeySet

	dataOwnerClients map[string]pb.DataOwnerServiceClient
	dataOwnerConns   map[string]*grpc.ClientConn

	zeroCache map[string]*mkckks.Ciphertext
}

func NewCSPServer(params mkckks.Parameters) *CSPServer {
	return &CSPServer{
		mkParams:         params,
		evaluator:        mkckks.NewEvaluator(params),
		encryptor:        mkckks.NewEncryptor(params),
		encryptedModels:  make(map[string]*EncryptedFMNISTModel),
		pkSet:            mkrlwe.NewPublicKeyKeySet(),
		rlkSet:           mkrlwe.NewRelinearizationKeySet(params.Parameters),
		rtkSet:           mkrlwe.NewRotationKeySet(),
		dataOwnerClients: make(map[string]pb.DataOwnerServiceClient),
		dataOwnerConns:   make(map[string]*grpc.ClientConn),
		zeroCache:        make(map[string]*mkckks.Ciphertext),
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
			if err != grpc.ErrServerStopped {
				fmt.Printf("CSP server error: %v\n", err)
			}
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

	deserializeEnd := time.Now().UnixNano()
	s.timing.DataOwnerTransferTime = time.Duration(deserializeEnd - req.SerializationStartTime)

	convWeights, err := ser.DeserializeCiphertext(req.ConvWeights, s.mkParams)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize conv weights: %v", err)
	}

	convBias, err := ser.DeserializeCiphertext(req.ConvBias, s.mkParams)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize conv bias: %v", err)
	}

	var fc1Weights [8]*mkckks.Ciphertext
	for i := 0; i < 8; i++ {
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

	s.encryptedModels[req.OwnerId] = &EncryptedFMNISTModel{
		ConvWeights: convWeights,
		ConvBias:    convBias,
		FC1Weights:  fc1Weights,
		FC1Bias:     fc1Bias,
		FC2Weights:  fc2Weights,
		FC2Bias:     fc2Bias,
	}

	return &pb.EnrollModelResponse{
		Success: true,
		Message: fmt.Sprintf("Successfully enrolled model for %s", req.OwnerId),
	}, nil
}

func (s *CSPServer) RequestInference(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	deserializeEnd := time.Now().UnixNano()
	s.timing.ClientTransferStats.AddSample(time.Duration(deserializeEnd - req.SerializationStartTime))

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

	encInput, err := ser.DeserializeCiphertext(req.EncryptedInput, s.mkParams)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize input: %v", err)
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
		ensembleResultBytes, err := ser.SerializeCiphertext(ensembleDecrypted)
		if err != nil {
			return nil, fmt.Errorf("failed to serialize result: %v", err)
		}

		serializationStart := time.Now().UnixNano()
		partialDecBytes, err := s.dataOwnerClients[ownerID].PerformPartialDecryption(ctx, &pb.PartialDecryptionRequest{
			PartyId:                ownerID,
			EncryptedResult:        ensembleResultBytes,
			SerializationStartTime: serializationStart,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to get partial decryption from owner %s: %v", ownerID, err)
		}
		partialDec, err := ser.DeserializeCiphertext(partialDecBytes.PartialDecryption, s.mkParams)
		if err != nil {
			return nil, fmt.Errorf("failed to deserialize partial decryption: %v", err)
		}
		ensembleDecrypted.Ciphertext.Value = partialDec.Value
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

func (s *CSPServer) performInference(input *mkckks.Ciphertext, model *EncryptedFMNISTModel) *mkckks.Ciphertext {
	convOutput := s.convLayer(input, model.ConvWeights, model.ConvBias)
	convSquared := s.evaluator.MulRelinNew(convOutput, convOutput, s.rlkSet)

	layer1Output := s.fc1Layer(convSquared, model.FC1Weights, model.FC1Bias)
	layer1Squared := s.evaluator.MulRelinNew(layer1Output, layer1Output, s.rlkSet)

	return s.fc2Layer(layer1Squared, model.FC2Weights, model.FC2Bias)
}

func (s *CSPServer) performEnsemble(predictions []*mkckks.Ciphertext) *mkckks.Ciphertext {
	result := predictions[0]
	for i := 1; i < len(predictions); i++ {
		result = s.evaluator.AddNew(result, predictions[i])
	}
	return result
}

func (s *CSPServer) convLayer(input *mkckks.Ciphertext, weights *mkckks.Ciphertext, bias *mkckks.Ciphertext) *mkckks.Ciphertext {
	result := s.evaluator.MulRelinNew(input, weights, s.rlkSet)

	rotated1024 := s.evaluator.RotateNew(result, 1024, s.rtkSet)
	result = s.evaluator.AddNew(result, rotated1024)

	rotated2048 := s.evaluator.RotateNew(result, 2048, s.rtkSet)
	result = s.evaluator.AddNew(result, rotated2048)

	return s.evaluator.AddNew(result, bias)
}

func (s *CSPServer) fc1Layer(input *mkckks.Ciphertext, weights [8]*mkckks.Ciphertext, bias *mkckks.Ciphertext) *mkckks.Ciphertext {
	var babySteps [2]*mkckks.Ciphertext
	for i := 0; i < 2; i++ {
		babySteps[i] = s.evaluator.RotateNew(input, i, s.rtkSet)
	}

	result := s.encryptZero("client")
	for i := 0; i < 8; i++ {
		rots := s.vecRotsOpt1(babySteps[:], i, 2, 8, 8192)
		partialResult := s.evaluator.MulRelinNew(weights[i], rots, s.rlkSet)
		result = s.evaluator.AddNew(result, partialResult)
	}

	for i := 1; i <= 7; i++ {
		rotated := s.evaluator.RotateNew(result, 8192/(1<<i), s.rtkSet)
		result = s.evaluator.AddNew(result, rotated)
	}

	return s.evaluator.AddNew(result, bias)
}

func (s *CSPServer) fc2Layer(input *mkckks.Ciphertext, weights *mkckks.Ciphertext, bias *mkckks.Ciphertext) *mkckks.Ciphertext {
	inputHoisted := s.evaluator.HoistedForm(input)
	var babySteps [4]*mkckks.Ciphertext
	for i := 0; i < 4; i++ {
		babySteps[i] = s.evaluator.RotateHoistedNew(input, i, inputHoisted, s.rtkSet)
	}

	rots := s.vecRotsOpt2(babySteps[:], 0, 4, 16, 8192)
	result := s.evaluator.MulRelinNew(weights, rots, s.rlkSet)

	for i := 1; i <= 6; i++ {
		rotated := s.evaluator.RotateNew(result, 1024/(1<<i), s.rtkSet)
		result = s.evaluator.AddNew(result, rotated)
	}

	return s.evaluator.AddNew(result, bias)
}

func (s *CSPServer) vecRotsOpt1(preRotatedArrays []*mkckks.Ciphertext, is int, np int, stride int, numSlots int) *mkckks.Ciphertext {
	result := s.encryptZero("client")

	for j := 0; j < stride/np; j++ {
		T := s.encryptZero("client")

		for i := 0; i < np; i++ {
			msk := generateMaskVector(numSlots, np*j+i, 1024)
			msk = vectorRotate(msk, -is*stride-j*np)

			msg := &mkckks.Message{Value: msk}
			pmsk := s.encryptor.EncodeMsgNew(msg)

			masked := s.evaluator.MulPtxtNew(preRotatedArrays[i], pmsk)
			T = s.evaluator.AddNew(T, masked)
		}
		rotated := s.evaluator.RotateNew(T, is*stride+j*np, s.rtkSet)
		result = s.evaluator.AddNew(result, rotated)
	}

	return result
}

func (s *CSPServer) vecRotsOpt2(preRotatedArrays []*mkckks.Ciphertext, is int, np int, stride int, numSlots int) *mkckks.Ciphertext {
	result := s.encryptZero("client")

	for j := 0; j < stride/np; j++ {
		T := s.encryptZero("client")

		for i := 0; i < np; i++ {
			msk := generateRepeatedMaskVector(numSlots, 1024, 64, np*j+i)
			msk = vectorRotate(msk, -is*stride-j*np)

			msg := &mkckks.Message{Value: msk}
			pmsk := s.encryptor.EncodeMsgNew(msg)

			masked := s.evaluator.MulPtxtNew(preRotatedArrays[i], pmsk)
			T = s.evaluator.AddNew(T, masked)
		}
		rotated := s.evaluator.RotateNew(T, is*stride+j*np, s.rtkSet)
		result = s.evaluator.AddNew(result, rotated)
	}

	return result
}

func (s *CSPServer) encryptZero(id string) *mkckks.Ciphertext {
	if ct, exists := s.zeroCache[id]; exists {
		return ct.CopyNew()
	}

	msg := mkckks.NewMessage(s.mkParams)
	for i := 0; i < msg.Slots(); i++ {
		msg.Value[i] = complex(0.0, 0)
	}
	ct := s.encryptor.EncryptMsgNew(msg, s.pkSet.GetPublicKey(id))
	s.zeroCache[id] = ct.CopyNew()
	return ct
}

func generateMaskVector(batchSize int, k int, N int) []complex128 {
	result := make([]complex128, batchSize)
	for i := k * N; i < (k+1)*N; i++ {
		result[i] = complex(1.0, 0)
	}
	return result
}

func generateRepeatedMaskVector(totalBatchSize int, subBatchSize int, N int, k int) []complex128 {
	result := make([]complex128, totalBatchSize)
	subMask := make([]complex128, subBatchSize)

	for i := k * N; i < (k+1)*N; i++ {
		subMask[i] = complex(1.0, 0)
	}

	for i := 0; i < totalBatchSize; i += subBatchSize {
		copy(result[i:i+subBatchSize], subMask)
	}

	return result
}

func vectorRotate(vec []complex128, rotateIndex int) []complex128 {
	n := len(vec)
	if n == 0 {
		return []complex128{}
	}

	result := make([]complex128, n)
	copy(result, vec)

	if rotateIndex < 0 {
		rotateIndex = rotateIndex + n
	}
	rotateIndex = rotateIndex % n

	temp := make([]complex128, rotateIndex)
	copy(temp, result[:rotateIndex])
	copy(result, result[rotateIndex:])
	copy(result[n-rotateIndex:], temp)

	return result
}

func (s *CSPServer) ConnectToDataOwner(ownerID string, address string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	maxSize := 1024 * 1024 * 1024 * 8 // 8GB
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
	s.encryptedModels = make(map[string]*EncryptedFMNISTModel)
	s.zeroCache = make(map[string]*mkckks.Ciphertext)

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
		delete(s.encryptedModels, ownerID)
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
	s.encryptedModels = make(map[string]*EncryptedFMNISTModel)
	s.zeroCache = make(map[string]*mkckks.Ciphertext)

	return nil
}
