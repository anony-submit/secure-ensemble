package server

import (
	"context"
	"fmt"
	"math/bits"
	"sort"
	"sync"
	"time"

	pb "logistic_regression/go_inference/proto"
	"secure-ensemble/pkg/activation"
	"secure-ensemble/pkg/logistic"
	"secure-ensemble/pkg/serialization"

	"github.com/anony-submit/snu-mghe/mkckks"
	"github.com/anony-submit/snu-mghe/mkrlwe"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type CSPTiming struct {
	SoftVotingCompute      time.Duration
	LogitSoftVotingCompute time.Duration
	ClientDataTransfer     time.Duration
	ModelTransferTime      time.Duration
}

type ScalingConfig struct {
	SoftVoting      float64
	LogitSoftVoting float64
}

var datasetScaling = map[string]ScalingConfig{
	"wdbc": {
		SoftVoting:      25.0,
		LogitSoftVoting: 50.0,
	},
	"heart_disease": {
		SoftVoting:      5.0,
		LogitSoftVoting: 20.0,
	},
	"pima": {
		SoftVoting:      5.0,
		LogitSoftVoting: 20.0,
	},
}

type EncryptedModel struct {
	EncWeights   *mkckks.Ciphertext
	EncIntercept *mkckks.Ciphertext
}

type CSPServer struct {
	pb.UnimplementedCSPServiceServer
	mu sync.RWMutex

	// Crypto parameters
	mkParams  mkckks.Parameters
	evaluator *mkckks.Evaluator
	encryptor *mkckks.Encryptor
	config    logistic.BatchConfig
	dataSet   string

	// Timing measurements
	timing CSPTiming

	// Enrolled models and results
	encryptedModels map[string]*EncryptedModel
	pendingResults  map[string]*mkckks.Ciphertext

	// Key sets for all parties
	pkSet  *mkrlwe.PublicKeySet
	rlkSet *mkrlwe.RelinearizationKeySet
	rtkSet *mkrlwe.RotationKeySet
	skSet  *mkrlwe.SecretKeySet

	// Data Owner connections
	dataOwnerClients map[string]pb.DataOwnerServiceClient
	dataOwnerConns   map[string]*grpc.ClientConn
}

func (s *CSPServer) GetTiming() CSPTiming {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.timing
}

func NewCSPServer(params mkckks.Parameters, config logistic.BatchConfig, dataSet string) *CSPServer {
	return &CSPServer{
		mkParams:         params,
		evaluator:        mkckks.NewEvaluator(params),
		encryptor:        mkckks.NewEncryptor(params),
		config:           config,
		dataSet:          dataSet,
		encryptedModels:  make(map[string]*EncryptedModel),
		pendingResults:   make(map[string]*mkckks.Ciphertext),
		pkSet:            mkrlwe.NewPublicKeyKeySet(),
		rlkSet:           mkrlwe.NewRelinearizationKeySet(params.Parameters),
		rtkSet:           mkrlwe.NewRotationKeySet(),
		skSet:            mkrlwe.NewSecretKeySet(),
		dataOwnerClients: make(map[string]pb.DataOwnerServiceClient),
		dataOwnerConns:   make(map[string]*grpc.ClientConn),
	}
}

func (s *CSPServer) ConnectToDataOwner(ownerID string, address string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.dataOwnerClients[ownerID]; exists {
		return fmt.Errorf("connection to data owner %s already exists", ownerID)
	}

	// Setup gRPC connection with appropriate options
	maxSize := 1024 * 1024 * 1024 * 4 // 4GB
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

func (s *CSPServer) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	var errs []error
	for ownerID, conn := range s.dataOwnerConns {
		if err := conn.Close(); err != nil {
			errs = append(errs, fmt.Errorf("failed to close connection to data owner %s: %v", ownerID, err))
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors while closing CSP server: %v", errs)
	}
	return nil
}

func (s *CSPServer) EnrollModel(ctx context.Context, req *pb.EnrollModelRequest) (*pb.EnrollModelResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// fmt.Printf("[CSP] Starting model enrollment for owner %s\n", req.OwnerId)

	deserializeEnd := time.Now().UnixNano()
	s.timing.ModelTransferTime = time.Duration(deserializeEnd - req.SerializationStartTime)

	encWeights, err := serialization.DeserializeCiphertext(req.EncryptedWeights, s.mkParams)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize weights: %v", err)
	}

	encIntercept, err := serialization.DeserializeCiphertext(req.EncryptedIntercept, s.mkParams)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize intercept: %v", err)
	}

	s.encryptedModels[req.OwnerId] = &EncryptedModel{
		EncWeights:   encWeights,
		EncIntercept: encIntercept,
	}

	if err := serialization.AddPublicKeyFromBytes(s.pkSet, req.PublicKey, s.mkParams.Parameters); err != nil {
		return nil, fmt.Errorf("failed to add public key: %v", err)
	}

	if err := serialization.AddRelinKeyFromBytes(s.rlkSet, req.RelinearizationKey, s.mkParams.Parameters); err != nil {
		return nil, fmt.Errorf("failed to add relinearization key: %v", err)
	}

	for i, rtkBytes := range req.RotationKeys {
		if err := serialization.AddRotationKeyFromBytes(s.rtkSet, rtkBytes, s.mkParams.Parameters); err != nil {
			return nil, fmt.Errorf("failed to add rotation key %d: %v", i, err)
		}
	}

	// fmt.Printf("[CSP] Model enrollment completed for owner %s\n", req.OwnerId)
	return &pb.EnrollModelResponse{
		Success: true,
		Message: fmt.Sprintf("Successfully enrolled model for %s", req.OwnerId),
	}, nil
}

func (s *CSPServer) applyLogistic(result *mkckks.Ciphertext, scaling float64) (*mkckks.Ciphertext, error) {
	// Apply scaling
	s.evaluator.MultByConst(result, complex(1.0/scaling, 0), result)
	if err := s.evaluator.Rescale(result, s.mkParams.Scale(), result); err != nil {
		return nil, fmt.Errorf("rescale failed after scaling down: %w", err)
	}

	// Create constant one for logistic function
	constMsg := mkckks.NewMessage(s.mkParams)
	for i := 0; i < constMsg.Slots(); i++ {
		constMsg.Value[i] = complex(1.0, 0)
	}
	constOne := s.encryptor.EncryptMsgNew(constMsg, s.pkSet.GetPublicKey("client"))

	return activation.EvalLogistic(result, constOne, s.mkParams, s.evaluator, s.rlkSet)
}

func (s *CSPServer) performSoftVoting(encTestData *mkckks.Ciphertext) (*mkckks.Ciphertext, error) {
	// Phase 1: Perform inference for each model
	results := make([]*mkckks.Ciphertext, 0, len(s.encryptedModels))
	for _, model := range s.encryptedModels {
		// Perform basic inference (linear + rotations + intercept)
		result, err := s.performInference(model, encTestData)
		if err != nil {
			return nil, fmt.Errorf("inference failed: %v", err)
		}

		// Apply scaling and logistic
		scaling := datasetScaling[s.dataSet].SoftVoting
		result, err = s.applyLogistic(result, scaling)
		if err != nil {
			return nil, fmt.Errorf("soft voting logistic failed: %v", err)
		}

		results = append(results, result)
	}

	// Phase 2: Ensemble the results
	if len(results) == 0 {
		return nil, fmt.Errorf("no models available for ensemble")
	}

	finalResult := results[0]
	for i := 1; i < len(results); i++ {
		finalResult = s.evaluator.AddNew(finalResult, results[i])
	}

	// Average the results
	s.evaluator.MultByConst(finalResult, complex(1.0/float64(len(results)), 0), finalResult)
	if err := s.evaluator.Rescale(finalResult, s.mkParams.Scale(), finalResult); err != nil {
		return nil, fmt.Errorf("rescale failed after averaging: %w", err)
	}

	return finalResult, nil
}

func (s *CSPServer) performLogitSoftVoting(encTestData *mkckks.Ciphertext) (*mkckks.Ciphertext, error) {
	// Phase 1: Perform inference for each model
	results := make([]*mkckks.Ciphertext, 0, len(s.encryptedModels))
	for _, model := range s.encryptedModels {
		// Perform basic inference (linear + rotations + intercept)
		result, err := s.performInference(model, encTestData)
		if err != nil {
			return nil, fmt.Errorf("inference failed: %v", err)
		}
		results = append(results, result)
	}

	// Phase 2: Sum all results
	if len(results) == 0 {
		return nil, fmt.Errorf("no models available for ensemble")
	}

	sumResult := results[0]
	for i := 1; i < len(results); i++ {
		sumResult = s.evaluator.AddNew(sumResult, results[i])
	}

	// Phase 3: Apply scaling and logistic to the sum
	scaling := datasetScaling[s.dataSet].LogitSoftVoting
	return s.applyLogistic(sumResult, scaling)
}

func (s *CSPServer) performInference(encModel *EncryptedModel, encTestData *mkckks.Ciphertext) (*mkckks.Ciphertext, error) {
	// Linear inference
	result := s.evaluator.MulRelinNew(encModel.EncWeights, encTestData, s.rlkSet)

	// Rotation operations
	for j := bits.Len(uint(s.config.FeaturePad)) - 2; j >= 0; j-- {
		rotated := s.evaluator.RotateNew(result, (1<<j)*s.config.SamplePad, s.rtkSet)
		result = s.evaluator.AddNew(result, rotated)
	}

	// Add intercept
	return s.evaluator.AddNew(result, encModel.EncIntercept), nil
}

func (s *CSPServer) RequestInference(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// fmt.Printf("[CSP] Processing inference request from client %s\n", req.ClientId)

	if err := serialization.AddPublicKeyFromBytes(s.pkSet, req.PublicKey, s.mkParams.Parameters); err != nil {
		return nil, fmt.Errorf("failed to add client public key: %v", err)
	}
	if err := serialization.AddRelinKeyFromBytes(s.rlkSet, req.RelinearizationKey, s.mkParams.Parameters); err != nil {
		return nil, fmt.Errorf("failed to add client relinearization key: %v", err)
	}
	for _, rtkBytes := range req.RotationKeys {
		if err := serialization.AddRotationKeyFromBytes(s.rtkSet, rtkBytes, s.mkParams.Parameters); err != nil {
			return nil, fmt.Errorf("failed to add client rotation key: %v", err)
		}
	}

	encTestData, err := serialization.DeserializeCiphertext(req.EncryptedData, s.mkParams)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize test data: %v", err)
	}

	deserializeEnd := time.Now().UnixNano()
	s.timing.ClientDataTransfer = time.Duration(deserializeEnd - req.SerializationStartTime)

	svStart := time.Now()
	softVotingResult, err := s.performSoftVoting(encTestData)
	if err != nil {
		return nil, fmt.Errorf("soft voting failed: %v", err)
	}
	s.timing.SoftVotingCompute = time.Since(svStart)

	lsvStart := time.Now()
	logitSoftVotingResult, err := s.performLogitSoftVoting(encTestData)
	if err != nil {
		return nil, fmt.Errorf("logit soft voting failed: %v", err)
	}
	s.timing.LogitSoftVotingCompute = time.Since(lsvStart)

	decryptionStart := time.Now().UnixNano()
	softVotingDecrypted := softVotingResult.CopyNew()
	logitSoftVotingDecrypted := logitSoftVotingResult.CopyNew()

	ownerIDs := make([]string, 0, len(s.encryptedModels))
	for ownerID := range s.encryptedModels {
		ownerIDs = append(ownerIDs, ownerID)
	}
	sort.Strings(ownerIDs)

	for _, ownerID := range ownerIDs {
		// fmt.Printf("[CSP] Requesting partial decryption from owner %s\n", ownerID)

		softVotingBytes, err := serialization.SerializeCiphertext(softVotingDecrypted)
		if err != nil {
			return nil, fmt.Errorf("failed to serialize soft voting result: %v", err)
		}

		serializationStart := time.Now().UnixNano()
		softDecResp, err := s.dataOwnerClients[ownerID].PerformPartialDecryption(ctx, &pb.PartialDecryptionRequest{
			PartyId:                ownerID,
			EncryptedResult:        softVotingBytes,
			SerializationStartTime: serializationStart,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to get soft voting partial decryption from owner %s: %v", ownerID, err)
		}

		logitSoftVotingBytes, err := serialization.SerializeCiphertext(logitSoftVotingDecrypted)
		if err != nil {
			return nil, fmt.Errorf("failed to serialize logit soft voting result: %v", err)
		}

		serializationStart = time.Now().UnixNano()
		logitDecResp, err := s.dataOwnerClients[ownerID].PerformPartialDecryption(ctx, &pb.PartialDecryptionRequest{
			PartyId:                ownerID,
			EncryptedResult:        logitSoftVotingBytes,
			SerializationStartTime: serializationStart,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to get logit soft voting partial decryption from owner %s: %v", ownerID, err)
		}

		softPartialDec, err := serialization.DeserializeCiphertext(softDecResp.PartialDecryption, s.mkParams)
		if err != nil {
			return nil, fmt.Errorf("failed to deserialize soft voting partial decryption: %v", err)
		}

		logitPartialDec, err := serialization.DeserializeCiphertext(logitDecResp.PartialDecryption, s.mkParams)
		if err != nil {
			return nil, fmt.Errorf("failed to deserialize logit soft voting partial decryption: %v", err)
		}

		softVotingDecrypted.Ciphertext.Value = softPartialDec.Value
		logitSoftVotingDecrypted.Ciphertext.Value = logitPartialDec.Value

		// fmt.Printf("[CSP] Received partial decryption from owner %s\n", ownerID)
	}

	resultBytes, err := serialization.SerializeVotingResults(softVotingDecrypted, logitSoftVotingDecrypted, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize results: %v", err)
	}

	return &pb.InferenceResponse{
		EncryptedResult:     resultBytes,
		DecryptionStartTime: decryptionStart,
	}, nil
}
