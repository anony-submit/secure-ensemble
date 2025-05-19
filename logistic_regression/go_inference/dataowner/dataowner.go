package dataowner

import (
	"context"
	"fmt"
	"net"
	"sync"
	"time"

	pb "logistic_regression/go_inference/proto"
	"secure-ensemble/pkg/logistic"
	"secure-ensemble/pkg/serialization"

	"github.com/anony-submit/snu-mghe/mkckks"
	"github.com/anony-submit/snu-mghe/mkrlwe"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type DataOwnerTiming struct {
	KeyGeneration     time.Duration
	ModelEncryption   time.Duration
	PartialDecryption time.Duration
}

func (d *DataOwner) GetTiming() DataOwnerTiming {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.timing
}

type DataOwner struct {
	// Basic info
	ownerID string
	dataSet string

	// Crypto components
	mkParams  mkckks.Parameters
	evaluator *mkckks.Evaluator
	encryptor *mkckks.Encryptor
	config    logistic.BatchConfig

	// Keys
	sk   *mkrlwe.SecretKey
	pk   *mkrlwe.PublicKey
	rlk  *mkrlwe.RelinearizationKey
	rtks map[int]*mkrlwe.RotationKey

	// CSP connection
	cspClient pb.CSPServiceClient
	timing    DataOwnerTiming
	mu        sync.RWMutex
}

type DataOwnerServer struct {
	pb.UnimplementedDataOwnerServiceServer
	dataOwner *DataOwner
}

func NewDataOwnerServer(dataOwner *DataOwner) *DataOwnerServer {
	return &DataOwnerServer{
		dataOwner: dataOwner,
	}
}

func NewDataOwner(ownerID, dataSet string, params mkckks.Parameters, config logistic.BatchConfig, cspAddr string) (*DataOwner, error) {
	conn, err := grpc.DialContext(
		context.Background(),
		cspAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to CSP: %v", err)
	}

	d := &DataOwner{
		ownerID:   ownerID,
		dataSet:   dataSet,
		mkParams:  params,
		evaluator: mkckks.NewEvaluator(params),
		encryptor: mkckks.NewEncryptor(params),
		config:    config,
		cspClient: pb.NewCSPServiceClient(conn),
		rtks:      make(map[int]*mkrlwe.RotationKey),
	}

	return d, nil
}

func (d *DataOwner) GenerateKeys() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	startTime := time.Now()

	kgen := mkckks.NewKeyGenerator(d.mkParams)
	d.sk, d.pk = kgen.GenKeyPair(d.ownerID)
	d.rlk = kgen.GenRelinearizationKey(d.sk)

	rotations := getRotations(d.config)
	for _, rot := range rotations {
		rtk := kgen.GenRotationKey(rot, d.sk)
		d.rtks[rot] = rtk
	}

	d.timing.KeyGeneration = time.Since(startTime)
	return nil
}

func (d *DataOwner) EnrollModel(weights, intercept []float64) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	// fmt.Printf("[DataOwner %s] Starting model enrollment process...\n", d.ownerID)

	encryptStart := time.Now()
	weightMatrix := logistic.CreateWeightMatrix(weights, d.config.SampleCount)
	weightsBatched := logistic.CreateBatchedMatrix(weightMatrix, d.config)
	weightMsg := mkckks.NewMessage(d.mkParams)
	copy(weightMsg.Value, weightsBatched)
	encWeights := d.encryptor.EncryptMsgNew(weightMsg, d.pk)

	interceptMatrix := make([][]float64, d.config.FeatureDim)
	for k := range interceptMatrix {
		interceptMatrix[k] = make([]float64, d.config.SampleCount)
		for l := 0; l < d.config.SampleCount; l++ {
			interceptMatrix[k][l] = intercept[0]
		}
	}
	interceptBatched := logistic.CreateBatchedMatrix(interceptMatrix, d.config)
	interceptMsg := mkckks.NewMessage(d.mkParams)
	copy(interceptMsg.Value, interceptBatched)
	encIntercept := d.encryptor.EncryptMsgNew(interceptMsg, d.pk)
	d.timing.ModelEncryption = time.Since(encryptStart)

	serializationStart := time.Now().UnixNano()
	encWeightsBytes, err := serialization.SerializeCiphertext(encWeights)
	if err != nil {
		return fmt.Errorf("failed to serialize weights: %v", err)
	}
	encInterceptBytes, err := serialization.SerializeCiphertext(encIntercept)
	if err != nil {
		return fmt.Errorf("failed to serialize intercept: %v", err)
	}
	pkBytes, err := serialization.SerializePublicKey(d.pk)
	if err != nil {
		return fmt.Errorf("failed to serialize public key: %v", err)
	}
	rlkBytes, err := serialization.SerializeRelinearizationKey(d.rlk)
	if err != nil {
		return fmt.Errorf("failed to serialize relinearization key: %v", err)
	}
	allRtkBytes := make([][]byte, 0)
	for _, rtk := range d.rtks {
		rtkBytes, err := serialization.SerializeRotationKey(rtk)
		if err != nil {
			return fmt.Errorf("failed to serialize rotation key: %v", err)
		}
		allRtkBytes = append(allRtkBytes, rtkBytes)
	}

	_, err = d.cspClient.EnrollModel(context.Background(), &pb.EnrollModelRequest{
		OwnerId:                d.ownerID,
		EncryptedWeights:       encWeightsBytes,
		EncryptedIntercept:     encInterceptBytes,
		PublicKey:              pkBytes,
		RelinearizationKey:     rlkBytes,
		RotationKeys:           allRtkBytes,
		SerializationStartTime: serializationStart,
	})
	if err != nil {
		return err
	}

	return nil
}

func (d *DataOwner) PerformPartialDecryption(ctx context.Context, req *pb.PartialDecryptionRequest) (*pb.PartialDecryptionResponse, error) {
	computeStart := time.Now()
	encResult, err := serialization.DeserializeCiphertext(req.EncryptedResult, d.mkParams)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize encrypted result: %v", err)
	}

	decryptor := mkckks.NewDecryptor(d.mkParams)
	decryptor.PartialDecrypt(encResult, d.sk)
	d.timing.PartialDecryption = time.Since(computeStart)

	resultBytes, err := serialization.SerializeCiphertext(encResult)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize partial decryption: %v", err)
	}

	return &pb.PartialDecryptionResponse{
		PartialDecryption: resultBytes,
	}, nil
}

func (s *DataOwnerServer) PerformPartialDecryption(ctx context.Context, req *pb.PartialDecryptionRequest) (*pb.PartialDecryptionResponse, error) {
	return s.dataOwner.PerformPartialDecryption(ctx, req)
}

func (d *DataOwner) StartServer(address string) (func(), error) {
	listener, err := net.Listen("tcp", address)
	if err != nil {
		return nil, fmt.Errorf("failed to start listener: %v", err)
	}

	maxSize := 1024 * 1024 * 1024 * 4 // 4GB
	server := grpc.NewServer(
		grpc.MaxRecvMsgSize(maxSize),
		grpc.MaxSendMsgSize(maxSize),
	)

	pb.RegisterDataOwnerServiceServer(server, NewDataOwnerServer(d))

	// fmt.Printf("[DataOwner %s] Starting server on %s\n", d.ownerID, address)
	go func() {
		if err := server.Serve(listener); err != nil {
			fmt.Printf("[DataOwner %s] Server error: %v\n", d.ownerID, err)
		}
	}()
	cleanup := func() {
		// fmt.Printf("[DataOwner %s] Shutting down server...\n", d.ownerID)
		server.GracefulStop()
		listener.Close()
	}

	return cleanup, nil
}

// Helper function to get rotations based on config
func getRotations(config logistic.BatchConfig) []int {
	rotations := []int{}
	for i := 0; (1 << i) < config.FeaturePad; i++ {
		rotations = append(rotations, (1<<i)*config.SamplePad)
	}
	return rotations
}
