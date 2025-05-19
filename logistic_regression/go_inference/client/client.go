package client

import (
	"context"
	"fmt"
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

type ClientTiming struct {
	KeyGeneration             time.Duration
	DataEncryption            time.Duration
	SoftVotingDecryption      time.Duration
	LogitSoftVotingDecryption time.Duration
	TotalDecryptionTime       time.Duration
}

type Client struct {
	clientID     string
	dataSet      string
	mkParams     mkckks.Parameters
	config       logistic.BatchConfig
	dataOwnerIDs []string // For partial decryption

	evaluator *mkckks.Evaluator
	encryptor *mkckks.Encryptor
	decryptor *mkckks.Decryptor

	sk   *mkrlwe.SecretKey
	pk   *mkrlwe.PublicKey
	rlk  *mkrlwe.RelinearizationKey
	rtks map[int]*mkrlwe.RotationKey

	cspClient pb.CSPServiceClient
	mu        sync.RWMutex
	timing    ClientTiming
}

func (c *Client) GetTiming() ClientTiming {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.timing
}

func NewClient(clientID, dataSet string, params mkckks.Parameters, config logistic.BatchConfig,
	dataOwnerIDs []string, cspAddr string) (*Client, error) {

	maxSize := 1024 * 1024 * 1024 * 4
	opts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(maxSize),
			grpc.MaxCallSendMsgSize(maxSize),
		),
	}
	conn, err := grpc.Dial(cspAddr, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to CSP: %v", err)
	}

	return &Client{
		clientID:     clientID,
		dataSet:      dataSet,
		mkParams:     params,
		config:       config,
		dataOwnerIDs: dataOwnerIDs,
		evaluator:    mkckks.NewEvaluator(params),
		encryptor:    mkckks.NewEncryptor(params),
		decryptor:    mkckks.NewDecryptor(params),
		cspClient:    pb.NewCSPServiceClient(conn),
		rtks:         make(map[int]*mkrlwe.RotationKey),
	}, nil
}

func (c *Client) GenerateKeys() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	startTime := time.Now()

	kgen := mkckks.NewKeyGenerator(c.mkParams)
	c.sk, c.pk = kgen.GenKeyPair(c.clientID)
	c.rlk = kgen.GenRelinearizationKey(c.sk)

	rotations := getRotations(c.config)
	for _, rot := range rotations {
		rtk := kgen.GenRotationKey(rot, c.sk)
		c.rtks[rot] = rtk
	}

	c.timing.KeyGeneration = time.Since(startTime)
	return nil
}

func (c *Client) RequestInference(testData [][]float64) (*mkckks.Message, *mkckks.Message, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// fmt.Printf("[Client] Starting inference request process...\n")

	encryptStart := time.Now()
	testDataBatched := logistic.CreateBatchedMatrix(testData, c.config)
	testDataMsg := mkckks.NewMessage(c.mkParams)
	copy(testDataMsg.Value, testDataBatched)
	encTestData := c.encryptor.EncryptMsgNew(testDataMsg, c.pk)
	c.timing.DataEncryption = time.Since(encryptStart)

	transferStart := time.Now()
	pkBytes, err := serialization.SerializePublicKey(c.pk)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to serialize public key: %v", err)
	}
	rlkBytes, err := serialization.SerializeRelinearizationKey(c.rlk)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to serialize relinearization key: %v", err)
	}
	allRtkBytes := make([][]byte, 0)
	for _, rtk := range c.rtks {
		rtkBytes, err := serialization.SerializeRotationKey(rtk)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to serialize rotation key: %v", err)
		}
		allRtkBytes = append(allRtkBytes, rtkBytes)
	}
	encDataBytes, err := serialization.SerializeCiphertext(encTestData)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to serialize test data: %v", err)
	}

	resp, err := c.cspClient.RequestInference(context.Background(), &pb.InferenceRequest{
		ClientId:               c.clientID,
		EncryptedData:          encDataBytes,
		PublicKey:              pkBytes,
		RelinearizationKey:     rlkBytes,
		RotationKeys:           allRtkBytes,
		SerializationStartTime: transferStart.UnixNano(),
	})
	if err != nil {
		return nil, nil, fmt.Errorf("inference request failed: %v", err)
	}

	softVoting, logitSoftVoting, _, err := serialization.DeserializeVotingResults(resp.EncryptedResult, c.mkParams)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to deserialize voting results: %v", err)
	}

	skSet := mkrlwe.NewSecretKeySet()
	skSet.AddSecretKey(c.sk)

	softStart := time.Now()
	softResult := c.decryptor.Decrypt(softVoting, skSet)
	c.timing.SoftVotingDecryption = time.Since(softStart)

	logitStart := time.Now()
	logitResult := c.decryptor.Decrypt(logitSoftVoting, skSet)
	c.timing.LogitSoftVotingDecryption = time.Since(logitStart)

	totalDecryptionTime := time.Duration(time.Now().UnixNano() - resp.DecryptionStartTime)
	c.timing.TotalDecryptionTime = totalDecryptionTime / 2

	return softResult, logitResult, nil
}

// Helper function to get rotations based on config
func getRotations(config logistic.BatchConfig) []int {
	rotations := []int{}
	for i := 0; (1 << i) < config.FeaturePad; i++ {
		rotations = append(rotations, (1<<i)*config.SamplePad)
	}
	return rotations
}
