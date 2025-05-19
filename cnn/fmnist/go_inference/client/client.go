package client

import (
	"context"
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
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

type ClientTiming struct {
	KeyGenStats          common.TimingStats
	DataEncryptionStats  common.TimingStats
	DecryptionStats      common.TimingStats
	TotalDecryptionStats common.TimingStats
}

type Client struct {
	clientID  string
	mkParams  mkckks.Parameters
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

func NewClient(clientID string, params mkckks.Parameters, cspAddr string) (*Client, error) {
	maxSize := 1024 * 1024 * 1024 * 8
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
		clientID:  clientID,
		mkParams:  params,
		evaluator: mkckks.NewEvaluator(params),
		encryptor: mkckks.NewEncryptor(params),
		decryptor: mkckks.NewDecryptor(params),
		cspClient: pb.NewCSPServiceClient(conn),
		rtks:      make(map[int]*mkrlwe.RotationKey),
	}, nil
}

func (c *Client) GetTiming() ClientTiming {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.timing
}

func (c *Client) GenerateKeys() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	startTime := time.Now()
	kgen := mkckks.NewKeyGenerator(c.mkParams)
	c.sk, c.pk = kgen.GenKeyPair(c.clientID)
	c.rlk = kgen.GenRelinearizationKey(c.sk)

	rotations := []int{1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
		34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
		128, 256, 512, 1024, 2048, 4096}
	for _, rot := range rotations {
		rtk := kgen.GenRotationKey(rot, c.sk)
		c.rtks[rot] = rtk
	}

	c.timing.KeyGenStats.AddSample(time.Since(startTime))
	return nil
}

func (c *Client) LoadTestData(index int) ([]float64, int, error) {
	file, err := os.Open("../data/fmnist_test.csv")
	if err != nil {
		return nil, 0, fmt.Errorf("failed to open test data: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, 0, fmt.Errorf("failed to read CSV: %v", err)
	}

	if index >= len(records) {
		return nil, 0, fmt.Errorf("index out of range: %d", index)
	}

	input := make([]float64, 784)
	label, err := strconv.Atoi(records[index][len(records[index])-1])
	if err != nil {
		return nil, 0, fmt.Errorf("failed to parse label: %v", err)
	}

	for i := 0; i < 784; i++ {
		val, err := strconv.ParseFloat(records[index][i], 64)
		if err != nil {
			return nil, 0, fmt.Errorf("failed to parse input value: %v", err)
		}
		input[i] = val
	}

	return input, label, nil
}

func (c *Client) RequestInference(input []float64) ([]float64, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	encryptStart := time.Now()
	encInput, err := c.encryptInput(input)
	if err != nil {
		return nil, fmt.Errorf("failed to encrypt input: %v", err)
	}
	c.timing.DataEncryptionStats.AddSample(time.Since(encryptStart))

	serializationStart := time.Now().UnixNano()
	pkBytes, err := ser.SerializePublicKey(c.pk)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize public key: %v", err)
	}

	rlkBytes, err := ser.SerializeRelinearizationKey(c.rlk)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize relinearization key: %v", err)
	}

	allRtkBytes := make([][]byte, 0, len(c.rtks))
	for _, rtk := range c.rtks {
		rtkBytes, err := ser.SerializeRotationKey(rtk)
		if err != nil {
			return nil, fmt.Errorf("failed to serialize rotation key: %v", err)
		}
		allRtkBytes = append(allRtkBytes, rtkBytes)
	}

	encInputBytes, err := ser.SerializeCiphertext(encInput)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize encrypted input: %v", err)
	}

	resp, err := c.cspClient.RequestInference(context.Background(), &pb.InferenceRequest{
		ClientId:               c.clientID,
		EncryptedInput:         encInputBytes,
		PublicKey:              pkBytes,
		RelinearizationKey:     rlkBytes,
		RotationKeys:           allRtkBytes,
		SerializationStartTime: serializationStart,
	})
	if err != nil {
		return nil, fmt.Errorf("inference request failed: %v", err)
	}

	encResult, err := ser.DeserializeCiphertext(resp.EncryptedResult, c.mkParams)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize result: %v", err)
	}

	decryptStart := time.Now()
	skSet := mkrlwe.NewSecretKeySet()
	skSet.AddSecretKey(c.sk)
	result := c.decryptor.Decrypt(encResult, skSet)
	c.timing.DecryptionStats.AddSample(time.Since(decryptStart))
	c.timing.TotalDecryptionStats.AddSample(time.Since(time.Unix(0, resp.DecryptionProtocolStartTime)))

	scores := make([]float64, 10)
	for i := 0; i < 10; i++ {
		scores[i] = real(result.Value[i])
	}
	return scores, nil
}

func (c *Client) encryptInput(input []float64) (*mkckks.Ciphertext, error) {
	msg := mkckks.NewMessage(c.mkParams)
	encoded := encodeImage(input)
	copy(msg.Value, encoded)
	return c.encryptor.EncryptMsgNew(msg, c.pk), nil
}

func encodeImage(pixels []float64) []complex128 {
	if len(pixels) != 784 {
		panic("Image must be 784 pixels")
	}

	img := make([][]float64, 28)
	for i := range img {
		img[i] = make([]float64, 28)
		for j := 0; j < 28; j++ {
			img[i][j] = pixels[i*28+j]
		}
	}

	A := make([]float64, 196)
	B := make([]float64, 196)
	C := make([]float64, 196)
	D := make([]float64, 196)
	idx := 0
	for i := 0; i < 28; i += 2 {
		for j := 0; j < 28; j += 2 {
			A[idx] = img[i][j]     // top-left
			B[idx] = img[i][j+1]   // top-right
			C[idx] = img[i+1][j]   // bottom-left
			D[idx] = img[i+1][j+1] // bottom-right
			idx++
		}
	}

	result := make([]complex128, 8192)
	vectors := [][]float64{A, B, C, D}
	for vecIdx, vec := range vectors {
		baseOffset := vecIdx * 1024
		for i := 0; i < 196; i++ {
			for j := 0; j < 4; j++ {
				result[baseOffset+i+j*196] = complex(vec[i], 0)
			}
		}
	}

	copy(result[4096:], result[:4096])
	return result
}
