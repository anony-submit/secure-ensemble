package client

import (
	"context"
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
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

func (c *Client) GetTiming() ClientTiming {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.timing
}

func NewClient(clientID string, params mkckks.Parameters, cspAddr string) (*Client, error) {
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
		clientID:  clientID,
		mkParams:  params,
		evaluator: mkckks.NewEvaluator(params),
		encryptor: mkckks.NewEncryptor(params),
		decryptor: mkckks.NewDecryptor(params),
		cspClient: pb.NewCSPServiceClient(conn),
		rtks:      make(map[int]*mkrlwe.RotationKey),
	}, nil
}

func (c *Client) GenerateKeys() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	startTime := time.Now()
	kgen := mkckks.NewKeyGenerator(c.mkParams)
	c.sk, c.pk = kgen.GenKeyPair(c.clientID)
	c.rlk = kgen.GenRelinearizationKey(c.sk)

	rotations := []int{16384, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 4096, 6144, 8192, 10240, 12288, 14336,
		64, 128, 32640, 32704, 3072, 1, 2, 4, 8, 16, 32, 32767}

	for _, rot := range rotations {
		rtk := kgen.GenRotationKey(rot, c.sk)
		c.rtks[rot] = rtk
	}

	c.timing.KeyGenStats.AddSample(time.Since(startTime))
	return nil
}

func (c *Client) LoadTestData(index int) ([]float64, int, error) {
	file, err := os.Open("../data/cifar10_test.csv")
	if err != nil {
		return nil, 0, fmt.Errorf("failed to open test data: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	if _, err := reader.Read(); err != nil {
		return nil, 0, fmt.Errorf("failed to skip header: %v", err)
	}

	for i := 0; i < index; i++ {
		if _, err := reader.Read(); err != nil {
			return nil, 0, fmt.Errorf("failed to skip to sample %d: %v", index, err)
		}
	}

	record, err := reader.Read()
	if err != nil {
		return nil, 0, fmt.Errorf("failed to read sample: %v", err)
	}

	label, err := strconv.Atoi(record[len(record)-1])
	if err != nil {
		return nil, 0, fmt.Errorf("failed to parse label: %v", err)
	}

	image := [3][32][32]float64{}
	idx := 0
	for c := 0; c < 3; c++ {
		for i := 0; i < 32; i++ {
			for j := 0; j < 32; j++ {
				val, err := strconv.ParseFloat(record[idx], 64)
				if err != nil {
					return nil, 0, fmt.Errorf("failed to parse pixel value: %v", err)
				}
				image[c][i][j] = val
				idx++
			}
		}
	}

	encodedImage := encodeImage(image) // Using existing cifar10 encoding function

	flattenedImage := make([]float64, 0)
	for _, vec := range encodedImage {
		flattenedImage = append(flattenedImage, vec...)
	}

	return flattenedImage, label, nil
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

	encInputBytes := make([][]byte, 6)
	for i := 0; i < 6; i++ {
		bytes, err := ser.SerializeCiphertext(encInput[i])
		if err != nil {
			return nil, fmt.Errorf("failed to serialize encrypted input: %v", err)
		}
		encInputBytes[i] = bytes
	}

	resp, err := c.cspClient.RequestInference(context.Background(), &pb.InferenceRequest{
		ClientId:           c.clientID,
		EncryptedInput:     encInputBytes,
		PublicKey:          pkBytes,
		RelinearizationKey: rlkBytes,
		RotationKeys:       allRtkBytes,
		RequestStartTime:   time.Now().UnixNano(),
	})
	if err != nil {
		return nil, fmt.Errorf("inference request failed: %v", err)
	}

	decryptStart := time.Now()
	c.timing.TotalDecryptionStats.AddSample(time.Duration(decryptStart.UnixNano() - resp.DecryptionProtocolStartTime))

	encResult, err := ser.DeserializeCiphertext(resp.EncryptedResult, c.mkParams)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize result: %v", err)
	}

	skSet := mkrlwe.NewSecretKeySet()
	skSet.AddSecretKey(c.sk)
	result := c.decryptor.Decrypt(encResult, skSet)
	c.timing.DecryptionStats.AddSample(time.Since(decryptStart))

	scores := make([]float64, 10)
	for i := 0; i < 5; i++ {
		scores[i*2] = real(result.Value[i*4096])
		scores[i*2+1] = real(result.Value[i*4096+1])
	}
	return scores, nil
}

func (c *Client) encryptInput(input []float64) ([6]*mkckks.Ciphertext, error) {
	var encryptedInput [6]*mkckks.Ciphertext

	for i := 0; i < 6; i++ {
		msg := mkckks.NewMessage(c.mkParams)
		start := i * len(input) / 6
		end := (i + 1) * len(input) / 6

		for j, val := range input[start:end] {
			msg.Value[j] = complex(val, 0)
		}

		encryptedInput[i] = c.encryptor.EncryptMsgNew(msg, c.pk)
	}

	return encryptedInput, nil
}

////////////////////////////////////////////////////////////////////////
///////////////////////////// Image Encoding ////////////////////////////
////////////////////////////////////////////////////////////////////////

func createVectorsFromImage(image [32][32]float64) [][]float64 {
	vectors := make([][]float64, 16)
	for i := range vectors {
		vectors[i] = make([]float64, 64)
	}

	vectorIdx := 0
	for startRow := 0; startRow < 4; startRow++ {
		for startCol := 0; startCol < 4; startCol++ {
			vec := vectors[vectorIdx]
			idx := 0
			for i := 0; i < 32; i += 4 {
				for j := 0; j < 32; j += 4 {
					vec[idx] = image[(startRow+i)%32][(startCol+j)%32]
					idx++
				}
			}
			vectorIdx++
		}
	}
	return vectors
}

func encodeChannel(vectors [][]float64) [][]float64 {
	enc1FirstHalf := make([]float64, 0)
	enc1SecondHalf := make([]float64, 0)
	enc2FirstHalf := make([]float64, 0)
	enc2SecondHalf := make([]float64, 0)

	firstGroup := []int{0, 2, 8, 10}   // Vec1,Vec3,Vec9,Vec11
	secondGroup := []int{1, 3, 9, 11}  // Vec2,Vec4,Vec10,Vec12
	thirdGroup := []int{4, 6, 12, 14}  // Vec5,Vec7,Vec13,Vec15
	fourthGroup := []int{5, 7, 13, 15} // Vec6,Vec8,Vec14,Vec16

	for i := 0; i < 64; i++ {
		for _, idx := range firstGroup {
			enc1FirstHalf = append(enc1FirstHalf, vectors[idx]...)
		}
		for _, idx := range secondGroup {
			enc1SecondHalf = append(enc1SecondHalf, vectors[idx]...)
		}
		for _, idx := range thirdGroup {
			enc2FirstHalf = append(enc2FirstHalf, vectors[idx]...)
		}
		for _, idx := range fourthGroup {
			enc2SecondHalf = append(enc2SecondHalf, vectors[idx]...)
		}
	}

	encoding1 := append(enc1FirstHalf, enc1SecondHalf...)
	encoding2 := append(enc2FirstHalf, enc2SecondHalf...)

	return [][]float64{encoding1, encoding2}
}

func encodeImage(image [3][32][32]float64) [][]float64 {
	result := make([][]float64, 6)

	for channel := 0; channel < 3; channel++ {
		vectors := createVectorsFromImage(image[channel])
		channelEncodings := encodeChannel(vectors)
		copy(result[channel*2:(channel+1)*2], channelEncodings)
	}

	return result
}
