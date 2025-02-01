package dataowner

import (
	"context"
	"fmt"
	"net"
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

type DataOwnerTiming struct {
	KeyGenStats           common.TimingStats
	ModelEncryptionStats  common.TimingStats
	PartialDecryptionTime time.Duration
}

type DataOwner struct {
	ownerID   string
	mkParams  mkckks.Parameters
	evaluator *mkckks.Evaluator
	encryptor *mkckks.Encryptor

	sk   *mkrlwe.SecretKey
	pk   *mkrlwe.PublicKey
	rlk  *mkrlwe.RelinearizationKey
	rtks map[int]*mkrlwe.RotationKey

	cspClient pb.CSPServiceClient
	timing    DataOwnerTiming
	mu        sync.RWMutex
}

func NewDataOwner(ownerID string, params mkckks.Parameters, cspAddr string) (*DataOwner, error) {
	maxSize := 1024 * 1024 * 1024 * 4
	conn, err := grpc.Dial(
		cspAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(maxSize),
			grpc.MaxCallSendMsgSize(maxSize),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to CSP: %v", err)
	}

	return &DataOwner{
		ownerID:   ownerID,
		mkParams:  params,
		evaluator: mkckks.NewEvaluator(params),
		encryptor: mkckks.NewEncryptor(params),
		cspClient: pb.NewCSPServiceClient(conn),
		rtks:      make(map[int]*mkrlwe.RotationKey),
	}, nil
}

func (d *DataOwner) GenerateKeys() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	startTime := time.Now()
	kgen := mkckks.NewKeyGenerator(d.mkParams)
	d.sk, d.pk = kgen.GenKeyPair(d.ownerID)
	d.rlk = kgen.GenRelinearizationKey(d.sk)

	rotations := []int{16384, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 4096, 6144, 8192, 10240, 12288, 14336,
		64, 128, 32640, 32704, 3072, 1, 2, 4, 8, 16, 32, 32767}

	for _, rot := range rotations {
		rtk := kgen.GenRotationKey(rot, d.sk)
		d.rtks[rot] = rtk
	}

	d.timing.KeyGenStats.AddSample(time.Since(startTime))
	return nil
}

func (d *DataOwner) GetOwnerID() string {
	return d.ownerID
}

func (d *DataOwner) GetTiming() DataOwnerTiming {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.timing
}

type DataOwnerServer struct {
	pb.UnimplementedDataOwnerServiceServer
	dataOwner *DataOwner
}

func (s *DataOwnerServer) PerformPartialDecryption(ctx context.Context, req *pb.PartialDecryptionRequest) (*pb.PartialDecryptionResponse, error) {
	return s.dataOwner.PerformPartialDecryption(ctx, req)
}

func (d *DataOwner) EnrollModel(modelParams *common.ModelParams) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	encryptStart := time.Now()

	encryptor := mkckks.NewEncryptor(d.mkParams)

	// Conv1 weights encryption
	var conv1WeightsEnc [6]*mkckks.Ciphertext
	encodedConv1Weights := encodeConv1Weight(modelParams.ConvParams.Conv1.Weight)
	for i := 0; i < 6; i++ {
		msg := mkckks.NewMessage(d.mkParams)
		for j := range encodedConv1Weights[i] {
			msg.Value[j] = complex(encodedConv1Weights[i][j], 0)
		}
		conv1WeightsEnc[i] = encryptor.EncryptMsgNew(msg, d.pk)
	}

	// Conv1 bias encryption
	conv1BiasMsg := mkckks.NewMessage(d.mkParams)
	copy(conv1BiasMsg.Value, encodeConv1Bias(modelParams.ConvParams.Conv1.Bias))
	conv1BiasEnc := encryptor.EncryptMsgNew(conv1BiasMsg, d.pk)

	// Conv2 weights encryption
	var conv2WeightsEnc [64]*mkckks.Ciphertext
	encodedConv2Weights := encodeConv2Weights(modelParams.ConvParams.Conv2.Weight)
	for i := 0; i < 64; i++ {
		msg := mkckks.NewMessage(d.mkParams)
		copy(msg.Value, encodedConv2Weights[i])
		conv2WeightsEnc[i] = encryptor.EncryptMsgNew(msg, d.pk)
	}

	// Conv2 bias encryption
	conv2BiasMsg := mkckks.NewMessage(d.mkParams)
	copy(conv2BiasMsg.Value, encodeConv2Bias(modelParams.ConvParams.Conv2.Bias))
	conv2BiasEnc := encryptor.EncryptMsgNew(conv2BiasMsg, d.pk)

	// FC1 weights encryption
	var fc1WeightsEnc [16]*mkckks.Ciphertext
	encodedFC1Weights := encodeFC1Weights(modelParams.ClassifierParams.FC1.Weight)
	for i := 0; i < 16; i++ {
		msg := mkckks.NewMessage(d.mkParams)
		copy(msg.Value, encodedFC1Weights[i])
		fc1WeightsEnc[i] = encryptor.EncryptMsgNew(msg, d.pk)
	}

	// FC1 bias encryption
	fc1BiasMsg := mkckks.NewMessage(d.mkParams)
	copy(fc1BiasMsg.Value, encodeFC1Bias(modelParams.ClassifierParams.FC1.Bias))
	fc1BiasEnc := encryptor.EncryptMsgNew(fc1BiasMsg, d.pk)

	// FC2 weights encryption
	fc2WeightsMsg := mkckks.NewMessage(d.mkParams)
	copy(fc2WeightsMsg.Value, encodeFC2Weight(modelParams.ClassifierParams.FC2.Weight))
	fc2WeightsEnc := encryptor.EncryptMsgNew(fc2WeightsMsg, d.pk)

	// FC2 bias encryption
	fc2BiasMsg := mkckks.NewMessage(d.mkParams)
	copy(fc2BiasMsg.Value, encodeFC2Bias(modelParams.ClassifierParams.FC2.Bias))
	fc2BiasEnc := encryptor.EncryptMsgNew(fc2BiasMsg, d.pk)

	d.timing.ModelEncryptionStats.AddSample(time.Since(encryptStart))

	// Convert to bytes and send to CSP
	conv1WeightsBytes := make([][]byte, 6)
	for i := 0; i < 6; i++ {
		bytes, err := ser.SerializeCiphertext(conv1WeightsEnc[i])
		if err != nil {
			return fmt.Errorf("failed to serialize Conv1 weight %d: %v", i, err)
		}
		conv1WeightsBytes[i] = bytes
	}

	conv1BiasBytes, err := ser.SerializeCiphertext(conv1BiasEnc)
	if err != nil {
		return fmt.Errorf("failed to serialize Conv1 bias: %v", err)
	}

	conv2WeightsBytes := make([][]byte, 64)
	for i := 0; i < 64; i++ {
		bytes, err := ser.SerializeCiphertext(conv2WeightsEnc[i])
		if err != nil {
			return fmt.Errorf("failed to serialize Conv2 weight %d: %v", i, err)
		}
		conv2WeightsBytes[i] = bytes
	}

	conv2BiasBytes, err := ser.SerializeCiphertext(conv2BiasEnc)
	if err != nil {
		return fmt.Errorf("failed to serialize Conv2 bias: %v", err)
	}

	fc1WeightsBytes := make([][]byte, 16)
	for i := 0; i < 16; i++ {
		bytes, err := ser.SerializeCiphertext(fc1WeightsEnc[i])
		if err != nil {
			return fmt.Errorf("failed to serialize FC1 weight %d: %v", i, err)
		}
		fc1WeightsBytes[i] = bytes
	}

	fc1BiasBytes, err := ser.SerializeCiphertext(fc1BiasEnc)
	if err != nil {
		return fmt.Errorf("failed to serialize FC1 bias: %v", err)
	}

	fc2WeightsBytes, err := ser.SerializeCiphertext(fc2WeightsEnc)
	if err != nil {
		return fmt.Errorf("failed to serialize FC2 weights: %v", err)
	}

	fc2BiasBytes, err := ser.SerializeCiphertext(fc2BiasEnc)
	if err != nil {
		return fmt.Errorf("failed to serialize FC2 bias: %v", err)
	}

	pkBytes, err := ser.SerializePublicKey(d.pk)
	if err != nil {
		return fmt.Errorf("failed to serialize public key: %v", err)
	}

	rlkBytes, err := ser.SerializeRelinearizationKey(d.rlk)
	if err != nil {
		return fmt.Errorf("failed to serialize relinearization key: %v", err)
	}

	allRtkBytes := make([][]byte, 0, len(d.rtks))
	for _, rtk := range d.rtks {
		rtkBytes, err := ser.SerializeRotationKey(rtk)
		if err != nil {
			return fmt.Errorf("failed to serialize rotation key: %v", err)
		}
		allRtkBytes = append(allRtkBytes, rtkBytes)
	}

	_, err = d.cspClient.EnrollModel(context.Background(), &pb.EnrollModelRequest{
		OwnerId:            d.ownerID,
		Conv1Weights:       conv1WeightsBytes,
		Conv1Bias:          conv1BiasBytes,
		Conv2Weights:       conv2WeightsBytes,
		Conv2Bias:          conv2BiasBytes,
		Fc1Weights:         fc1WeightsBytes,
		Fc1Bias:            fc1BiasBytes,
		Fc2Weights:         fc2WeightsBytes,
		Fc2Bias:            fc2BiasBytes,
		PublicKey:          pkBytes,
		RelinearizationKey: rlkBytes,
		RotationKeys:       allRtkBytes,
		RequestStartTime:   time.Now().UnixNano(),
	})

	return err
}

func (d *DataOwner) PerformPartialDecryption(ctx context.Context, req *pb.PartialDecryptionRequest) (*pb.PartialDecryptionResponse, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	startTime := time.Now()
	encResult, err := ser.DeserializeCiphertext(req.EncryptedResult, d.mkParams)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize encrypted result: %v", err)
	}

	decryptor := mkckks.NewDecryptor(d.mkParams)
	decryptor.PartialDecrypt(encResult, d.sk)
	d.timing.PartialDecryptionTime = time.Since(startTime)

	resultBytes, err := ser.SerializeCiphertext(encResult)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize partial decryption: %v", err)
	}

	return &pb.PartialDecryptionResponse{
		PartialDecryption: resultBytes,
	}, nil
}

func (d *DataOwner) StartServer(address string) (func(), error) {
	maxSize := 1024 * 1024 * 1024 * 4
	server := grpc.NewServer(
		grpc.MaxRecvMsgSize(maxSize),
		grpc.MaxSendMsgSize(maxSize),
	)

	pb.RegisterDataOwnerServiceServer(server, &DataOwnerServer{dataOwner: d})

	listener, err := net.Listen("tcp", address)
	if err != nil {
		return nil, fmt.Errorf("failed to start listener: %v", err)
	}

	go func() {
		if err := server.Serve(listener); err != nil {
			fmt.Printf("[DataOwner %s] Server error: %v\n", d.ownerID, err)
		}
	}()

	cleanup := func() {
		server.GracefulStop()
		listener.Close()
	}

	return cleanup, nil
}

////////////////////////////////////////////////////////////////////////
///////////////////////////// Conv1 Encoding ////////////////////////////
////////////////////////////////////////////////////////////////////////

func encodeConv1Weight(weights [][][][]float64) [][]float64 {
	encodings := make([][]float64, 6)
	for i := range encodings {
		encodings[i] = make([]float64, 32768)
	}

	for in := 0; in < 3; in++ {
		for h := 0; h < 2; h++ {
			encIdx := in*2 + h
			pos := 0
			for w := 0; w < 2; w++ {
				for out := 0; out < 64; out++ {
					val := weights[out][in][h][w]
					for j := 0; j < 256; j++ {
						encodings[encIdx][pos] = val
						pos++
					}
				}
			}
		}
	}
	return encodings
}

func encodeConv1Bias(bias []float64) []complex128 {
	result := make([]complex128, 32768)
	pos := 0

	for k := 0; k < 2; k++ {
		for i := 0; i < 64; i++ {
			biasVal := bias[i]
			for j := 0; j < 256; j++ {
				result[pos] = complex(biasVal, 0)
				pos++
			}
		}
	}
	return result
}

// //////////////////////////////////////////////////////////////////////
// /////////////////////////// Conv2 Encoding ////////////////////////////
// //////////////////////////////////////////////////////////////////////

func encodeConv2Weights(weights [][][][]float64) [][]complex128 {
	if len(weights) != 128 || len(weights[0]) != 64 || len(weights[0][0]) != 2 || len(weights[0][0][0]) != 2 {
		panic(fmt.Sprintf("Conv2: Expected weight shape (128,64,2,2), got (%d,%d,%d,%d)",
			len(weights), len(weights[0]), len(weights[0][0]), len(weights[0][0][0])))
	}

	encodings := make([][]complex128, 64)

	for patternIdx := 0; patternIdx < 64; patternIdx++ {
		encoding := make([]complex128, 32768)
		pos := 0
		for outCh := 1; outCh <= 128; outCh++ {
			inCh := ((outCh - 1 + patternIdx) % 64) + 1

			for h := 0; h < 2; h++ {
				for w := 0; w < 2; w++ {
					value := weights[outCh-1][inCh-1][h][w]
					for i := 0; i < 64; i++ {
						encoding[pos+i] = complex(value, 0)
					}
					pos += 64
				}
			}
		}

		if pos != 32768 {
			panic(fmt.Sprintf("Invalid encoding length for pattern %d: got %d, expected 32768", patternIdx, pos))
		}

		groupIdx := patternIdx / 8
		rotation := -(groupIdx * 8 * 256)
		encodings[patternIdx] = rotateVector(encoding, rotation)
	}

	if len(encodings) != 64 {
		panic(fmt.Sprintf("Invalid number of encodings: got %d, expected 64", len(encodings)))
	}

	return encodings
}

func encodeConv2Bias(bias []float64) []complex128 {
	result := make([]complex128, 32768)

	for i := 0; i < 128; i++ {
		biasVal := bias[i]
		startIdx := i * 256
		for j := 0; j < 256; j++ {
			result[startIdx+j] = complex(biasVal, 0)
		}
	}
	return result
}

// //////////////////////////////////////////////////////////////////////
// /////////////////////////// FC1 Encoding //////////////////////////////
// //////////////////////////////////////////////////////////////////////
func encodeFC1Weights(weights [][]float64) [][]complex128 {
	if len(weights) != 64 || len(weights[0]) != 8192 {
		panic("Expected weights dimension: 64x8192")
	}

	encodings := make([][]complex128, 16)
	for i := range encodings {
		encodings[i] = make([]complex128, 32768)
	}

	for shift := 0; shift < 16; shift++ {
		pos := 0

		for block := 0; block < 8; block++ {
			blockStartCol := (block*1024 + shift*64) % 8192

			for patternIdx := 0; patternIdx < 16; patternIdx++ {

				startRow := (patternIdx * 4) % 64
				colOffset := (patternIdx * 64) % 8192
				startCol := (blockStartCol + colOffset) % 8192

				for row := startRow; row < startRow+4; row++ {

					for colIdx := 0; colIdx < 64; colIdx++ {
						col := (startCol + colIdx) % 8192
						encodings[shift][pos] = complex(weights[row][col], 0)
						pos++
					}
				}
			}
		}
		groupIdx := shift / 4
		rotation := -(groupIdx * 256 * 4)

		encodings[shift] = rotateVector(encodings[shift], rotation)
	}

	return encodings
}

func encodeFC1Bias(bias []float64) []complex128 {
	if len(bias) != 64 {
		panic(fmt.Sprintf("Incorrect bias length: %d", len(bias)))
	}

	pattern := make([]complex128, 4096)
	for i, b := range bias {
		for j := 0; j < 64; j++ {
			pattern[i*64+j] = complex(b, 0)
		}
	}

	result := make([]complex128, 32768)
	for i := 0; i < 8; i++ {
		copy(result[i*4096:(i+1)*4096], pattern)
	}

	return result
}

// //////////////////////////////////////////////////////////////////////
// /////////////////////////// FC2 Encoding //////////////////////////////
// //////////////////////////////////////////////////////////////////////
func encodeFC2Weight(weights [][]float64) []complex128 {
	if len(weights) != 10 || len(weights[0]) != 64 {
		panic("FC2: Expected weight shape (10, 64)")
	}
	result := make([]complex128, 32768)
	pos := 0

	for blockIdx := 0; blockIdx < 5; blockIdx++ {
		row1 := blockIdx * 2
		row2 := row1 + 1

		for col := 0; col < 64; col++ {
			result[pos] = complex(weights[row1][col], 0)
			result[pos+1] = complex(weights[row2][col], 0)
			pos += 64
		}
	}

	return result
}

func encodeFC2Bias(bias []float64) []complex128 {
	if len(bias) != 10 {
		panic(fmt.Sprintf("Incorrect bias length: %d", len(bias)))
	}

	result := make([]complex128, 32768)

	for i := 0; i < 10; i += 2 {
		blockStart := (i / 2) * 4096

		result[blockStart] = complex(bias[i], 0)
		result[blockStart+1] = complex(bias[i+1], 0)
	}

	return result
}

// ////////////////////////////////////////////////////////////////
// /////////////////////////// Utility ////////////////////////////
// ////////////////////////////////////////////////////////////////
func rotateVector(vec []complex128, rotation int) []complex128 {
	n := len(vec)
	result := make([]complex128, n)

	rotation = -rotation

	rotation = rotation % n
	if rotation < 0 {
		rotation = n + rotation
	}

	for i := 0; i < n; i++ {
		newPos := (i + rotation) % n
		result[newPos] = vec[i]
	}

	return result
}
