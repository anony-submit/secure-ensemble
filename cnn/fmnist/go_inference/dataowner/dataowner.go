package dataowner

import (
	"context"
	"fmt"
	"net"
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

type DataOwnerTiming struct {
	KeyGenStats            common.TimingStats
	ModelEncryptionStats   common.TimingStats
	PartialDecryptionStats common.TimingStats
}

type DataOwner struct {
	ownerID   string
	mkParams  mkckks.Parameters
	evaluator *mkckks.Evaluator
	encryptor *mkckks.Encryptor
	decryptor *mkckks.Decryptor

	sk   *mkrlwe.SecretKey
	pk   *mkrlwe.PublicKey
	rlk  *mkrlwe.RelinearizationKey
	rtks map[int]*mkrlwe.RotationKey

	cspClient pb.CSPServiceClient
	timing    DataOwnerTiming
	mu        sync.RWMutex
}

type DataOwnerServer struct {
	pb.UnimplementedDataOwnerServiceServer
	dataOwner *DataOwner
}

func (d *DataOwner) GetOwnerID() string {
	return d.ownerID
}

func (d *DataOwner) GetTiming() DataOwnerTiming {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.timing
}

func NewDataOwner(ownerID string, params mkckks.Parameters, cspAddr string) (*DataOwner, error) {
	maxSize := 1024 * 1024 * 1024 * 8
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
		decryptor: mkckks.NewDecryptor(params),
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

	rotations := []int{1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
		34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
		128, 256, 512, 1024, 2048, 4096}
	for _, rot := range rotations {
		rtk := kgen.GenRotationKey(rot, d.sk)
		d.rtks[rot] = rtk
	}

	d.timing.KeyGenStats.AddSample(time.Since(startTime))
	return nil
}

func (d *DataOwner) EnrollModel(convW [][]float64, convB []float64, fc1w [][]float64, fc1b []float64,
	fc2w [][]float64, fc2b []float64) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	encryptStart := time.Now()
	encConvWeights := encryptConvWeights(d.mkParams, d.encryptor, convW, d.pk)
	encConvBias := encryptConvBias(d.mkParams, d.encryptor, convB, d.pk)

	fc1WeightsCipher := make([]*mkckks.Ciphertext, 8)
	flattenedWeights := flattenMatrix(fc1w, 1024)

	for group := 0; group < 8; group++ {
		msg := mkckks.NewMessage(d.mkParams)
		valueIdx := 0
		for d := group * 8; d < (group+1)*8; d++ {
			diagonalVec := getDiagonalVector(flattenedWeights, 64, 1024, d)
			for _, val := range diagonalVec {
				msg.Value[valueIdx] = complex(val, 0)
				valueIdx++
			}
		}
		fc1WeightsCipher[group] = d.encryptor.EncryptMsgNew(msg, d.pk)
	}

	fc1BiasMsg := mkckks.NewMessage(d.mkParams)
	for i := 0; i < 8192; i++ {
		fc1BiasMsg.Value[i] = complex(fc1b[i%64], 0)
	}
	encFC1Bias := d.encryptor.EncryptMsgNew(fc1BiasMsg, d.pk)

	paddedMatrix := make([][]float64, 16)
	for i := 0; i < 16; i++ {
		paddedMatrix[i] = make([]float64, 64)
		if i < len(fc2w) {
			copy(paddedMatrix[i], fc2w[i])
		}
	}

	fc2WeightsMsg := mkckks.NewMessage(d.mkParams)
	diagonals := make([]complex128, 16*64)
	for d := 0; d < 16; d++ {
		for i := 0; i < 64; i++ {
			row := i % 16
			col := (i + d) % 64
			diagonals[d*64+i] = complex(paddedMatrix[row][col], 0)
		}
	}

	for i := 0; i < 8; i++ {
		copy(fc2WeightsMsg.Value[i*1024:i*1024+1024], diagonals)
	}
	encFC2Weights := d.encryptor.EncryptMsgNew(fc2WeightsMsg, d.pk)

	paddedBias := make([]float64, 16)
	copy(paddedBias, fc2b)

	fc2BiasMsg := mkckks.NewMessage(d.mkParams)
	for i := 0; i < 512; i++ {
		for j := 0; j < 16; j++ {
			fc2BiasMsg.Value[i*16+j] = complex(paddedBias[j], 0)
		}
	}
	encFC2Bias := d.encryptor.EncryptMsgNew(fc2BiasMsg, d.pk)
	d.timing.ModelEncryptionStats.AddSample(time.Since(encryptStart))

	serializationStart := time.Now().UnixNano()

	convWeightsBytes, err := ser.SerializeCiphertext(encConvWeights)
	if err != nil {
		return fmt.Errorf("failed to serialize conv weights: %v", err)
	}

	convBiasBytes, err := ser.SerializeCiphertext(encConvBias)
	if err != nil {
		return fmt.Errorf("failed to serialize conv bias: %v", err)
	}

	fc1WeightsBytes := make([][]byte, 8)
	for i := 0; i < 8; i++ {
		bytes, err := ser.SerializeCiphertext(fc1WeightsCipher[i])
		if err != nil {
			return fmt.Errorf("failed to serialize FC1 weight %d: %v", i, err)
		}
		fc1WeightsBytes[i] = bytes
	}

	fc1BiasBytes, err := ser.SerializeCiphertext(encFC1Bias)
	if err != nil {
		return fmt.Errorf("failed to serialize FC1 bias: %v", err)
	}

	fc2WeightsBytes, err := ser.SerializeCiphertext(encFC2Weights)
	if err != nil {
		return fmt.Errorf("failed to serialize FC2 weights: %v", err)
	}

	fc2BiasBytes, err := ser.SerializeCiphertext(encFC2Bias)
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
		OwnerId:                d.ownerID,
		ConvWeights:            convWeightsBytes,
		ConvBias:               convBiasBytes,
		Fc1Weights:             fc1WeightsBytes,
		Fc1Bias:                fc1BiasBytes,
		Fc2Weights:             fc2WeightsBytes,
		Fc2Bias:                fc2BiasBytes,
		PublicKey:              pkBytes,
		RelinearizationKey:     rlkBytes,
		RotationKeys:           allRtkBytes,
		SerializationStartTime: serializationStart,
	})

	return err
}

func (d *DataOwner) PerformPartialDecryption(ctx context.Context, req *pb.PartialDecryptionRequest) (*pb.PartialDecryptionResponse, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	encResult, err := ser.DeserializeCiphertext(req.EncryptedResult, d.mkParams)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize encrypted result: %v", err)
	}

	startTime := time.Now().UnixNano()
	decryptor := mkckks.NewDecryptor(d.mkParams)
	decryptor.PartialDecrypt(encResult, d.sk)
	decryptionTime := time.Duration(time.Now().UnixNano() - startTime)

	d.timing.PartialDecryptionStats = common.TimingStats{
		Mean:    decryptionTime,
		StdDev:  0,
		Samples: []time.Duration{decryptionTime},
	}

	resultBytes, err := ser.SerializeCiphertext(encResult)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize partial decryption: %v", err)
	}

	return &pb.PartialDecryptionResponse{
		PartialDecryption: resultBytes,
	}, nil
}

func (d *DataOwner) StartServer(address string) (func(), error) {
	maxSize := 1024 * 1024 * 1024 * 8
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

func (s *DataOwnerServer) PerformPartialDecryption(ctx context.Context, req *pb.PartialDecryptionRequest) (*pb.PartialDecryptionResponse, error) {
	return s.dataOwner.PerformPartialDecryption(ctx, req)
}

func encryptConvWeights(mkParams mkckks.Parameters, encryptor *mkckks.Encryptor, kernels [][]float64, pk *mkrlwe.PublicKey) *mkckks.Ciphertext {
	msg := mkckks.NewMessage(mkParams)
	encoded := encodeKernel(kernels)
	copy(msg.Value, encoded)
	return encryptor.EncryptMsgNew(msg, pk)
}

func encryptConvBias(mkParams mkckks.Parameters, encryptor *mkckks.Encryptor, bias []float64, pk *mkrlwe.PublicKey) *mkckks.Ciphertext {
	msg := mkckks.NewMessage(mkParams)
	encoded := encodeConvBias(bias)
	copy(msg.Value, encoded)
	return encryptor.EncryptMsgNew(msg, pk)
}

func encodeKernel(kernels [][]float64) []complex128 {
	if len(kernels) != 4 || len(kernels[0]) != 4 {
		panic("Expected 4 kernels, each with 4 values")
	}

	result := make([]complex128, 8192)

	A_values := make([]float64, 4)
	B_values := make([]float64, 4)
	C_values := make([]float64, 4)
	D_values := make([]float64, 4)

	for i := 0; i < 4; i++ {
		A_values[i] = kernels[i][0]
		B_values[i] = kernels[i][1]
		C_values[i] = kernels[i][2]
		D_values[i] = kernels[i][3]
	}

	positions := [][]float64{A_values, B_values, C_values, D_values}
	for pos, values := range positions {
		startIdx := pos * 1024
		for i := 0; i < 4; i++ {
			value := values[i]
			for j := 0; j < 196; j++ {
				result[startIdx+i*196+j] = complex(value, 0)
			}
		}
	}

	copy(result[4096:], result[:4096])
	return result
}

func encodeConvBias(biasValues []float64) []complex128 {
	if len(biasValues) != 4 {
		panic("Expected 4 bias values")
	}

	result := make([]complex128, 8192)

	for i, bias := range biasValues {
		startIdx := i * 196
		for j := 0; j < 196; j++ {
			result[j+startIdx] = complex(bias, 0)
		}
	}

	for i := 1; i < 8; i++ {
		copy(result[i*1024:(i+1)*1024], result[:1024])
	}

	return result
}

func flattenMatrix(matrix [][]float64, padCols int) []float64 {
	rows := len(matrix)
	flattened := make([]float64, rows*padCols)
	for i := 0; i < rows; i++ {
		for j := 0; j < len(matrix[i]); j++ {
			flattened[i*padCols+j] = matrix[i][j]
		}
	}
	return flattened
}

func getDiagonalVector(matrix []float64, rows, cols, diagonalIndex int) []float64 {
	diagonalVec := make([]float64, cols)
	for i := 0; i < cols; i++ {
		row := i % rows
		index := row*cols + ((i + diagonalIndex) % cols)
		if index < len(matrix) {
			diagonalVec[i] = matrix[index]
		}
	}
	return diagonalVec
}
