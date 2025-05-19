package experiments

import (
	"encoding/json"
	"fmt"
	"math/bits"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"logistic_regression/go_inference/client"
	"logistic_regression/go_inference/common"
	"logistic_regression/go_inference/dataowner"
	pb "logistic_regression/go_inference/proto"
	"logistic_regression/go_inference/server"
	"secure-ensemble/pkg/logistic"

	"github.com/anony-submit/snu-mghe/mkckks"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
	"google.golang.org/grpc"
)

type ExperimentResults struct {
	SoftVoting      ExperimentResult
	LogitSoftVoting ExperimentResult
}

type ExperimentResult struct {
	Timing   TimingInfo
	Accuracy AccuracyInfo
}

type TimingInfo struct {
	Client struct {
		KeyGeneration             time.Duration
		DataEncryption            time.Duration
		DataTransfer              time.Duration
		SoftVotingDecryption      time.Duration
		LogitSoftVotingDecryption time.Duration
	}
	DataOwner struct {
		KeyGeneration     common.TimingStats
		ModelEncryption   common.TimingStats
		ModelTransfer     time.Duration
		PartialDecryption common.TimingStats
	}
	CSP struct {
		SoftVotingCompute      time.Duration
		LogitSoftVotingCompute time.Duration
		TotalDecryptionTime    time.Duration
	}
}

type AccuracyInfo struct {
	Percentage float64
	Correct    int
	Total      int
}

type ModelParams struct {
	Weights   []float64 `json:"weights"`
	Intercept []float64 `json:"intercept"`
}

type serverInstance struct {
	grpcServer *grpc.Server
	cspServer  *server.CSPServer
	listener   net.Listener
}

func setupCryptoParams(config ExperimentConfig) mkckks.Parameters {
	logSlots := bits.Len(uint(config.BatchConfig.FeaturePad*config.BatchConfig.SamplePad)) - 1
	params, _ := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:     15,
		LogSlots: logSlots,
		Q: []uint64{0x4000000120001, 0x10000140001, 0xffffe80001,
			0x10000290001, 0xffffc40001, 0x100003e0001,
			0x10000470001, 0x100004b0001, 0xffffb20001},
		P:     []uint64{0x40000001b0001, 0x3ffffffdf0001, 0x4000000270001},
		Scale: 1 << 40,
		Sigma: rlwe.DefaultSigma,
	})

	mkParams := mkckks.NewParameters(params)
	rotations := getRotations(config.BatchConfig)
	for _, rot := range rotations {
		mkParams.AddCRS(rot)
	}

	return mkParams
}

func startServer(config ExperimentConfig, mkParams mkckks.Parameters) (*serverInstance, error) {
	maxSize := 1024 * 1024 * 1024 * 4 // 4GB

	listener, err := net.Listen("tcp", config.CSPAddress)
	if err != nil {
		return nil, fmt.Errorf("failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer(
		grpc.MaxRecvMsgSize(maxSize),
		grpc.MaxSendMsgSize(maxSize),
	)
	cspServer := server.NewCSPServer(mkParams, config.BatchConfig, config.DataSet)
	pb.RegisterCSPServiceServer(grpcServer, cspServer)

	go func() {
		if err := grpcServer.Serve(listener); err != nil {
			fmt.Printf("failed to serve: %v\n", err)
		}
	}()

	time.Sleep(100 * time.Millisecond)
	// fmt.Println("[Server] Server started successfully")

	return &serverInstance{
		grpcServer: grpcServer,
		cspServer:  cspServer,
		listener:   listener,
	}, nil
}

func stopServer(s *serverInstance) {
	if s.grpcServer != nil {
		s.grpcServer.GracefulStop()
	}
	if s.listener != nil {
		s.listener.Close()
	}
}

func getRotations(config logistic.BatchConfig) []int {
	rotations := []int{}
	for i := 0; (1 << i) < config.FeaturePad; i++ {
		rotations = append(rotations, (1<<i)*config.SamplePad)
	}
	return rotations
}

func enrollModels(dataOwners []*dataowner.DataOwner, config ExperimentConfig) error {
	modelPath := filepath.Join("data", config.DataSet, config.Imbalance, config.Split,
		fmt.Sprintf("%s_%s_n%d_models.json", config.DataSet, config.Split, config.PartyCount))

	modelFile, err := os.ReadFile(modelPath)
	if err != nil {
		return fmt.Errorf("failed to read model file: %v", err)
	}

	var models []ModelParams
	if err := json.Unmarshal(modelFile, &models); err != nil {
		return fmt.Errorf("failed to unmarshal models: %v", err)
	}

	// var wg sync.WaitGroup
	// errChan := make(chan error, len(dataOwners))

	// for i, owner := range dataOwners {
	// 	wg.Add(1)
	// 	go func(index int, owner *dataowner.DataOwner) {
	// 		defer wg.Done()
	// 		if err := owner.EnrollModel(models[index].Weights, models[index].Intercept); err != nil {
	// 			errChan <- fmt.Errorf("failed to enroll model for owner %d: %v", index, err)
	// 		}
	// 	}(i, owner)
	// }

	// wg.Wait()
	// close(errChan)
	// for err := range errChan {
	// 	if err != nil {
	// 		return err
	// 	}
	// }
	for i, owner := range dataOwners {
		if err := owner.EnrollModel(models[i].Weights, models[i].Intercept); err != nil {
			return fmt.Errorf("failed to enroll model for owner %d: %v", i, err)
		}
	}
	return nil
}

func loadTestData(config ExperimentConfig) ([][]float64, []int, error) {
	testDataPath := filepath.Join("data", config.DataSet,
		fmt.Sprintf("%s_test.csv", config.DataSet))

	return logistic.LoadTestData(testDataPath,
		config.BatchConfig.FeatureDim,
		config.BatchConfig.SampleCount)
}

func calculateAccuracy(decResult *mkckks.Message, trueLabels []int, sampleCount int) AccuracyInfo {
	correct := 0
	for i := 0; i < sampleCount; i++ {
		val := real(decResult.Value[i])
		prediction := 0
		if val > 0.5 {
			prediction = 1
		}
		if (prediction == 1 && trueLabels[i] == 1) ||
			(prediction == 0 && trueLabels[i] == 0) {
			correct++
		}
	}

	return AccuracyInfo{
		Percentage: float64(correct) * 100.0 / float64(sampleCount),
		Correct:    correct,
		Total:      sampleCount,
	}
}

func writeResultsToFile(config ExperimentConfig, results map[string]ExperimentResult) error {
	output := formatResults(config, results)
	fmt.Println(output)

	resultsDir := filepath.Join("results", config.DataSet)
	if err := os.MkdirAll(resultsDir, 0755); err != nil {
		return fmt.Errorf("failed to create results directory: %v", err)
	}

	filename := fmt.Sprintf("results_%s_%s_%s_n%d.txt",
		config.DataSet, config.Split, config.Imbalance, config.PartyCount)
	resultsFile := filepath.Join(resultsDir, filename)

	if _, err := os.Stat(resultsFile); err == nil {
		backupFile := resultsFile + ".bak"
		if err := os.Rename(resultsFile, backupFile); err != nil {
			return fmt.Errorf("failed to backup existing results file: %v", err)
		}
	}

	if err := os.WriteFile(resultsFile, []byte(output), 0644); err != nil {
		return fmt.Errorf("failed to write results: %v", err)
	}

	return nil
}

func formatResults(config ExperimentConfig, results map[string]ExperimentResult) string {
	return fmt.Sprintf(`
=== Experiment Results ===
Dataset: %s
Party Count: %d
Split: %s
Imbalance: %s

Accuracy:
  Soft Voting:        %.2f%% (%d/%d correct)
  Logit Soft Voting:  %.2f%% (%d/%d correct)

Client Times:
  Key Generation:     %v
  Data Encryption:    %v
  Data Transfer:      %v
  Soft Voting Dec:    %v
  Logit Soft Dec:     %v

Data Owner Times:
  Key Generation:     %.2f ± %.2f ms
  Model Encryption:   %.2f ± %.2f ms
  Model Transfer:     %v
  Partial Decryption: %.2f ± %.2f ms

CSP Times:
  Soft Voting:        %v
  Logit Soft Voting:  %v
  Total Decryption:   %v
`,

		config.DataSet,
		config.PartyCount,
		config.Split,
		config.Imbalance,

		results["soft_voting"].Accuracy.Percentage,
		results["soft_voting"].Accuracy.Correct,
		results["soft_voting"].Accuracy.Total,
		results["logit_soft_voting"].Accuracy.Percentage,
		results["logit_soft_voting"].Accuracy.Correct,
		results["logit_soft_voting"].Accuracy.Total,

		results["soft_voting"].Timing.Client.KeyGeneration,
		results["soft_voting"].Timing.Client.DataEncryption,
		results["soft_voting"].Timing.Client.DataTransfer,
		results["soft_voting"].Timing.Client.SoftVotingDecryption,
		results["soft_voting"].Timing.Client.LogitSoftVotingDecryption,

		float64(results["soft_voting"].Timing.DataOwner.KeyGeneration.Mean.Milliseconds()),
		float64(results["soft_voting"].Timing.DataOwner.KeyGeneration.StdDev.Milliseconds()),
		float64(results["soft_voting"].Timing.DataOwner.ModelEncryption.Mean.Milliseconds()),
		float64(results["soft_voting"].Timing.DataOwner.ModelEncryption.StdDev.Milliseconds()),
		results["soft_voting"].Timing.DataOwner.ModelTransfer,
		float64(results["soft_voting"].Timing.DataOwner.PartialDecryption.Mean.Milliseconds()),
		float64(results["soft_voting"].Timing.DataOwner.PartialDecryption.StdDev.Milliseconds()),

		results["soft_voting"].Timing.CSP.SoftVotingCompute,
		results["soft_voting"].Timing.CSP.LogitSoftVotingCompute,
		results["soft_voting"].Timing.CSP.TotalDecryptionTime)
}

func RunDatasetExperiments(t *testing.T, dataset string) {
	splits := []string{"horizontal", "vertical"}
	imbalances := []string{"balanced", "dirichlet_0.1", "dirichlet_0.5"}

	fmt.Printf("\n=== Starting Ideal Case for %s dataset ===\n", dataset)
	t.Run("Ideal", func(t *testing.T) {
		runIdealExperiment(t, dataset)
	})

	fmt.Printf("\n=== Starting Main Experiments for %s dataset ===\n", dataset)
	for _, split := range splits {
		partyCounts := GetValidPartyCounts(dataset, split)

		for _, imbalance := range imbalances {
			fmt.Printf("\n--- Starting experiments for %s split, %s imbalance ---\n",
				split, imbalance)

			for _, n := range partyCounts {
				fmt.Printf("\nRunning experiment with %d parties...\n", n)
				testName := fmt.Sprintf("%s_%s_N%d", split, imbalance, n)
				t.Run(testName, func(t *testing.T) {
					config := ExperimentConfig{
						DataSet:     dataset,
						PartyCount:  n,
						Split:       split,
						Imbalance:   imbalance,
						BatchConfig: datasetConfigs[dataset],
						CSPAddress:  "localhost:50051",
					}
					runExperiment(t, config)
				})
			}
		}
	}
}

func runIdealExperiment(t *testing.T, dataset string) {
	config := ExperimentConfig{
		DataSet:     dataset,
		PartyCount:  1,
		Split:       "ideal",
		Imbalance:   "ideal",
		BatchConfig: datasetConfigs[dataset],
		CSPAddress:  "localhost:50051",
	}
	runExperiment(t, config)
}

func runExperiment(t *testing.T, config ExperimentConfig) {
	fmt.Printf("\n=== Starting experiment for %s dataset ===\n", config.DataSet)
	fmt.Printf("Configuration: %d parties, %s split, %s imbalance\n",
		config.PartyCount, config.Split, config.Imbalance)

	mkParams := setupCryptoParams(config)

	fmt.Println("\n[Step 1] Starting CSP server...")
	server, err := startServer(config, mkParams)
	if err != nil {
		t.Fatal(fmt.Errorf("failed to start CSP server: %v", err))
	}

	cleanupFuncs := make([]func(), 0)
	cleanupFuncs = append(cleanupFuncs, func() {
		stopServer(server)
	})

	defer func() {
		for _, cleanup := range cleanupFuncs {
			cleanup()
		}
		time.Sleep(500 * time.Millisecond)
	}()

	fmt.Println("✓ CSP server started successfully")

	fmt.Println("\n[Step 2] Setting up Data Owners...")
	dataOwners := make([]*dataowner.DataOwner, config.PartyCount)
	ownerKeyGenStats := &common.TimingStats{}
	ownerModelEncStats := &common.TimingStats{}

	for i := 0; i < config.PartyCount; i++ {
		ownerID := fmt.Sprintf("owner%d", i)
		ownerPort := 50052 + i
		ownerAddress := fmt.Sprintf("localhost:%d", ownerPort)

		owner, err := dataowner.NewDataOwner(ownerID, config.DataSet, mkParams,
			config.BatchConfig, config.CSPAddress)
		if err != nil {
			t.Fatal(fmt.Errorf("failed to create data owner %s: %v", ownerID, err))
		}

		if err := owner.GenerateKeys(); err != nil {
			t.Fatal(fmt.Errorf("failed to generate keys for owner %s: %v", ownerID, err))
		}
		ownerKeyGenStats.AddSample(owner.GetTiming().KeyGeneration)

		cleanup, err := owner.StartServer(ownerAddress)
		if err != nil {
			t.Fatal(fmt.Errorf("failed to start data owner server %s: %v", ownerID, err))
		}
		cleanupFuncs = append(cleanupFuncs, cleanup)

		if err := server.cspServer.ConnectToDataOwner(ownerID, ownerAddress); err != nil {
			t.Fatal(fmt.Errorf("failed to connect CSP to data owner %s: %v", ownerID, err))
		}

		dataOwners[i] = owner
	}
	fmt.Printf("✓ All %d Data Owners created and started successfully\n", config.PartyCount)

	fmt.Println("\n[Step 3] Encrypting and enrolling models...")
	if err := enrollModels(dataOwners, config); err != nil {
		t.Fatal(fmt.Errorf("model enrollment failed: %v", err))
	}
	for _, owner := range dataOwners {
		ownerModelEncStats.AddSample(owner.GetTiming().ModelEncryption)
	}
	fmt.Printf("✓ All models encrypted and enrolled successfully\n")

	fmt.Println("\n[Step 4] Setting up Client...")
	ownerIDs := make([]string, config.PartyCount)
	for i := 0; i < config.PartyCount; i++ {
		ownerIDs[i] = fmt.Sprintf("owner%d", i)
	}

	client, err := client.NewClient("client", config.DataSet, mkParams,
		config.BatchConfig, ownerIDs, config.CSPAddress)
	if err != nil {
		t.Fatal(fmt.Errorf("failed to create client: %v", err))
	}

	if err := client.GenerateKeys(); err != nil {
		t.Fatal(fmt.Errorf("failed to generate client keys: %v", err))
	}
	fmt.Printf("✓ Client created successfully\n")

	fmt.Println("\n[Step 5] Loading test data and requesting inference...")
	testData, trueLabels, err := loadTestData(config)
	if err != nil {
		t.Fatal(fmt.Errorf("failed to load test data: %v", err))
	}
	fmt.Printf("✓ Test data loaded: %d samples\n", len(trueLabels))

	fmt.Println("\n[Step 6] Performing inference and ensemble...")
	softResult, logitResult, err := client.RequestInference(testData)
	if err != nil {
		t.Fatal(fmt.Errorf("inference request failed: %v", err))
	}
	fmt.Printf("✓ Inference completed successfully\n")

	softAccuracy := calculateAccuracy(softResult, trueLabels, config.BatchConfig.SampleCount)
	logitAccuracy := calculateAccuracy(logitResult, trueLabels, config.BatchConfig.SampleCount)

	ownerPartialDecStats := &common.TimingStats{}
	for _, owner := range dataOwners {
		ownerPartialDecStats.AddSample(owner.GetTiming().PartialDecryption)
	}

	results := map[string]ExperimentResult{
		"soft_voting": {
			Timing: TimingInfo{
				Client: struct {
					KeyGeneration             time.Duration
					DataEncryption            time.Duration
					DataTransfer              time.Duration
					SoftVotingDecryption      time.Duration
					LogitSoftVotingDecryption time.Duration
				}{
					KeyGeneration:             client.GetTiming().KeyGeneration,
					DataEncryption:            client.GetTiming().DataEncryption,
					DataTransfer:              server.cspServer.GetTiming().ClientDataTransfer,
					SoftVotingDecryption:      client.GetTiming().SoftVotingDecryption,
					LogitSoftVotingDecryption: client.GetTiming().LogitSoftVotingDecryption,
				},
				DataOwner: struct {
					KeyGeneration     common.TimingStats
					ModelEncryption   common.TimingStats
					ModelTransfer     time.Duration
					PartialDecryption common.TimingStats
				}{
					KeyGeneration:     *ownerKeyGenStats,
					ModelEncryption:   *ownerModelEncStats,
					ModelTransfer:     server.cspServer.GetTiming().ModelTransferTime,
					PartialDecryption: *ownerPartialDecStats,
				},
				CSP: struct {
					SoftVotingCompute      time.Duration
					LogitSoftVotingCompute time.Duration
					TotalDecryptionTime    time.Duration
				}{
					SoftVotingCompute:      server.cspServer.GetTiming().SoftVotingCompute,
					LogitSoftVotingCompute: server.cspServer.GetTiming().LogitSoftVotingCompute,
					TotalDecryptionTime:    client.GetTiming().TotalDecryptionTime, // CSP에서 Client로 변경
				},
			},
			Accuracy: softAccuracy,
		},
		"logit_soft_voting": {
			Timing: TimingInfo{
				Client: struct {
					KeyGeneration             time.Duration
					DataEncryption            time.Duration
					DataTransfer              time.Duration
					SoftVotingDecryption      time.Duration
					LogitSoftVotingDecryption time.Duration
				}{
					KeyGeneration:             client.GetTiming().KeyGeneration,
					DataEncryption:            client.GetTiming().DataEncryption,
					DataTransfer:              server.cspServer.GetTiming().ClientDataTransfer,
					SoftVotingDecryption:      client.GetTiming().SoftVotingDecryption,
					LogitSoftVotingDecryption: client.GetTiming().LogitSoftVotingDecryption,
				},
				DataOwner: struct {
					KeyGeneration     common.TimingStats
					ModelEncryption   common.TimingStats
					ModelTransfer     time.Duration
					PartialDecryption common.TimingStats
				}{
					KeyGeneration:     *ownerKeyGenStats,
					ModelEncryption:   *ownerModelEncStats,
					ModelTransfer:     server.cspServer.GetTiming().ModelTransferTime,
					PartialDecryption: *ownerPartialDecStats,
				},
				CSP: struct {
					SoftVotingCompute      time.Duration
					LogitSoftVotingCompute time.Duration
					TotalDecryptionTime    time.Duration
				}{
					SoftVotingCompute:      server.cspServer.GetTiming().SoftVotingCompute,
					LogitSoftVotingCompute: server.cspServer.GetTiming().LogitSoftVotingCompute,
					TotalDecryptionTime:    client.GetTiming().TotalDecryptionTime,
				},
			},
			Accuracy: logitAccuracy,
		},
	}
	if err := writeResultsToFile(config, results); err != nil {
		t.Error(err)
	}
	fmt.Println("✓ Results saved successfully")
}
