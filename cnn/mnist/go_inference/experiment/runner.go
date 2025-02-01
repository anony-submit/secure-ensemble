package mnist_ensemble

import (
	"cnn/mnist/go_inference/client"
	"cnn/mnist/go_inference/common"
	"cnn/mnist/go_inference/dataowner"
	"cnn/mnist/go_inference/server"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/anony-submit/snu-mghe/mkckks"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
)

type ExperimentConfig struct {
	NumDataOwners  int
	NumTestSamples int
	Distribution   string
}

type ModelParams struct {
	FC1Weights [][]float64 `json:"fc1.weight"`
	FC1Bias    []float64   `json:"fc1.bias"`
	FC2Weights [][]float64 `json:"fc2.weight"`
	FC2Bias    []float64   `json:"fc2.bias"`
}

type ExperimentResults struct {
	Config   ExperimentConfig
	Timing   common.TimingInfo
	Accuracy float64
}

func getModelPath(config ExperimentConfig, ownerIndex int) string {
	baseDir := fmt.Sprintf("../data/n%d", config.NumDataOwners)
	if config.NumDataOwners == 1 {
		return filepath.Join(baseDir, "model_params.json")
	}
	return filepath.Join(baseDir, config.Distribution, fmt.Sprintf("model_params%d.json", ownerIndex+1))
}

func loadModelParams(config ExperimentConfig, ownerIndex int) ([][]float64, []float64, [][]float64, []float64, error) {
	modelPath := getModelPath(config, ownerIndex)
	modelFile, err := os.ReadFile(modelPath)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("failed to read model file at %s: %v", modelPath, err)
	}

	var modelParams ModelParams
	err = json.Unmarshal(modelFile, &modelParams)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("failed to unmarshal model params from %s: %v", modelPath, err)
	}

	return modelParams.FC1Weights, modelParams.FC1Bias, modelParams.FC2Weights, modelParams.FC2Bias, nil
}

func setupParameters() mkckks.Parameters {
	params, _ := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:     14,
		LogSlots: 13,
		Q: []uint64{
			0x2000000002b0001,
			0x800000020001, 0x800000280001,
			0x800000520001, 0x800000770001,
			0x800000aa0001, 0x800000ad0001,
		},
		P: []uint64{
			0x800000df0001, 0x800000f80001,
		},
		Scale: 1 << 47,
		Sigma: rlwe.DefaultSigma,
	})

	mkParams := mkckks.NewParameters(params)
	rotations := []int{1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
		34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
		128, 256, 512, 1024, 2048, 4096}

	for _, rot := range rotations {
		mkParams.AddCRS(rot)
	}

	return mkParams
}

func saveResults(config ExperimentConfig, results ExperimentResults) error {
	resultsDir := "results"
	if err := os.MkdirAll(resultsDir, 0755); err != nil {
		return fmt.Errorf("failed to create results directory: %v", err)
	}

	filename := fmt.Sprintf("n%d", config.NumDataOwners)
	if config.Distribution != "" {
		filename = fmt.Sprintf("%s_%s", filename, config.Distribution)
	}
	filepath := filepath.Join(resultsDir, filename+".txt")

	file, err := os.Create(filepath)
	if err != nil {
		return fmt.Errorf("failed to create result file: %v", err)
	}
	defer file.Close()

	fmt.Fprintf(file, "=== Experiment Configuration ===\n")
	fmt.Fprintf(file, "Number of Data Owners: %d\n", config.NumDataOwners)
	fmt.Fprintf(file, "Number of Test Samples: %d\n", config.NumTestSamples)
	fmt.Fprintf(file, "Distribution: %s\n\n", config.Distribution)

	fmt.Fprintf(file, "=== Accuracy Results ===\n")
	fmt.Fprintf(file, "Accuracy: %.2f%%\n\n", results.Accuracy)

	fmt.Fprintf(file, "=== Client Times ===\n")
	fmt.Fprintf(file, "Key Generation: %.6f ± %.6f ms\n",
		float64(results.Timing.ClientKeyGenStats.Mean)/float64(time.Millisecond),
		float64(results.Timing.ClientKeyGenStats.StdDev)/float64(time.Millisecond))
	fmt.Fprintf(file, "Data Encryption: %.6f ± %.6f ms\n",
		float64(results.Timing.DataEncryptionStats.Mean)/float64(time.Millisecond),
		float64(results.Timing.DataEncryptionStats.StdDev)/float64(time.Millisecond))
	fmt.Fprintf(file, "Final Decryption: %.6f ± %.6f ms\n",
		float64(results.Timing.FinalDecryptionStats.Mean)/float64(time.Millisecond),
		float64(results.Timing.FinalDecryptionStats.StdDev)/float64(time.Millisecond))
	fmt.Fprintf(file, "Total Decryption: %.6f ± %.6f ms\n\n",
		float64(results.Timing.TotalDecryptionStats.Mean)/float64(time.Millisecond),
		float64(results.Timing.TotalDecryptionStats.StdDev)/float64(time.Millisecond))

	fmt.Fprintf(file, "=== Data Owner Times ===\n")
	fmt.Fprintf(file, "Key Generation: %.6f ± %.6f ms\n",
		float64(results.Timing.DataOwnerKeyGenStats.Mean)/float64(time.Millisecond),
		float64(results.Timing.DataOwnerKeyGenStats.StdDev)/float64(time.Millisecond))
	fmt.Fprintf(file, "Model Encryption: %.6f ± %.6f ms\n",
		float64(results.Timing.ModelEncryptionStats.Mean)/float64(time.Millisecond),
		float64(results.Timing.ModelEncryptionStats.StdDev)/float64(time.Millisecond))
	fmt.Fprintf(file, "Partial Decryption: %.6f ms\n\n",
		float64(results.Timing.PartialDecryptionTime)/float64(time.Millisecond))

	fmt.Fprintf(file, "=== CSP Times ===\n")
	fmt.Fprintf(file, "Inference Time: %.6f ± %.6f ms\n",
		float64(results.Timing.InferenceStats.Mean)/float64(time.Millisecond),
		float64(results.Timing.InferenceStats.StdDev)/float64(time.Millisecond))
	fmt.Fprintf(file, "Ensemble Time: %.6f ± %.6f ms\n",
		float64(results.Timing.EnsembleStats.Mean)/float64(time.Millisecond),
		float64(results.Timing.EnsembleStats.StdDev)/float64(time.Millisecond))
	fmt.Fprintf(file, "Total Compute Time: %.6f ± %.6f ms\n",
		float64(results.Timing.TotalComputeStats.Mean)/float64(time.Millisecond),
		float64(results.Timing.TotalComputeStats.StdDev)/float64(time.Millisecond))
	fmt.Fprintf(file, "Client Transfer Latency: %.6f ± %.6f ms\n",
		float64(results.Timing.ClientTransferStats.Mean)/float64(time.Millisecond),
		float64(results.Timing.ClientTransferStats.StdDev)/float64(time.Millisecond))
	fmt.Fprintf(file, "Data Owner Transfer Time: %.6f ms\n",
		float64(results.Timing.DataOwnerTransferTime)/float64(time.Millisecond))

	return nil
}

func RunExperiment(config ExperimentConfig) (ExperimentResults, error) {
	fmt.Printf("\n=== Starting MNIST Ensemble Experiment ===\n")
	fmt.Printf("Configuration: %d data owners, %d test samples, distribution: %s\n",
		config.NumDataOwners, config.NumTestSamples, config.Distribution)

	mkParams := setupParameters()
	var timing common.TimingInfo

	fmt.Println("\n[Step 1] Starting CSP server...")
	cspServer := server.NewCSPServer(mkParams)
	cspAddress, err := common.GetAvailableAddress()
	if err != nil {
		return ExperimentResults{}, fmt.Errorf("failed to get address for CSP: %v", err)
	}
	if err := cspServer.Start(cspAddress); err != nil {
		return ExperimentResults{}, fmt.Errorf("failed to start CSP server: %v", err)
	}

	cleanupFuncs := make([]func(), 0)
	cleanupFuncs = append(cleanupFuncs, func() {
		if err := cspServer.Close(); err != nil {
			fmt.Printf("Error closing CSP server: %v\n", err)
		}
	})

	defer func() {
		for _, cleanup := range cleanupFuncs {
			cleanup()
		}
		time.Sleep(500 * time.Millisecond)
	}()

	fmt.Println("✓ CSP server started successfully")

	fmt.Println("\n[Step 2] Setting up Data Owners...")
	dataOwners := make([]*dataowner.DataOwner, config.NumDataOwners)

	for i := 0; i < config.NumDataOwners; i++ {
		ownerID := fmt.Sprintf("owner%d", i)

		address, err := common.GetAvailableAddress()
		if err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to get address for owner %d: %v", i, err)
		}

		owner, err := dataowner.NewDataOwner(ownerID, mkParams, cspAddress)
		if err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to create data owner %d: %v", i, err)
		}

		if err := owner.GenerateKeys(); err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to generate keys for owner %d: %v", i, err)
		}

		cleanup, err := owner.StartServer(address)
		if err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to start owner server %d: %v", i, err)
		}
		cleanupFuncs = append(cleanupFuncs, cleanup)

		if err := cspServer.ConnectToDataOwner(ownerID, address); err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to connect CSP to owner %d: %v", i, err)
		}

		dataOwners[i] = owner
	}
	fmt.Printf("✓ All %d Data Owners created and started successfully\n", config.NumDataOwners)

	fmt.Println("\n[Step 3] Loading and enrolling models...")
	for i := 0; i < config.NumDataOwners; i++ {
		fc1w, fc1b, fc2w, fc2b, err := loadModelParams(config, i)
		if err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to load model params for owner %d: %v", i, err)
		}

		if err := dataOwners[i].EnrollModel(fc1w, fc1b, fc2w, fc2b); err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to enroll model for owner %d: %v", i, err)
		}
	}
	fmt.Println("✓ All models loaded and enrolled successfully")

	fmt.Println("\n[Step 4] Setting up Client...")
	testClient, err := client.NewClient("client", mkParams, cspAddress)
	if err != nil {
		return ExperimentResults{}, fmt.Errorf("failed to create client: %v", err)
	}

	if err := testClient.GenerateKeys(); err != nil {
		return ExperimentResults{}, fmt.Errorf("failed to generate client keys: %v", err)
	}
	fmt.Println("✓ Client setup completed successfully")

	fmt.Println("\n[Step 5] Performing inference...")
	correctCount := 0

	for _, owner := range dataOwners {
		ownerTiming := owner.GetTiming()
		timing.DataOwnerKeyGenStats = ownerTiming.KeyGenStats
		timing.ModelEncryptionStats = ownerTiming.ModelEncryptionStats
		timing.PartialDecryptionTime = ownerTiming.PartialDecryptionTime
	}

	for i := 0; i < config.NumTestSamples; i++ {
		if i%10 == 0 {
			fmt.Printf("Processing test sample %d/%d\n", i+1, config.NumTestSamples)
		}

		input, label, err := testClient.LoadTestData(i)
		if err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to load test data %d: %v", i, err)
		}

		scores, err := testClient.RequestInference(input)
		if err != nil {
			return ExperimentResults{}, fmt.Errorf("inference failed for sample %d: %v", i, err)
		}

		predictedClass := getPredictedClass(scores)
		if predictedClass == label {
			correctCount++
		}
		// fmt.Printf("\npredict: %d", predictedClass)
		// fmt.Printf("\nreal: %d\n", label)
	}

	fmt.Println("\n[Step 6] Collecting timing information...")
	cspTiming := cspServer.GetTiming()
	timing.InferenceStats = cspTiming.InferenceStats
	timing.EnsembleStats = cspTiming.EnsembleStats
	timing.TotalComputeStats = cspTiming.TotalComputeStats
	timing.ClientTransferStats = cspTiming.ClientTransferStats
	timing.DataOwnerTransferTime = cspTiming.DataOwnerTransferTime

	clientTiming := testClient.GetTiming()
	timing.DataEncryptionStats = clientTiming.DataEncryptionStats
	timing.FinalDecryptionStats = clientTiming.DecryptionStats
	timing.TotalDecryptionStats = clientTiming.TotalDecryptionStats
	timing.ClientKeyGenStats = clientTiming.KeyGenStats

	results := ExperimentResults{
		Config:   config,
		Timing:   timing,
		Accuracy: float64(correctCount) / float64(config.NumTestSamples) * 100,
	}

	if err := saveResults(config, results); err != nil {
		return results, fmt.Errorf("failed to save results: %v", err)
	}

	return results, nil
}

func getPredictedClass(scores []float64) int {
	maxScore := scores[0]
	predictedClass := 0
	for i := 1; i < len(scores); i++ {
		if scores[i] > maxScore {
			maxScore = scores[i]
			predictedClass = i
		}
	}
	return predictedClass
}
