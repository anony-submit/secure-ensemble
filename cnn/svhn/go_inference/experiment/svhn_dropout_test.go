package svhn_ensemble

import (
	"cnn/svhn/go_inference/client"
	"cnn/svhn/go_inference/common"
	"cnn/svhn/go_inference/dataowner"
	"cnn/svhn/go_inference/server"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"testing"
	"time"
)

type DropoutExperimentConfig struct {
	BaseConfig  ExperimentConfig
	DropoutRate float64
	NumTrials   int
}

type DropoutExperimentResult struct {
	Mean    float64
	StdDev  float64
	Results []ExperimentResults
}

func TestSVHNDropout(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	// Entire Config
	// partyCounts := []int{10, 20}
	// dropoutRates := []float64{0.1, 0.3}
	// distributions := []string{"balanced", "dirichlet0.1", "dirichlet0.5"}

	partyCounts := []int{10}
	dropoutRates := []float64{0.3}
	distributions := []string{"dirichlet0.5"}

	for _, n := range partyCounts {
		for _, f := range dropoutRates {
			for _, dist := range distributions {
				testName := fmt.Sprintf("n%d_f%.1f_%s", n, f, dist)
				t.Run(testName, func(t *testing.T) {
					config := DropoutExperimentConfig{
						BaseConfig: ExperimentConfig{
							NumDataOwners:  n,
							NumTestSamples: 100,
							Distribution:   dist,
						},
						DropoutRate: f,
						NumTrials:   1,
					}
					runDropoutExperiment(t, config)
				})
				time.Sleep(2 * time.Second)
			}
		}
	}
}

func runDropoutExperiment(t *testing.T, config DropoutExperimentConfig) {
	printMemStats()

	fmt.Printf("\n=== Starting SVHN Dropout Experiment ===\n")
	fmt.Printf("Configuration: %d data owners, dropout rate %.1f, distribution: %s\n",
		config.BaseConfig.NumDataOwners, config.DropoutRate, config.BaseConfig.Distribution)
	fmt.Printf("Number of trials: %d\n", config.NumTrials)

	onlineCount := int(float64(config.BaseConfig.NumDataOwners) * (1 - config.DropoutRate))
	allResults := make([]ExperimentResults, config.NumTrials)

	for trial := 0; trial < config.NumTrials; trial++ {
		fmt.Printf("\n\n=== Trial %d/%d ===\n", trial+1, config.NumTrials)
		fmt.Printf("Using %d out of %d owners\n", onlineCount, config.BaseConfig.NumDataOwners)

		onlineOwners := selectRandomOnlineOwners(config.BaseConfig.NumDataOwners, onlineCount)
		fmt.Printf("Selected owners: %v\n", onlineOwners)

		results, err := runDropoutTrial(config.BaseConfig, onlineOwners)
		if err != nil {
			t.Fatalf("Trial %d failed: %v", trial+1, err)
		}

		allResults[trial] = results
		fmt.Printf("\nTrial %d Results:\n", trial+1)
		fmt.Printf("Accuracy: %.2f%%\n", results.Accuracy)
		debug.FreeOSMemory()
	}

	accuracies := make([]float64, len(allResults))
	for i, result := range allResults {
		accuracies[i] = result.Accuracy
	}

	dropoutResults := DropoutExperimentResult{
		Mean:    calculateMean(accuracies),
		StdDev:  calculateStdDev(accuracies),
		Results: allResults,
	}

	if err := writeDropoutResultsToFile(config, dropoutResults); err != nil {
		t.Error(fmt.Errorf("failed to write results: %v", err))
	}
}

func runDropoutTrial(config ExperimentConfig, onlineOwners []int) (ExperimentResults, error) {
	mkParams := setupParameters()
	var timing common.TimingInfo
	debug.FreeOSMemory()

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
	dataOwners := make([]*dataowner.DataOwner, len(onlineOwners))

	for i, ownerIdx := range onlineOwners {
		ownerID := fmt.Sprintf("owner%d", ownerIdx)

		address, err := common.GetAvailableAddress()
		if err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to get address for owner %d: %v", ownerIdx, err)
		}

		owner, err := dataowner.NewDataOwner(ownerID, mkParams, cspAddress)
		if err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to create data owner %d: %v", ownerIdx, err)
		}

		if err := owner.GenerateKeys(); err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to generate keys for owner %d: %v", ownerIdx, err)
		}

		cleanup, err := owner.StartServer(address)
		if err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to start owner server %d: %v", ownerIdx, err)
		}
		cleanupFuncs = append(cleanupFuncs, cleanup)

		if err := cspServer.ConnectToDataOwner(ownerID, address); err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to connect CSP to owner %d: %v", ownerIdx, err)
		}

		dataOwners[i] = owner
	}
	fmt.Printf("✓ All %d Data Owners created and started successfully\n", len(onlineOwners))

	fmt.Println("\n[Step 3] Loading and enrolling models...")
	for i, ownerIdx := range onlineOwners {
		modelPath := getModelPath(config, ownerIdx)
		modelParams, err := loadModelParams(modelPath)
		if err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to load model %d: %v", ownerIdx, err)
		}

		if err := dataOwners[i].EnrollModel(modelParams); err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to enroll model %d: %v", ownerIdx, err)
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
	}

	cspTiming := cspServer.GetTiming()
	timing.InferenceStats = cspTiming.InferenceStats
	timing.EnsembleStats = cspTiming.EnsembleStats
	timing.TotalComputeStats = cspTiming.TotalComputeStats
	timing.ClientTransferStats = cspTiming.ClientTransferStats
	timing.DataOwnerTransferTime = cspTiming.DataOwnerTransferTime

	clientTiming := testClient.GetTiming()
	timing.ClientKeyGenStats = clientTiming.KeyGenStats
	timing.DataEncryptionStats = clientTiming.DataEncryptionStats
	timing.FinalDecryptionStats = clientTiming.DecryptionStats
	timing.TotalDecryptionStats = clientTiming.TotalDecryptionStats

	results := ExperimentResults{
		Config:   config,
		Timing:   timing,
		Accuracy: float64(correctCount) / float64(config.NumTestSamples) * 100,
	}

	return results, nil
}

func selectRandomOnlineOwners(totalCount, onlineCount int) []int {
	allOwners := make([]int, totalCount)
	for i := range allOwners {
		allOwners[i] = i
	}

	rand.Shuffle(len(allOwners), func(i, j int) {
		allOwners[i], allOwners[j] = allOwners[j], allOwners[i]
	})

	return allOwners[:onlineCount]
}

func calculateMean(values []float64) float64 {
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func calculateStdDev(values []float64) float64 {
	mean := calculateMean(values)
	sumSquares := 0.0
	for _, v := range values {
		diff := v - mean
		sumSquares += diff * diff
	}
	variance := sumSquares / float64(len(values))
	return math.Sqrt(variance)
}

func writeDropoutResultsToFile(config DropoutExperimentConfig, results DropoutExperimentResult) error {
	resultsDir := filepath.Join("results", "dropout")
	if err := os.MkdirAll(resultsDir, 0755); err != nil {
		return fmt.Errorf("failed to create results directory: %v", err)
	}

	filename := fmt.Sprintf("dropout_n%d_f%.1f_%s_%s.txt",
		config.BaseConfig.NumDataOwners,
		config.DropoutRate,
		config.BaseConfig.Distribution,
		time.Now().Format("20060102_150405"))

	filepath := filepath.Join(resultsDir, filename)

	output := fmt.Sprintf(`
=== SVHN Dropout Experiment Results ===
Number of Data Owners: %d
Dropout Rate: %.1f
Distribution: %s
Number of Test Samples: %d
Number of Trials: %d

Overall Results:
Mean Accuracy: %.2f%%
StdDev Accuracy: %.2f%%

Individual Trial Results:
`,
		config.BaseConfig.NumDataOwners,
		config.DropoutRate,
		config.BaseConfig.Distribution,
		config.BaseConfig.NumTestSamples,
		config.NumTrials,
		results.Mean,
		results.StdDev)

	for i, result := range results.Results {
		output += fmt.Sprintf("\nTrial %d:\n", i+1)
		output += fmt.Sprintf("Accuracy: %.2f%%\n", result.Accuracy)
		output += formatTimingInfo(result.Timing)
	}

	return os.WriteFile(filepath, []byte(output), 0644)
}

func printMemStats() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("Alloc = %v MiB", m.Alloc/1024/1024)
	fmt.Printf("\tTotalAlloc = %v MiB", m.TotalAlloc/1024/1024)
	fmt.Printf("\tSys = %v MiB", m.Sys/1024/1024)
	fmt.Printf("\tNumGC = %v\n", m.NumGC)
}

func formatTimingInfo(timing common.TimingInfo) string {
	return fmt.Sprintf(`
Client Times:
- Key Generation: %.2f ± %.2f ms
- Data Encryption: %.2f ± %.2f ms
- Final Decryption: %.2f ± %.2f ms
- Total Decryption: %.2f ± %.2f ms

Data Owner Times:
- Key Generation: %.2f ± %.2f ms
- Model Encryption: %.2f ± %.2f ms
- Partial Decryption: %.2f ms

CSP Times:
- Inference Time: %.2f ± %.2f ms
- Ensemble Time: %.2f ± %.2f ms
- Total Compute Time: %.2f ± %.2f ms
- Client Transfer Latency: %.2f ± %.2f ms
- Data Owner Transfer Latency: %.2f ms
`,
		float64(timing.ClientKeyGenStats.Mean)/float64(time.Millisecond),
		float64(timing.ClientKeyGenStats.StdDev)/float64(time.Millisecond),
		float64(timing.DataEncryptionStats.Mean)/float64(time.Millisecond),
		float64(timing.DataEncryptionStats.StdDev)/float64(time.Millisecond),
		float64(timing.FinalDecryptionStats.Mean)/float64(time.Millisecond),
		float64(timing.FinalDecryptionStats.StdDev)/float64(time.Millisecond),
		float64(timing.TotalDecryptionStats.Mean)/float64(time.Millisecond),
		float64(timing.TotalDecryptionStats.StdDev)/float64(time.Millisecond),
		float64(timing.DataOwnerKeyGenStats.Mean)/float64(time.Millisecond),
		float64(timing.DataOwnerKeyGenStats.StdDev)/float64(time.Millisecond),
		float64(timing.ModelEncryptionStats.Mean)/float64(time.Millisecond),
		float64(timing.ModelEncryptionStats.StdDev)/float64(time.Millisecond),
		float64(timing.PartialDecryptionTime)/float64(time.Millisecond),
		float64(timing.InferenceStats.Mean)/float64(time.Millisecond),
		float64(timing.InferenceStats.StdDev)/float64(time.Millisecond),
		float64(timing.EnsembleStats.Mean)/float64(time.Millisecond),
		float64(timing.EnsembleStats.StdDev)/float64(time.Millisecond),
		float64(timing.TotalComputeStats.Mean)/float64(time.Millisecond),
		float64(timing.TotalComputeStats.StdDev)/float64(time.Millisecond),
		float64(timing.ClientTransferStats.Mean)/float64(time.Millisecond),
		float64(timing.ClientTransferStats.StdDev)/float64(time.Millisecond),
		float64(timing.DataOwnerTransferTime)/float64(time.Millisecond),
	)
}

// go test -v -timeout 0 -run TestSVHNDropout
