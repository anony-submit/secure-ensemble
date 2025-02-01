package mnist_ensemble

import (
	"cnn/mnist/go_inference/client"
	"cnn/mnist/go_inference/common"
	"cnn/mnist/go_inference/dataowner"
	"cnn/mnist/go_inference/server"
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

func printMemStats() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("Alloc = %v MiB", m.Alloc/1024/1024)
	fmt.Printf("\tTotalAlloc = %v MiB", m.TotalAlloc/1024/1024)
	fmt.Printf("\tSys = %v MiB", m.Sys/1024/1024)
	fmt.Printf("\tNumGC = %v\n", m.NumGC)
}

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

func TestMNISTDropout(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	partyCounts := []int{10, 20}
	dropoutRates := []float64{0.1, 0.2, 0.3}
	distributions := []string{"balanced", "dirichlet0.1", "dirichlet0.5"}

	for _, n := range partyCounts {
		for _, f := range dropoutRates {
			for _, dist := range distributions {
				testName := fmt.Sprintf("n%d_f%.1f_%s", n, f, dist)
				t.Run(testName, func(t *testing.T) {
					config := DropoutExperimentConfig{
						BaseConfig: ExperimentConfig{
							NumDataOwners:  n,
							NumTestSamples: 5,
							Distribution:   dist,
						},
						DropoutRate: f,
						NumTrials:   10,
					}
					runDropoutExperiment(t, config)
				})
				time.Sleep(2 * time.Second) // Wait between experiments
			}
		}
	}
}

func runDropoutExperiment(t *testing.T, config DropoutExperimentConfig) {
	fmt.Printf("\n=== Starting MNIST Dropout Experiment ===\n")
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

		results, err := runDropoutTrial(t, config.BaseConfig, onlineOwners)
		if err != nil {
			t.Fatalf("Trial %d failed: %v", trial+1, err)
		}

		allResults[trial] = results
		fmt.Printf("\nTrial %d Results:\n", trial+1)
		fmt.Printf("Accuracy: %.2f%%\n", results.Accuracy)
		printTimingInfo(results.Timing)
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

func runDropoutTrial(t *testing.T, config ExperimentConfig, onlineOwners []int) (ExperimentResults, error) {
	mkParams := setupParameters()
	var timing common.TimingInfo
	debug.FreeOSMemory()

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
		for i := len(cleanupFuncs) - 1; i >= 0; i-- {
			cleanupFuncs[i]()
		}
		time.Sleep(500 * time.Millisecond)
	}()

	if err := cspServer.DisconnectAllDataOwners(); err != nil {
		return ExperimentResults{}, fmt.Errorf("failed to clean previous connections: %v", err)
	}

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
	fmt.Printf("✓ %d Data Owners setup completed\n", len(onlineOwners))
	debug.FreeOSMemory()

	cleanupFuncs = append(cleanupFuncs, func() {
		cleanupDataOwners(cspServer, dataOwners)
	})

	for i, ownerIdx := range onlineOwners {
		fc1w, fc1b, fc2w, fc2b, err := loadModelParams(ExperimentConfig{
			NumDataOwners:  config.NumDataOwners,
			NumTestSamples: config.NumTestSamples,
			Distribution:   config.Distribution,
		}, ownerIdx)
		if err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to load model params for owner %d: %v", ownerIdx, err)
		}

		if err := dataOwners[i].EnrollModel(fc1w, fc1b, fc2w, fc2b); err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to enroll model for owner %d: %v", ownerIdx, err)
		}
	}
	fmt.Println("✓ Model enrollment completed")
	debug.FreeOSMemory()

	testClient, err := client.NewClient("client", mkParams, cspAddress)
	if err != nil {
		return ExperimentResults{}, fmt.Errorf("failed to create client: %v", err)
	}

	if err := testClient.GenerateKeys(); err != nil {
		return ExperimentResults{}, fmt.Errorf("failed to generate client keys: %v", err)
	}
	fmt.Println("✓ Client setup completed")
	debug.FreeOSMemory()

	correctCount := 0

	for _, owner := range dataOwners {
		ownerTiming := owner.GetTiming()
		timing.DataOwnerKeyGenStats = ownerTiming.KeyGenStats
		timing.ModelEncryptionStats = ownerTiming.ModelEncryptionStats
		timing.PartialDecryptionTime = ownerTiming.PartialDecryptionTime
	}

	for i := 0; i < config.NumTestSamples; i++ {
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
		// fmt.Printf("Predicted: %d, Actual: %d\n", predictedClass, label)
	}

	cspTiming := cspServer.GetTiming()
	clientTiming := testClient.GetTiming()

	timing.InferenceStats = cspTiming.InferenceStats
	timing.EnsembleStats = cspTiming.EnsembleStats
	timing.TotalComputeStats = cspTiming.TotalComputeStats
	timing.ClientTransferStats = cspTiming.ClientTransferStats
	timing.DataOwnerTransferTime = cspTiming.DataOwnerTransferTime
	timing.ClientKeyGenStats = clientTiming.KeyGenStats
	timing.DataEncryptionStats = clientTiming.DataEncryptionStats
	timing.FinalDecryptionStats = clientTiming.DecryptionStats
	timing.TotalDecryptionStats = clientTiming.TotalDecryptionStats

	results := ExperimentResults{
		Config:   config,
		Timing:   timing,
		Accuracy: float64(correctCount) / float64(config.NumTestSamples) * 100,
	}
	debug.FreeOSMemory()
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
=== MNIST Dropout Experiment Results ===
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

	if err := os.WriteFile(filepath, []byte(output), 0644); err != nil {
		return fmt.Errorf("failed to write results: %v", err)
	}

	fmt.Println(output)
	return nil
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
- Partial Decryption: %.6f ms

CSP Times:
- Inference Time: %.2f ± %.2f ms
- Ensemble Time: %.2f ± %.2f ms
- Total Compute Time: %.2f ± %.2f ms
- Client Transfer Latency: %.6f ms
- Data Owner Transfer Latency: %.2f ± %.2f ms
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

func printTimingInfo(timing common.TimingInfo) {
	fmt.Println("\nTiming Information:")
	fmt.Printf("Client Times:\n")
	fmt.Printf("- Key Generation: %.2f ± %.2f ms\n",
		float64(timing.ClientKeyGenStats.Mean)/float64(time.Millisecond),
		float64(timing.ClientKeyGenStats.StdDev)/float64(time.Millisecond))
	fmt.Printf("- Data Encryption: %.2f ± %.2f ms\n",
		float64(timing.DataEncryptionStats.Mean)/float64(time.Millisecond),
		float64(timing.DataEncryptionStats.StdDev)/float64(time.Millisecond))
	fmt.Printf("- Final Decryption: %.2f ± %.2f ms\n",
		float64(timing.FinalDecryptionStats.Mean)/float64(time.Millisecond),
		float64(timing.FinalDecryptionStats.StdDev)/float64(time.Millisecond))
	fmt.Printf("- Total Decryption: %.2f ± %.2f ms\n",
		float64(timing.TotalDecryptionStats.Mean)/float64(time.Millisecond),
		float64(timing.TotalDecryptionStats.StdDev)/float64(time.Millisecond))

	fmt.Printf("\nData Owner Times:\n")
	fmt.Printf("- Key Generation: %.2f ± %.2f ms\n",
		float64(timing.DataOwnerKeyGenStats.Mean)/float64(time.Millisecond),
		float64(timing.DataOwnerKeyGenStats.StdDev)/float64(time.Millisecond))
	fmt.Printf("- Model Encryption: %.2f ± %.2f ms\n",
		float64(timing.ModelEncryptionStats.Mean)/float64(time.Millisecond),
		float64(timing.ModelEncryptionStats.StdDev)/float64(time.Millisecond))
	fmt.Printf("- Partial Decryption: %.2f ms\n",
		float64(timing.PartialDecryptionTime)/float64(time.Millisecond),
	)

	fmt.Printf("\nCSP Times:\n")
	fmt.Printf("- Inference Time: %.2f ± %.2f ms\n",
		float64(timing.InferenceStats.Mean)/float64(time.Millisecond),
		float64(timing.InferenceStats.StdDev)/float64(time.Millisecond))
	fmt.Printf("- Ensemble Time: %.2f ± %.2f ms\n",
		float64(timing.EnsembleStats.Mean)/float64(time.Millisecond),
		float64(timing.EnsembleStats.StdDev)/float64(time.Millisecond))
	fmt.Printf("- Total Compute Time: %.2f ± %.2f ms\n",
		float64(timing.TotalComputeStats.Mean)/float64(time.Millisecond),
		float64(timing.TotalComputeStats.StdDev)/float64(time.Millisecond))
	fmt.Printf("- Client Transfer Latency: %.2f ± %.2f ms\n",
		float64(timing.ClientTransferStats.Mean)/float64(time.Millisecond),
		float64(timing.ClientTransferStats.StdDev)/float64(time.Millisecond))
	fmt.Printf("- Data Owner Transfer Latency: %.2f ms\n",
		float64(timing.DataOwnerTransferTime)/float64(time.Millisecond),
	)
}

func cleanupDataOwners(cspServer *server.CSPServer, dataOwners []*dataowner.DataOwner) {
	// Disconnect all Data Owner connections from CSP
	for _, owner := range dataOwners {
		if owner != nil {
			ownerID := owner.GetOwnerID()
			if err := cspServer.DisconnectFromDataOwner(ownerID); err != nil {
				fmt.Printf("Error disconnecting owner %s: %v\n", ownerID, err)
			}
		}
	}
}

// go test -v -timeout 0 -run TestMNISTEnsemble
// go test -v -timeout 0 -run TestMNISTDropout
