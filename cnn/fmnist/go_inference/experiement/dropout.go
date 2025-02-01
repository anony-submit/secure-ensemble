package experiment

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"time"

	"cnn/fmnist/go_inference/client"
	"cnn/fmnist/go_inference/common"
	"cnn/fmnist/go_inference/dataowner"
	"cnn/fmnist/go_inference/server"

	"github.com/anony-submit/snu-mghe/mkckks"
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

func RunDropoutExperiment(config DropoutExperimentConfig) (DropoutExperimentResult, error) {
	fmt.Printf("\n=== Starting FMNIST Dropout Experiment ===\n")
	fmt.Printf("Configuration: %d data owners, dropout rate %.1f, distribution: %s\n",
		config.BaseConfig.NumDataOwners, config.DropoutRate, config.BaseConfig.Distribution)
	fmt.Printf("Number of trials: %d\n", config.NumTrials)

	mkParams := setupParameters()
	cspServer := server.NewCSPServer(mkParams)
	cspAddress, err := common.GetAvailableAddress()
	if err != nil {
		return DropoutExperimentResult{}, fmt.Errorf("failed to get address for CSP: %v", err)
	}
	if err := cspServer.Start(cspAddress); err != nil {
		return DropoutExperimentResult{}, fmt.Errorf("failed to start CSP server: %v", err)
	}
	defer cspServer.Close()

	onlineCount := int(float64(config.BaseConfig.NumDataOwners) * (1 - config.DropoutRate))
	allResults := make([]ExperimentResults, config.NumTrials)

	for trial := 0; trial < config.NumTrials; trial++ {
		fmt.Printf("\n\n=== Trial %d/%d ===\n", trial+1, config.NumTrials)
		fmt.Printf("Using %d out of %d owners\n", onlineCount, config.BaseConfig.NumDataOwners)

		runtime.GC()
		debug.FreeOSMemory()

		if err := cspServer.DisconnectAllDataOwners(); err != nil {
			return DropoutExperimentResult{}, fmt.Errorf("failed to clean previous connections: %v", err)
		}

		onlineOwners := selectRandomOnlineOwners(config.BaseConfig.NumDataOwners, onlineCount)
		fmt.Printf("Selected owners: %v\n", onlineOwners)

		results, err := runDropoutTrial(config.BaseConfig, onlineOwners, mkParams, cspServer, cspAddress)
		if err != nil {
			return DropoutExperimentResult{}, fmt.Errorf("trial %d failed: %v", trial+1, err)
		}

		allResults[trial] = results
		fmt.Printf("\nTrial %d Results:\n", trial+1)
		fmt.Printf("Accuracy: %.2f%%\n", results.Accuracy)

		time.Sleep(2 * time.Second)
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

	if err := writeDropoutResults(config, dropoutResults); err != nil {
		return dropoutResults, fmt.Errorf("failed to write results: %v", err)
	}

	return dropoutResults, nil
}

func runDropoutTrial(config ExperimentConfig, onlineOwners []int, mkParams mkckks.Parameters, cspServer *server.CSPServer, cspAddress string) (ExperimentResults, error) {
	var timing common.TimingInfo
	debug.FreeOSMemory()

	dataOwners := make([]*dataowner.DataOwner, len(onlineOwners))
	cleanupFuncs := make([]func(), 0)

	defer func() {
		for i := len(cleanupFuncs) - 1; i >= 0; i-- {
			cleanupFuncs[i]()
		}
		time.Sleep(500 * time.Millisecond)
	}()

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

	for i, ownerIdx := range onlineOwners {
		convParams, fc1w, fc1b, fc2w, fc2b, err := loadModelParams(config, ownerIdx)
		if err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to load model params for owner %d: %v", ownerIdx, err)
		}

		if err := dataOwners[i].EnrollModel(convParams.Kernels, convParams.Bias, fc1w, fc1b, fc2w, fc2b); err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to enroll model for owner %d: %v", ownerIdx, err)
		}
	}

	testClient, err := client.NewClient("client", mkParams, cspAddress)
	if err != nil {
		return ExperimentResults{}, fmt.Errorf("failed to create client: %v", err)
	}

	if err := testClient.GenerateKeys(); err != nil {
		return ExperimentResults{}, fmt.Errorf("failed to generate client keys: %v", err)
	}

	correctCount := 0

	for _, owner := range dataOwners {
		ownerTiming := owner.GetTiming()
		timing.DataOwnerKeyGenStats = ownerTiming.KeyGenStats
		timing.ModelEncryptionStats = ownerTiming.ModelEncryptionStats
		timing.PartialDecryptionStats = ownerTiming.PartialDecryptionStats
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
	}

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

	debug.FreeOSMemory()

	return ExperimentResults{
		Config:   config,
		Timing:   timing,
		Accuracy: float64(correctCount) / float64(config.NumTestSamples) * 100,
	}, nil
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

func writeDropoutResults(config DropoutExperimentConfig, results DropoutExperimentResult) error {
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
	file, err := os.Create(filepath)
	if err != nil {
		return fmt.Errorf("failed to create result file: %v", err)
	}
	defer file.Close()

	writeDropoutResultsContent(file, config, results)
	return nil
}

func writeDropoutResultsContent(file *os.File, config DropoutExperimentConfig, results DropoutExperimentResult) {
	fmt.Fprintf(file, "=== FMNIST Dropout Experiment Results ===\n")
	fmt.Fprintf(file, "Number of Data Owners: %d\n", config.BaseConfig.NumDataOwners)
	fmt.Fprintf(file, "Dropout Rate: %.1f\n", config.DropoutRate)
	fmt.Fprintf(file, "Distribution: %s\n", config.BaseConfig.Distribution)
	fmt.Fprintf(file, "Number of Test Samples: %d\n", config.BaseConfig.NumTestSamples)
	fmt.Fprintf(file, "Number of Trials: %d\n\n", config.NumTrials)

	fmt.Fprintf(file, "=== Overall Results ===\n")
	fmt.Fprintf(file, "Mean Accuracy: %.2f%%\n", results.Mean)
	fmt.Fprintf(file, "StdDev Accuracy: %.2f%%\n\n", results.StdDev)

	for i, result := range results.Results {
		fmt.Fprintf(file, "=== Trial %d ===\n", i+1)
		fmt.Fprintf(file, "Accuracy: %.2f%%\n", result.Accuracy)
		writeTimingInfo(file, result.Timing)
		fmt.Fprintf(file, "\n")
	}
}
