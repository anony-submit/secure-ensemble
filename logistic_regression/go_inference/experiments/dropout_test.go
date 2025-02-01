package experiments

import (
	"fmt"
	"logistic_regression/go_inference/client"
	"logistic_regression/go_inference/common"
	"logistic_regression/go_inference/dataowner"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestDropoutExperiments(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	datasets := []string{"wdbc", "heart_disease", "pima"}
	splits := []string{"horizontal", "vertical"}
	imbalances := []string{"balanced", "dirichlet_0.1", "dirichlet_0.5"}
	dropoutRates := []float64{0.1, 0.2, 0.3}
	partyCounts := []int{10, 20}

	for _, dataset := range datasets {
		for _, split := range splits {
			for _, imbalance := range imbalances {
				for _, n := range partyCounts {
					if split == "vertical" && n > datasetConfigs[dataset].MaxVerticalParties {
						continue
					}

					for _, f := range dropoutRates {
						testName := fmt.Sprintf("%s_%s_%s_N%d_F%.1f",
							dataset, split, imbalance, n, f)
						t.Run(testName, func(t *testing.T) {
							config := DropoutExperimentConfig{
								BaseConfig: ExperimentConfig{
									DataSet:     dataset,
									PartyCount:  n,
									Split:       split,
									Imbalance:   imbalance,
									BatchConfig: datasetConfigs[dataset],
									CSPAddress:  "localhost:50051",
								},
								DropoutRate: f,
								NumTrials:   10,
							}
							runDropoutExperiment(t, config)
						})
					}
				}
			}
		}
	}
}

func runDropoutExperiment(t *testing.T, config DropoutExperimentConfig) {
	fmt.Printf("\n=== Starting Dropout Experiment ===\n")
	fmt.Printf("Dataset: %s, N: %d, F: %.1f, Split: %s, Imbalance: %s\n",
		config.BaseConfig.DataSet, config.BaseConfig.PartyCount,
		config.DropoutRate, config.BaseConfig.Split, config.BaseConfig.Imbalance)

	onlineCount := int(float64(config.BaseConfig.PartyCount) * (1 - config.DropoutRate))
	softVotingResults := make([]float64, config.NumTrials)
	logitSoftVotingResults := make([]float64, config.NumTrials)

	for trial := 0; trial < config.NumTrials; trial++ {
		fmt.Printf("\nTrial %d/%d: Using %d out of %d owners\n",
			trial+1, config.NumTrials, onlineCount, config.BaseConfig.PartyCount)

		onlineOwners := selectRandomOnlineOwners(config.BaseConfig.PartyCount, onlineCount)
		fmt.Printf("\nTrial %d selected owners: %v\n", trial+1, onlineOwners)

		trialResult, err := runDropoutTrial(t, config.BaseConfig, onlineOwners)
		if err != nil {
			t.Fatal(fmt.Errorf("trial %d failed: %v", trial+1, err))
		}

		softVotingResults[trial] = trialResult.SoftVoting.Accuracy.Percentage
		logitSoftVotingResults[trial] = trialResult.LogitSoftVoting.Accuracy.Percentage

		fmt.Printf("Trial %d Results:\n", trial+1)
		fmt.Printf("  Selected owners: %v\n", onlineOwners)
		fmt.Printf("  Soft Voting: %.2f%%\n", softVotingResults[trial])
		fmt.Printf("  Logit Soft Voting: %.2f%%\n", logitSoftVotingResults[trial])
	}

	dropoutResults := DropoutExperimentResult{
		SoftVoting: DropoutAccuracyInfo{
			Mean:    calculateMean(softVotingResults),
			StdDev:  calculateStdDev(softVotingResults),
			Results: softVotingResults,
		},
		LogitSoftVoting: DropoutAccuracyInfo{
			Mean:    calculateMean(logitSoftVotingResults),
			StdDev:  calculateStdDev(logitSoftVotingResults),
			Results: logitSoftVotingResults,
		},
	}

	if err := writeDropoutResultsToFile(config, dropoutResults); err != nil {
		t.Error(fmt.Errorf("failed to write results: %v", err))
	}
}

func runDropoutTrial(t *testing.T, config ExperimentConfig, onlineOwners []int) (ExperimentResults, error) {
	t.Logf("Starting dropout trial with %d online owners", len(onlineOwners))
	mkParams := setupCryptoParams(config)

	server, err := startServer(config, mkParams)
	if err != nil {
		return ExperimentResults{}, fmt.Errorf("failed to start CSP server: %v", err)
	}
	defer stopServer(server)

	dataOwners := make([]*dataowner.DataOwner, len(onlineOwners))
	ownerKeyGenStats := &common.TimingStats{}
	ownerModelEncStats := &common.TimingStats{}
	ownerPartialDecStats := &common.TimingStats{}

	for i, ownerIdx := range onlineOwners {
		ownerID := fmt.Sprintf("owner%d", ownerIdx)
		ownerPort := 50052 + ownerIdx
		ownerAddress := fmt.Sprintf("localhost:%d", ownerPort)

		owner, err := dataowner.NewDataOwner(ownerID, config.DataSet, mkParams,
			config.BatchConfig, config.CSPAddress)
		if err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to create data owner %s: %v", ownerID, err)
		}

		if err := owner.GenerateKeys(); err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to generate keys for owner %s: %v", ownerID, err)
		}
		ownerKeyGenStats.AddSample(owner.GetTiming().KeyGeneration)

		cleanup, err := owner.StartServer(ownerAddress)
		if err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to start data owner server %s: %v", ownerID, err)
		}
		defer cleanup()

		if err := server.cspServer.ConnectToDataOwner(ownerID, ownerAddress); err != nil {
			return ExperimentResults{}, fmt.Errorf("failed to connect CSP to data owner %s: %v", ownerID, err)
		}

		dataOwners[i] = owner
	}

	if err := enrollModels(dataOwners, config); err != nil {
		return ExperimentResults{}, fmt.Errorf("model enrollment failed: %v", err)
	}

	for _, owner := range dataOwners {
		ownerModelEncStats.AddSample(owner.GetTiming().ModelEncryption)
		ownerPartialDecStats.AddSample(owner.GetTiming().PartialDecryption)
	}

	ownerIDs := make([]string, len(onlineOwners))
	for i, ownerIdx := range onlineOwners {
		ownerIDs[i] = fmt.Sprintf("owner%d", ownerIdx)
	}

	client, err := client.NewClient("client", config.DataSet, mkParams,
		config.BatchConfig, ownerIDs, config.CSPAddress)
	if err != nil {
		return ExperimentResults{}, fmt.Errorf("failed to create client: %v", err)
	}

	if err := client.GenerateKeys(); err != nil {
		return ExperimentResults{}, fmt.Errorf("failed to generate client keys: %v", err)
	}

	testData, trueLabels, err := loadTestData(config)
	if err != nil {
		return ExperimentResults{}, fmt.Errorf("failed to load test data: %v", err)
	}

	softResult, logitResult, err := client.RequestInference(testData)
	if err != nil {
		return ExperimentResults{}, fmt.Errorf("inference request failed: %v", err)
	}

	timing := TimingInfo{
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
	}

	return ExperimentResults{
		SoftVoting: ExperimentResult{
			Timing:   timing,
			Accuracy: calculateAccuracy(softResult, trueLabels, config.BatchConfig.SampleCount),
		},
		LogitSoftVoting: ExperimentResult{
			Timing:   timing,
			Accuracy: calculateAccuracy(logitResult, trueLabels, config.BatchConfig.SampleCount),
		},
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
	if len(values) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func calculateStdDev(values []float64) float64 {
	if len(values) <= 1 {
		return 0.0
	}

	mean := calculateMean(values)
	sumSquares := 0.0
	for _, v := range values {
		diff := v - mean
		sumSquares += diff * diff
	}
	variance := sumSquares / float64(len(values)-1)
	return math.Sqrt(variance)
}

func writeDropoutResultsToFile(config DropoutExperimentConfig, results DropoutExperimentResult) error {
	resultsDir := filepath.Join("results", "dropout", config.BaseConfig.DataSet)
	if err := os.MkdirAll(resultsDir, 0755); err != nil {
		return fmt.Errorf("failed to create results directory: %v", err)
	}

	filename := fmt.Sprintf("dropout_%s_%s_%s_n%d_f%.1f_%s.txt",
		config.BaseConfig.DataSet,
		config.BaseConfig.Split,
		config.BaseConfig.Imbalance,
		config.BaseConfig.PartyCount,
		config.DropoutRate,
		time.Now().Format("20060102_150405"))

	resultsFile := filepath.Join(resultsDir, filename)
	output := fmt.Sprintf(`
=== Dropout Experiment Results ===
Dataset: %s
Party Count: %d
Dropout Rate: %.1f
Split: %s
Imbalance: %s
Number of Trials: %d

Accuracy Results:
  Soft Voting:       %.2f%% ± %.2f%%
  Logit Soft Voting: %.2f%% ± %.2f%%

Individual Trial Results:
Soft Voting: %v
Logit Soft Voting: %v
`,
		config.BaseConfig.DataSet,
		config.BaseConfig.PartyCount,
		config.DropoutRate,
		config.BaseConfig.Split,
		config.BaseConfig.Imbalance,
		config.NumTrials,
		results.SoftVoting.Mean,
		results.SoftVoting.StdDev,
		results.LogitSoftVoting.Mean,
		results.LogitSoftVoting.StdDev,
		results.SoftVoting.Results,
		results.LogitSoftVoting.Results)

	if err := os.WriteFile(resultsFile, []byte(output), 0644); err != nil {
		return fmt.Errorf("failed to write results: %v", err)
	}

	fmt.Println(output)
	return nil
}

// go test -timeout 0 -v -run TestDropoutExperiments
