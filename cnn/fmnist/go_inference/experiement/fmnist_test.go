package experiment

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func createExperimentConfigs() []ExperimentConfig {
	return []ExperimentConfig{
		{NumDataOwners: 1, NumTestSamples: 1, Distribution: ""},
		// {NumDataOwners: 2, NumTestSamples: 100, Distribution: "balanced"},
		// {NumDataOwners: 2, NumTestSamples: 100, Distribution: "dirichlet0.1"},
		// {NumDataOwners: 2, NumTestSamples: 100, Distribution: "dirichlet0.5"},
		// {NumDataOwners: 5, NumTestSamples: 100, Distribution: "balanced"},
		// {NumDataOwners: 5, NumTestSamples: 100, Distribution: "dirichlet0.1"},
		// {NumDataOwners: 5, NumTestSamples: 100, Distribution: "dirichlet0.5"},
		// {NumDataOwners: 10, NumTestSamples: 100, Distribution: "balanced"},
		// {NumDataOwners: 10, NumTestSamples: 100, Distribution: "dirichlet0.1"},
		// {NumDataOwners: 10, NumTestSamples: 100, Distribution: "dirichlet0.5"},
		// {NumDataOwners: 20, NumTestSamples: 100, Distribution: "balanced"},
		// {NumDataOwners: 20, NumTestSamples: 100, Distribution: "dirichlet0.1"},
		// {NumDataOwners: 20, NumTestSamples: 100, Distribution: "dirichlet0.5"},
	}
}

func createDropoutConfigs() []DropoutExperimentConfig {
	return []DropoutExperimentConfig{
		// {
		// 	BaseConfig:  ExperimentConfig{NumDataOwners: 10, NumTestSamples: 100, Distribution: "balanced"},
		// 	DropoutRate: 0.1,
		// 	NumTrials:   1,
		// },
		// {
		// 	BaseConfig:  ExperimentConfig{NumDataOwners: 10, NumTestSamples: 100, Distribution: "balanced"},
		// 	DropoutRate: 0.3,
		// 	NumTrials:   1,
		// },
		// {
		// 	BaseConfig:  ExperimentConfig{NumDataOwners: 10, NumTestSamples: 100, Distribution: "dirichlet0.1"},
		// 	DropoutRate: 0.1,
		// 	NumTrials:   1,
		// },
		// {
		// 	BaseConfig:  ExperimentConfig{NumDataOwners: 10, NumTestSamples: 100, Distribution: "dirichlet0.1"},
		// 	DropoutRate: 0.3,
		// 	NumTrials:   1,
		// },
		// {
		// 	BaseConfig:  ExperimentConfig{NumDataOwners: 10, NumTestSamples: 100, Distribution: "dirichlet0.5"},
		// 	DropoutRate: 0.1,
		// 	NumTrials:   1,
		// },
		// {
		// 	BaseConfig:  ExperimentConfig{NumDataOwners: 10, NumTestSamples: 100, Distribution: "dirichlet0.5"},
		// 	DropoutRate: 0.3,
		// 	NumTrials:   1,
		// },
		// {
		// 	BaseConfig:  ExperimentConfig{NumDataOwners: 20, NumTestSamples: 100, Distribution: "balanced"},
		// 	DropoutRate: 0.1,
		// 	NumTrials:   1,
		// },
		// {
		// 	BaseConfig:  ExperimentConfig{NumDataOwners: 20, NumTestSamples: 100, Distribution: "balanced"},
		// 	DropoutRate: 0.3,
		// 	NumTrials:   1,
		// },
		// {
		// 	BaseConfig:  ExperimentConfig{NumDataOwners: 20, NumTestSamples: 100, Distribution: "dirichlet0.1"},
		// 	DropoutRate: 0.1,
		// 	NumTrials:   1,
		// },
		// {
		// 	BaseConfig:  ExperimentConfig{NumDataOwners: 20, NumTestSamples: 100, Distribution: "dirichlet0.1"},
		// 	DropoutRate: 0.3,
		// 	NumTrials:   1,
		// },
		// {
		// 	BaseConfig:  ExperimentConfig{NumDataOwners: 20, NumTestSamples: 100, Distribution: "dirichlet0.5"},
		// 	DropoutRate: 0.1,
		// 	NumTrials:   1,
		// },
		// {
		// 	BaseConfig:  ExperimentConfig{NumDataOwners: 20, NumTestSamples: 100, Distribution: "dirichlet0.5"},
		// 	DropoutRate: 0.3,
		// 	NumTrials:   1,
		// },
	}
}

func TestFMNISTEnsemble(t *testing.T) {
	configs := createExperimentConfigs()

	for _, config := range configs {
		testName := fmt.Sprintf("n%d", config.NumDataOwners)
		if config.Distribution != "" {
			testName = fmt.Sprintf("%s_%s", testName, config.Distribution)
		}

		t.Run(testName, func(t *testing.T) {
			if err := checkDataDirectory(t, config); err != nil {
				t.Fatalf("Data directory check failed: %v", err)
			}

			results, err := RunExperiment(config)
			if err != nil {
				t.Fatalf("Experiment failed: %v", err)
			}

			t.Logf("\n=== Results for %s ===", testName)
			t.Logf("Accuracy: %.2f%%", results.Accuracy)
			t.Logf("Test Samples: %d", config.NumTestSamples)
			if config.Distribution != "" {
				t.Logf("Distribution: %s", config.Distribution)
			}
		})
	}
}

func TestFMNISTDropout(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	configs := createDropoutConfigs()

	for _, config := range configs {
		testName := fmt.Sprintf("n%d_f%.1f_%s",
			config.BaseConfig.NumDataOwners,
			config.DropoutRate,
			config.BaseConfig.Distribution)

		t.Run(testName, func(t *testing.T) {
			if err := checkDataDirectory(t, config.BaseConfig); err != nil {
				t.Fatalf("Data directory check failed: %v", err)
			}

			results, err := RunDropoutExperiment(config)
			if err != nil {
				t.Fatalf("Dropout experiment failed: %v", err)
			}

			t.Logf("\n=== Results for %s ===", testName)
			t.Logf("Mean Accuracy: %.2f%%", results.Mean)
			t.Logf("StdDev Accuracy: %.2f%%", results.StdDev)
			t.Logf("Number of Trials: %d", config.NumTrials)
			t.Logf("Dropout Rate: %.1f", config.DropoutRate)
			t.Logf("Distribution: %s", config.BaseConfig.Distribution)
		})
	}
}

func checkDataDirectory(t *testing.T, config ExperimentConfig) error {
	testDataPath := filepath.Join("../data", "fmnist_test.csv")
	if _, err := os.Stat(testDataPath); os.IsNotExist(err) {
		return fmt.Errorf("test data file not found: %s", testDataPath)
	}

	for i := 0; i < config.NumDataOwners; i++ {
		modelPath := getModelPath(config, i)
		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			return fmt.Errorf("model parameters file not found: %s", modelPath)
		}
	}
	return nil
}

// go test -v -timeout 0
// go test -v -timeout 0 -run TestFMNISTEnsemble
// go test -v -timeout 0 -run TestFMNISTDropout
