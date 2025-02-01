package mnist_ensemble

import (
	"fmt"
	"testing"
	"time"
)

// we should test one by one
func TestMNISTEnsemble(t *testing.T) {
	configs := []ExperimentConfig{
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

	for i, config := range configs {
		testName := fmt.Sprintf("n%d", config.NumDataOwners)
		if config.Distribution != "" {
			testName = fmt.Sprintf("%s_%s", testName, config.Distribution)
		}

		t.Run(testName, func(t *testing.T) {
			t.Logf("\n\n=== Starting Experiment %d/%d ===", i+1, len(configs))
			t.Logf("Configuration:")
			t.Logf("- Number of Data Owners: %d", config.NumDataOwners)
			t.Logf("- Number of Test Samples: %d", config.NumTestSamples)
			t.Logf("- Distribution: %s", config.Distribution)

			results, err := RunExperiment(config)
			if err != nil {
				t.Fatalf("Experiment failed: %v", err)
			}

			t.Logf("\n=== Results ===")
			t.Logf("Accuracy: %.2f%%", results.Accuracy)

			t.Logf("\n=== Client Times ===")
			t.Logf("- Key Generation: %.2f ± %.2f ms",
				float64(results.Timing.ClientKeyGenStats.Mean)/float64(time.Millisecond),
				float64(results.Timing.ClientKeyGenStats.StdDev)/float64(time.Millisecond))
			t.Logf("- Data Encryption: %.2f ± %.2f ms",
				float64(results.Timing.DataEncryptionStats.Mean)/float64(time.Millisecond),
				float64(results.Timing.DataEncryptionStats.StdDev)/float64(time.Millisecond))
			t.Logf("- Final Decryption: %.2f ± %.2f ms",
				float64(results.Timing.FinalDecryptionStats.Mean)/float64(time.Millisecond),
				float64(results.Timing.FinalDecryptionStats.StdDev)/float64(time.Millisecond))
			t.Logf("- Total Decryption: %.2f ± %.2f ms",
				float64(results.Timing.TotalDecryptionStats.Mean)/float64(time.Millisecond),
				float64(results.Timing.TotalDecryptionStats.StdDev)/float64(time.Millisecond))

			t.Logf("\n=== Data Owner Statistics ===")
			t.Logf("- Key Generation: %.2f ± %.2f ms",
				float64(results.Timing.DataOwnerKeyGenStats.Mean)/float64(time.Millisecond),
				float64(results.Timing.DataOwnerKeyGenStats.StdDev)/float64(time.Millisecond))
			t.Logf("- Model Encryption: %.2f ± %.2f ms",
				float64(results.Timing.ModelEncryptionStats.Mean)/float64(time.Millisecond),
				float64(results.Timing.ModelEncryptionStats.StdDev)/float64(time.Millisecond))
			t.Logf("- Partial Decryption: %.9f ms",
				float64(results.Timing.PartialDecryptionTime)/float64(time.Millisecond))

			t.Logf("\n=== CSP Times ===")
			t.Logf("- Inference Time: %.2f ± %.2f ms",
				float64(results.Timing.InferenceStats.Mean)/float64(time.Millisecond),
				float64(results.Timing.InferenceStats.StdDev)/float64(time.Millisecond))
			t.Logf("- Ensemble Time: %.2f ± %.2f ms",
				float64(results.Timing.EnsembleStats.Mean)/float64(time.Millisecond),
				float64(results.Timing.EnsembleStats.StdDev)/float64(time.Millisecond))
			t.Logf("- Total Compute Time: %.2f ± %.2f ms",
				float64(results.Timing.TotalComputeStats.Mean)/float64(time.Millisecond),
				float64(results.Timing.TotalComputeStats.StdDev)/float64(time.Millisecond))
			t.Logf("- Client Transfer Latency: %.2f ± %.2f ms",
				float64(results.Timing.ClientTransferStats.Mean)/float64(time.Millisecond),
				float64(results.Timing.ClientTransferStats.StdDev)/float64(time.Millisecond))
			t.Logf("- Data Owner Transfer Latency: %.2f ms",
				float64(results.Timing.DataOwnerTransferTime)/float64(time.Millisecond))
		})
	}
}

// go test -v -timeout 0 -run TestMNISTEnsemble
// go test -v -timeout 0 -run TestMNISTDropout
// go test -v -timeout 0
