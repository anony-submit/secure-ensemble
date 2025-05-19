package experiments

import "testing"

func TestInferenceWDBC(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	RunDatasetExperiments(t, "wdbc")
}

func TestInferenceHeartDisease(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	RunDatasetExperiments(t, "heart_disease")
}

func TestInferencePima(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	RunDatasetExperiments(t, "pima")
}
