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

// go test -timeout 0 -v -run TestInferenceWDBC
// go test -timeout 0 -v -run TestInferenceHeartDisease
// go test -timeout 0 -v -run TestInferencePima
// go test -timeout 0 -v
