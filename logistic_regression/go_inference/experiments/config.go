package experiments

import (
	"secure-ensemble/pkg/logistic"
)

type ExperimentConfig struct {
	DataSet     string
	PartyCount  int
	Split       string
	Imbalance   string
	BatchConfig logistic.BatchConfig
	CSPAddress  string
}

var datasetConfigs = map[string]logistic.BatchConfig{
	"wdbc": {
		FeatureDim:         30,
		SampleCount:        114,
		FeaturePad:         32,
		SamplePad:          128,
		MaxVerticalParties: 20,
	},
	"heart_disease": {
		FeatureDim:         13,
		SampleCount:        61,
		FeaturePad:         16,
		SamplePad:          64,
		MaxVerticalParties: 10,
	},
	"pima": {
		FeatureDim:         8,
		SampleCount:        154,
		FeaturePad:         8,
		SamplePad:          256,
		MaxVerticalParties: 5,
	},
}

func GetValidPartyCounts(dataset, split string) []int {
	allPartyCounts := []int{2, 5, 10, 20}
	if split != "vertical" {
		return allPartyCounts
	}

	config := datasetConfigs[dataset]
	validCounts := make([]int, 0)
	for _, count := range allPartyCounts {
		if count <= config.MaxVerticalParties {
			validCounts = append(validCounts, count)
		}
	}
	return validCounts
}

// //////////////////////////////// Drop-out Experiments /////////////////////////////////
type DropoutExperimentConfig struct {
	BaseConfig  ExperimentConfig
	DropoutRate float64
	NumTrials   int
}

type DropoutExperimentResult struct {
	SoftVoting      DropoutAccuracyInfo
	LogitSoftVoting DropoutAccuracyInfo
}

type DropoutAccuracyInfo struct {
	Mean    float64
	StdDev  float64
	Results []float64
}
