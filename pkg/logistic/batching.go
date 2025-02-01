package logistic

import (
	"encoding/csv"
	"encoding/json"
	"os"
	"strconv"
)

type Model struct {
	Weights   [][]float64 `json:"weights"`
	Intercept []float64   `json:"intercept"`
}

type BatchConfig struct {
	FeatureDim         int
	SampleCount        int
	FeaturePad         int
	SamplePad          int
	MaxVerticalParties int
}

func CreateBatchedMatrix(data [][]float64, config BatchConfig) []complex128 {
	batched := make([]complex128, config.FeaturePad*config.SamplePad)
	for col := 0; col < len(data[0]); col++ {
		for row := 0; row < len(data); row++ {
			batched[row*config.SamplePad+col] = complex(data[row][col], 0)
		}
	}
	return batched
}

func LoadModelFromJSON(filepath string) (*Model, error) {
	modelFile, err := os.ReadFile(filepath)
	if err != nil {
		return nil, err
	}

	var model Model
	if err := json.Unmarshal(modelFile, &model); err != nil {
		return nil, err
	}
	return &model, nil
}

func CreateWeightMatrix(weights []float64, numSamples int) [][]float64 {
	matrix := make([][]float64, len(weights))
	for i := range matrix {
		matrix[i] = make([]float64, numSamples)
		for j := 0; j < numSamples; j++ {
			matrix[i][j] = weights[i]
		}
	}
	return matrix
}

func LoadTestData(filepath string, numFeatures, maxSamples int) ([][]float64, []int, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	testData := make([][]float64, numFeatures)
	for i := range testData {
		testData[i] = make([]float64, maxSamples)
	}
	trueLabels := make([]int, maxSamples)

	reader := csv.NewReader(file)
	colIdx := 0
	for colIdx < maxSamples {
		record, err := reader.Read()
		if err != nil {
			break
		}

		for i := 0; i < numFeatures; i++ {
			val, _ := strconv.ParseFloat(record[i], 64)
			testData[i][colIdx] = val
		}

		label, _ := strconv.ParseFloat(record[numFeatures], 64)
		if label > 0 {
			trueLabels[colIdx] = 1
		}
		colIdx++
	}

	return testData, trueLabels, nil
}
