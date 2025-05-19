package logistic_ensemble

import (
	"encoding/json"
	"fmt"
	"math/bits"
	"os"
	"path/filepath"
	"testing"
	"time"

	"secure-ensemble/pkg/activation"
	"secure-ensemble/pkg/logistic"

	"github.com/anony-submit/snu-mghe/mkckks"
	"github.com/anony-submit/snu-mghe/mkrlwe"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
)

type ModelParams struct {
	Weights   []float64 `json:"weights"`
	Intercept []float64 `json:"intercept"`
}

type EncryptedModel struct {
	EncWeights   *mkckks.Ciphertext
	EncIntercept *mkckks.Ciphertext
}

type ExperimentConfig struct {
	DataSet     string
	PartyCount  int
	Split       string
	Imbalance   string
	BatchConfig logistic.BatchConfig
}

type TimingInfo struct {
	DataOwnerSetupTotal time.Duration // Data Owner Keygen
	DataOwnerSetupAvg   time.Duration
	ModelEncryption     time.Duration // Data Owner Model Encryption
	ClientSetup         time.Duration // Client Keygen
	DataEncryption      time.Duration // Client Data Encryption
	Inference           time.Duration
	Ensemble            time.Duration
	TotalEvaluation     time.Duration
	Decryption          time.Duration
}

type AccuracyInfo struct {
	Percentage float64
	Correct    int
	Total      int
}

type ExperimentResult struct {
	Timing   TimingInfo
	Accuracy AccuracyInfo
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

func getValidPartyCounts(dataset, split string) []int {
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

type ScalingConfig struct {
	SoftVoting      float64
	LogitSoftVoting float64
}

var datasetScaling = map[string]ScalingConfig{
	"wdbc": {
		SoftVoting:      25.0,
		LogitSoftVoting: 50.0,
	},
	"heart_disease": {
		SoftVoting:      5.0,
		LogitSoftVoting: 20.0,
	},
	"pima": {
		SoftVoting:      5.0,
		LogitSoftVoting: 20.0,
	},
}

func getRotations(config logistic.BatchConfig) []int {
	rotations := []int{}
	for i := 0; (1 << i) < config.FeaturePad; i++ {
		rotations = append(rotations, (1<<i)*config.SamplePad)
	}
	return rotations
}

func getLogSlots(config ExperimentConfig) int {
	totalSlots := config.BatchConfig.FeaturePad * config.BatchConfig.SamplePad
	return bits.Len(uint(totalSlots)) - 1
}

func setupCrypto(config ExperimentConfig) (mkckks.Parameters, *mkrlwe.SecretKeySet, *mkrlwe.PublicKeySet,
	*mkrlwe.RelinearizationKeySet, *mkrlwe.RotationKeySet, TimingInfo) {

	var timing TimingInfo

	logSlots := getLogSlots(config)
	params, _ := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:     15,
		LogSlots: logSlots,
		Q: []uint64{0x4000000120001, 0x10000140001, 0xffffe80001,
			0x10000290001, 0xffffc40001, 0x100003e0001,
			0x10000470001, 0x100004b0001, 0xffffb20001},
		P:     []uint64{0x40000001b0001, 0x3ffffffdf0001, 0x4000000270001},
		Scale: 1 << 40,
		Sigma: rlwe.DefaultSigma,
	})

	// 10개 남음
	// params, _ := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	// 	LogN:     15,
	// 	LogSlots: logSlots,
	// 	Q: []uint64{0x4000000120001, 0x10000140001, 0xffffe80001,
	// 		0x10000290001, 0xffffc40001, 0x100003e0001,
	// 		0x10000470001, 0x100004b0001, 0xffffb20001,
	// 		0x10000500001, 0x10000650001, 0xffff940001,
	// 		0xffff8a0001, 0xffff820001, 0xffff780001,
	// 		0x10000890001, 0xffff750001, 0x10000960001},
	// 	P:            []uint64{0x40000001b0001, 0x3ffffffdf0001, 0x4000000270001},
	// 	Scale: 1 << 40,
	// 	Sigma:        rlwe.DefaultSigma,
	// })

	mkParams := mkckks.NewParameters(params)
	rotations := getRotations(config.BatchConfig)
	for _, rot := range rotations {
		mkParams.AddCRS(rot)
	}

	kgen := mkckks.NewKeyGenerator(mkParams)
	skSet := mkrlwe.NewSecretKeySet()
	pkSet := mkrlwe.NewPublicKeyKeySet()
	rlkSet := mkrlwe.NewRelinearizationKeySet(mkParams.Parameters)
	rtkSet := mkrlwe.NewRotationKeySet()

	dataOwnerStart := time.Now()
	for i := 0; i < config.PartyCount; i++ {
		partyID := fmt.Sprintf("owner%d", i)
		sk, pk := kgen.GenKeyPair(partyID)
		skSet.AddSecretKey(sk)
		pkSet.AddPublicKey(pk)
		rlk := kgen.GenRelinearizationKey(sk)
		rlkSet.AddRelinearizationKey(rlk)
		for _, rot := range rotations {
			rtk := kgen.GenRotationKey(rot, sk)
			rtkSet.AddRotationKey(rtk)
		}
	}
	timing.DataOwnerSetupTotal = time.Since(dataOwnerStart)
	timing.DataOwnerSetupAvg = timing.DataOwnerSetupTotal / time.Duration(config.PartyCount)

	clientStart := time.Now()
	clientID := "client"
	sk, pk := kgen.GenKeyPair(clientID)
	skSet.AddSecretKey(sk)
	pkSet.AddPublicKey(pk)
	rlk := kgen.GenRelinearizationKey(sk)
	rlkSet.AddRelinearizationKey(rlk)
	for _, rot := range rotations {
		rtk := kgen.GenRotationKey(rot, sk)
		rtkSet.AddRotationKey(rtk)
	}
	timing.ClientSetup = time.Since(clientStart)

	return mkParams, skSet, pkSet, rlkSet, rtkSet, timing
}

func encryptModelParams(model ModelParams, config ExperimentConfig,
	mkParams mkckks.Parameters, encryptor *mkckks.Encryptor, ownerID string,
	pkSet *mkrlwe.PublicKeySet) (*EncryptedModel, error) {

	weightMatrix := logistic.CreateWeightMatrix(model.Weights, config.BatchConfig.SampleCount)
	weightsBatched := logistic.CreateBatchedMatrix(weightMatrix, config.BatchConfig)

	interceptMatrix := make([][]float64, config.BatchConfig.FeatureDim)
	for k := range interceptMatrix {
		interceptMatrix[k] = make([]float64, config.BatchConfig.SampleCount)
		for l := 0; l < config.BatchConfig.SampleCount; l++ {
			interceptMatrix[k][l] = model.Intercept[0]
		}
	}
	interceptBatched := logistic.CreateBatchedMatrix(interceptMatrix, config.BatchConfig)

	weightMsg := mkckks.NewMessage(mkParams)
	copy(weightMsg.Value, weightsBatched)
	encWeights := encryptor.EncryptMsgNew(weightMsg, pkSet.GetPublicKey(ownerID))

	interceptMsg := mkckks.NewMessage(mkParams)
	copy(interceptMsg.Value, interceptBatched)
	encIntercept := encryptor.EncryptMsgNew(interceptMsg, pkSet.GetPublicKey(ownerID))

	return &EncryptedModel{
		EncWeights:   encWeights,
		EncIntercept: encIntercept,
	}, nil
}

func performInference(encModel *EncryptedModel, encTestData *mkckks.Ciphertext,
	evaluator *mkckks.Evaluator, rlkSet *mkrlwe.RelinearizationKeySet,
	rtkSet *mkrlwe.RotationKeySet, config ExperimentConfig) (*mkckks.Ciphertext, error) {

	result := evaluator.MulRelinNew(encModel.EncWeights, encTestData, rlkSet)

	for j := bits.Len(uint(config.BatchConfig.FeaturePad)) - 2; j >= 0; j-- {
		rotated := evaluator.RotateNew(result, (1<<j)*config.BatchConfig.SamplePad, rtkSet)
		result = evaluator.AddNew(result, rotated)
	}

	return evaluator.AddNew(result, encModel.EncIntercept), nil
}

func applyLogistic(result *mkckks.Ciphertext, evaluator *mkckks.Evaluator,
	encryptor *mkckks.Encryptor, pkSet *mkrlwe.PublicKeySet,
	rlkSet *mkrlwe.RelinearizationKeySet, mkParams mkckks.Parameters,
	scalingFactor float64) (*mkckks.Ciphertext, error) {

	evaluator.MultByConst(result, complex(1.0/scalingFactor, 0), result)
	if err := evaluator.Rescale(result, mkParams.Scale(), result); err != nil {
		return nil, fmt.Errorf("rescale failed after scaling down: %w", err)
	}

	constMsg := mkckks.NewMessage(mkParams)
	for i := 0; i < constMsg.Slots(); i++ {
		constMsg.Value[i] = complex(1.0, 0)
	}
	constOne := encryptor.EncryptMsgNew(constMsg, pkSet.GetPublicKey("client"))

	return activation.EvalLogistic(result, constOne, mkParams, evaluator, rlkSet)
}

func performSoftVotingInference(encModel *EncryptedModel, encTestData *mkckks.Ciphertext,
	evaluator *mkckks.Evaluator, encryptor *mkckks.Encryptor,
	pkSet *mkrlwe.PublicKeySet, rlkSet *mkrlwe.RelinearizationKeySet,
	rtkSet *mkrlwe.RotationKeySet, mkParams mkckks.Parameters,
	config ExperimentConfig) (*mkckks.Ciphertext, error) {

	// Linear inference
	result, err := performInference(encModel, encTestData, evaluator, rlkSet, rtkSet, config)
	if err != nil {
		return nil, err
	}

	// Apply scaling and logistic in inference phase for soft voting
	scaling := datasetScaling[config.DataSet].SoftVoting
	return applyLogistic(result, evaluator, encryptor, pkSet, rlkSet, mkParams, scaling)
}

func calculateAccuracy(decResult *mkckks.Message, trueLabels []int, sampleCount int) AccuracyInfo {
	correct := 0
	for i := 0; i < sampleCount; i++ {
		val := real(decResult.Value[i])
		prediction := 0
		if val > 0.5 {
			prediction = 1
		}
		if (prediction == 1 && trueLabels[i] == 1) || (prediction == 0 && trueLabels[i] == 0) {
			correct++
		}
	}

	return AccuracyInfo{
		Percentage: float64(correct) * 100.0 / float64(sampleCount),
		Correct:    correct,
		Total:      sampleCount,
	}
}

func writeResultsToFile(config ExperimentConfig, results map[string]ExperimentResult) error {
	// Format results first
	output := formatResults(config, results)

	// Print to console immediately
	fmt.Println(output)

	// Create unique filename using configuration details
	filename := fmt.Sprintf("results_%s_%s_%s_n%d.txt",
		config.DataSet,
		config.Split,
		config.Imbalance,
		config.PartyCount)

	// Create directory structure
	resultsDir := filepath.Join("results", config.DataSet)
	if err := os.MkdirAll(resultsDir, 0755); err != nil {
		return fmt.Errorf("failed to create results directory: %v", err)
	}

	resultsFile := filepath.Join(resultsDir, filename)

	// Check if file exists
	if _, err := os.Stat(resultsFile); err == nil {
		// File exists - backup the old file
		backupFile := resultsFile + ".bak"
		if err := os.Rename(resultsFile, backupFile); err != nil {
			return fmt.Errorf("failed to backup existing results file: %v", err)
		}
	}

	// Write new results (create new file)
	if err := os.WriteFile(resultsFile, []byte(output), 0644); err != nil {
		return fmt.Errorf("failed to write results: %v", err)
	}

	return nil
}

func formatResults(config ExperimentConfig, results map[string]ExperimentResult) string {
	return fmt.Sprintf(`
=== Experiment Results ===
Dataset: %s
Party Count: %d
Split: %s
Imbalance: %s

Data Owner Setup Time (Total): %v
Data Owner Setup Time (Avg): %v
Model Encryption Time: %v
Client Setup Time: %v
Data Encryption Time: %v

Soft Voting:
Accuracy: %.2f%% (%d/%d)
Inference Time: %v
Ensemble Time: %v
Total Time: %v
Decryption Time: %v

Logit Soft Voting:
Accuracy: %.2f%% (%d/%d)
Inference Time: %v
Ensemble Time: %v
Total Time: %v
Decryption Time: %v

`,
		config.DataSet,
		config.PartyCount,
		config.Split,
		config.Imbalance,

		results["soft_voting"].Timing.DataOwnerSetupTotal,
		results["soft_voting"].Timing.DataOwnerSetupAvg,
		results["soft_voting"].Timing.ModelEncryption,
		results["soft_voting"].Timing.ClientSetup,
		results["soft_voting"].Timing.DataEncryption,

		results["soft_voting"].Accuracy.Percentage,
		results["soft_voting"].Accuracy.Correct,
		results["soft_voting"].Accuracy.Total,
		results["soft_voting"].Timing.Inference,
		results["soft_voting"].Timing.Ensemble,
		results["soft_voting"].Timing.TotalEvaluation,
		results["soft_voting"].Timing.Decryption,

		results["logit_soft_voting"].Accuracy.Percentage,
		results["logit_soft_voting"].Accuracy.Correct,
		results["logit_soft_voting"].Accuracy.Total,
		results["logit_soft_voting"].Timing.Inference,
		results["logit_soft_voting"].Timing.Ensemble,
		results["logit_soft_voting"].Timing.TotalEvaluation,
		results["logit_soft_voting"].Timing.Decryption)
}

func runLogisticInference(t *testing.T, config ExperimentConfig) {
	mkParams, skSet, pkSet, rlkSet, rtkSet, setupTiming := setupCrypto(config)
	encryptor := mkckks.NewEncryptor(mkParams)
	decryptor := mkckks.NewDecryptor(mkParams)
	evaluator := mkckks.NewEvaluator(mkParams)

	/// Client Side Setup & Encryption
	clientEncStart := time.Now()
	testDataPath := filepath.Join("data", config.DataSet,
		fmt.Sprintf("%s_test.csv", config.DataSet))
	testData, trueLabels, err := logistic.LoadTestData(testDataPath,
		config.BatchConfig.FeatureDim,
		config.BatchConfig.SampleCount)
	if err != nil {
		t.Fatal(err)
	}

	testDataBatched := logistic.CreateBatchedMatrix(testData, config.BatchConfig)
	testDataMsg := mkckks.NewMessage(mkParams)
	copy(testDataMsg.Value, testDataBatched)
	encTestData := encryptor.EncryptMsgNew(testDataMsg, pkSet.GetPublicKey("client"))
	dataEncTime := time.Since(clientEncStart)

	/// Model Owner Side : Model Encryption
	modelOwnerStart := time.Now()
	modelPath := filepath.Join("data", config.DataSet, config.Imbalance, config.Split,
		fmt.Sprintf("%s_%s_n%d_models.json", config.DataSet, config.Split, config.PartyCount))
	modelFile, err := os.ReadFile(modelPath)
	if err != nil {
		t.Logf("Skipping test for %s: %v", modelPath, err)
		return
	}

	var models []ModelParams
	if err := json.Unmarshal(modelFile, &models); err != nil {
		t.Fatal(err)
	}

	encryptedModels := make([]*EncryptedModel, config.PartyCount)
	for i := 0; i < config.PartyCount; i++ {
		ownerID := fmt.Sprintf("owner%d", i)
		encryptedModel, err := encryptModelParams(models[i], config, mkParams,
			encryptor, ownerID, pkSet)
		if err != nil {
			t.Fatal(err)
		}
		encryptedModels[i] = encryptedModel
	}
	modelEncTime := time.Since(modelOwnerStart)

	ensembleResults := make(map[string]ExperimentResult)

	/// Soft Voting
	{
		// Inference phase (includes logistic)
		inferenceStartTime := time.Now()
		results := make([]*mkckks.Ciphertext, config.PartyCount)
		for i := 0; i < config.PartyCount; i++ {
			result, err := performSoftVotingInference(encryptedModels[i], encTestData,
				evaluator, encryptor, pkSet, rlkSet, rtkSet, mkParams, config)
			if err != nil {
				t.Error(err)
				return
			}
			results[i] = result
		}
		inferenceTime := time.Since(inferenceStartTime)

		// Ensemble phase (just adding)
		ensembleStartTime := time.Now()
		finalResult := results[0]
		for i := 1; i < len(results); i++ {
			finalResult = evaluator.AddNew(finalResult, results[i])
		}
		evaluator.MultByConst(finalResult, complex(1.0/float64(len(results)), 0), finalResult)
		evaluator.Rescale(finalResult, mkParams.Scale(), finalResult)

		ensembleTime := time.Since(ensembleStartTime)
		fmt.Printf("Final Level in Soft Voting: %d\n", finalResult.Level())

		// Decryption
		decStartTime := time.Now()
		decResult := decryptor.Decrypt(finalResult, skSet)
		decTime := time.Since(decStartTime)

		accuracyInfo := calculateAccuracy(decResult, trueLabels, config.BatchConfig.SampleCount)

		ensembleResults["soft_voting"] = ExperimentResult{
			Timing: TimingInfo{
				DataOwnerSetupTotal: setupTiming.DataOwnerSetupTotal,
				DataOwnerSetupAvg:   setupTiming.DataOwnerSetupAvg,
				ClientSetup:         setupTiming.ClientSetup,
				DataEncryption:      dataEncTime,
				ModelEncryption:     modelEncTime,
				Inference:           inferenceTime,
				Ensemble:            ensembleTime,
				TotalEvaluation:     inferenceTime + ensembleTime,
				Decryption:          decTime,
			},
			Accuracy: accuracyInfo,
		}
	}

	/// Logit Soft Voting
	{
		// Inference phase (linear only)
		inferenceStartTime := time.Now()
		results := make([]*mkckks.Ciphertext, config.PartyCount)
		for i := 0; i < config.PartyCount; i++ {
			result, err := performInference(encryptedModels[i], encTestData,
				evaluator, rlkSet, rtkSet, config)
			if err != nil {
				t.Error(err)
				return
			}
			results[i] = result
		}
		inferenceTime := time.Since(inferenceStartTime)

		// Ensemble phase (adding + logistic)
		ensembleStartTime := time.Now()

		sumResult := results[0]
		for i := 1; i < len(results); i++ {
			sumResult = evaluator.AddNew(sumResult, results[i])
		}

		scaling := datasetScaling[config.DataSet].LogitSoftVoting
		finalResult, err := applyLogistic(sumResult, evaluator, encryptor, pkSet, rlkSet, mkParams, scaling)
		fmt.Printf("Final Level in Logit Soft Voting: %d\n", finalResult.Level())
		if err != nil {
			t.Error(err)
			return
		}
		ensembleTime := time.Since(ensembleStartTime)

		// Decryption
		decStartTime := time.Now()
		decResult := decryptor.Decrypt(finalResult, skSet)
		decTime := time.Since(decStartTime)

		accuracyInfo := calculateAccuracy(decResult, trueLabels, config.BatchConfig.SampleCount)

		ensembleResults["logit_soft_voting"] = ExperimentResult{
			Timing: TimingInfo{
				DataOwnerSetupTotal: setupTiming.DataOwnerSetupTotal,
				DataOwnerSetupAvg:   setupTiming.DataOwnerSetupAvg,
				ClientSetup:         setupTiming.ClientSetup,
				DataEncryption:      dataEncTime,
				ModelEncryption:     modelEncTime,
				Inference:           inferenceTime,
				Ensemble:            ensembleTime,
				TotalEvaluation:     inferenceTime + ensembleTime,
				Decryption:          decTime,
			},
			Accuracy: accuracyInfo,
		}
	}
	// fmt.Printf("\nDataset: %s, Split: %s, Imbalance: %s, Party Count: %d\n",
	// 	config.DataSet, config.Split, config.Imbalance, config.PartyCount)

	// fmt.Printf("\nSoft Voting Accuracy: %.2f%% (%d/%d)\n",
	// 	ensembleResults["soft_voting"].Accuracy.Percentage,
	// 	ensembleResults["soft_voting"].Accuracy.Correct,
	// 	ensembleResults["soft_voting"].Accuracy.Total)

	// fmt.Printf("Soft Voting Inference Time: %v\n",
	// 	ensembleResults["soft_voting"].Timing.Inference)

	// fmt.Printf("\nLogit Soft Voting Accuracy: %.2f%% (%d/%d)\n",
	// 	ensembleResults["logit_soft_voting"].Accuracy.Percentage,
	// 	ensembleResults["logit_soft_voting"].Accuracy.Correct,
	// 	ensembleResults["logit_soft_voting"].Accuracy.Total)

	// fmt.Printf("Logit Soft Voting Inference Time: %v\n\n",
	// 	ensembleResults["logit_soft_voting"].Timing.Inference)

	if err := writeResultsToFile(config, ensembleResults); err != nil {
		t.Error(err)
	}
}

func runIdealInference(t *testing.T, dataset string) {
	config := ExperimentConfig{
		DataSet:     dataset,
		PartyCount:  1,
		Split:       "ideal",
		Imbalance:   "ideal",
		BatchConfig: datasetConfigs[dataset],
	}
	runLogisticInference(t, config)
}

func runDatasetInference(t *testing.T, dataset string) {
	splits := []string{"horizontal", "vertical"}
	imbalances := []string{"balanced", "dirichlet_0.1", "dirichlet_0.5"}

	// Run ideal case first
	fmt.Printf("\n=== Starting Ideal Case for %s dataset ===\n", dataset)
	t.Run("Ideal", func(t *testing.T) {
		runIdealInference(t, dataset)
	})

	// Run main experiments
	fmt.Printf("\n=== Starting Main Experiments for %s dataset ===\n", dataset)
	for _, split := range splits {
		partyCounts := getValidPartyCounts(dataset, split)

		for _, imbalance := range imbalances {
			fmt.Printf("\n--- Starting experiments for %s split, %s imbalance ---\n",
				split, imbalance)

			for _, n := range partyCounts {
				fmt.Printf("\nRunning experiment with %d parties...\n", n)
				testName := fmt.Sprintf("%s_%s_N%d", split, imbalance, n)
				t.Run(testName, func(t *testing.T) {
					config := ExperimentConfig{
						DataSet:     dataset,
						PartyCount:  n,
						Split:       split,
						Imbalance:   imbalance,
						BatchConfig: datasetConfigs[dataset],
					}
					runLogisticInference(t, config)
				})
			}
		}
	}
}

// Then the individual test functions become much simpler:
func TestInferenceWDBC(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	runDatasetInference(t, "wdbc")
}

func TestInferenceHeartDisease(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	runDatasetInference(t, "heart_disease")
}

func TestInferencePima(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	runDatasetInference(t, "pima")
}

// go test -v -timeout 1h -run TestInferenceWDBC
// go test -timeout 1h -run TestInferenceHeartDisease
// go test -timeout 1h -run TestInferencePima
// go test -timeout 0
