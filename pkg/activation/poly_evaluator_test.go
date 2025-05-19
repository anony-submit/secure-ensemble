package activation

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/anony-submit/snu-mghe/mkckks"
	"github.com/anony-submit/snu-mghe/mkrlwe"
	"github.com/ldsec/lattigo/v2/ckks"
)

func evaluatePlainPolynomial(x float64, coefs []float64) float64 {
	result := 0.0
	for i, coef := range coefs {
		result += coef * math.Pow(x, float64(i))
	}
	return result
}

func TestPolynomialEvaluation(t *testing.T) {
	params, _ := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:     14,
		LogSlots: 12,
		Q: []uint64{
			0x7fffffffba0001,
			0x3fffffffd60001, 0x3fffffffca0001,
			0x3fffffff6d0001, 0x3fffffff5d0001,
			0x3fffffff550001, 0x3fffffff390001,
		},
		P:     []uint64{0x3ffc0001, 0x3fde0001},
		Scale: 1 << 54,
	})
	mkParams := mkckks.NewParameters(params)
	kgen := mkckks.NewKeyGenerator(mkParams)
	users := []string{"user1", "user2"}
	skSet := mkrlwe.NewSecretKeySet()
	pkSet := mkrlwe.NewPublicKeyKeySet()
	rlkSet := mkrlwe.NewRelinearizationKeySet(mkParams.Parameters)
	for _, user := range users {
		sk, pk := kgen.GenKeyPair(user)
		rlk := kgen.GenRelinearizationKey(sk)
		skSet.AddSecretKey(sk)
		pkSet.AddPublicKey(pk)
		rlkSet.AddRelinearizationKey(rlk)
	}
	encryptor := mkckks.NewEncryptor(mkParams)
	evaluator := mkckks.NewEvaluator(mkParams)
	decryptor := mkckks.NewDecryptor(mkParams)
	constMsg := mkckks.NewMessage(mkParams)
	for i := 0; i < constMsg.Slots(); i++ {
		constMsg.Value[i] = complex(1.0, 0)
	}
	constOne := encryptor.EncryptMsgNew(constMsg, pkSet.GetPublicKey("user1"))

	testCases := []struct {
		name     string
		coefs    []float64
		x        float64
		maxError float64
	}{
		{
			name:     "Simple Linear",
			coefs:    []float64{0.5, 0.5}, // 0.5x + 0.5
			x:        4.0,
			maxError: 1e-5,
		},
		{
			name:     "Simple Quadratic",
			coefs:    []float64{1.0, 2.0, 1.0}, // x^2 + 2x + 1
			x:        2.0,
			maxError: 1e-5,
		},
		{
			name:     "Cubic with Negative Terms",
			coefs:    []float64{1.0, -2.0, 1.0, -1.0}, // -x^3 + x^2 - 2x + 1
			x:        0.5,
			maxError: 1e-5,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			msg := mkckks.NewMessage(mkParams)
			msg.Value[0] = complex(tc.x, 0)
			ct := encryptor.EncryptMsgNew(msg, pkSet.GetPublicKey("user1"))

			fmt.Printf("\nInitial ciphertext level for %s: %d\n", tc.name, ct.Level())

			result, err := EvalPolynomialPS(tc.coefs, ct, constOne, mkParams, evaluator, rlkSet)
			if err != nil {
				t.Fatalf("EvaluatePolynomial failed: %v", err)
			}

			fmt.Printf("Final ciphertext level for %s: %d\n", tc.name, result.Level())

			decResult := decryptor.Decrypt(result, skSet)
			psValue := real(decResult.Value[0])
			plainValue := evaluatePlainPolynomial(tc.x, tc.coefs)
			error := math.Abs(psValue - plainValue)
			if error > tc.maxError {
				t.Errorf("Large error detected for %s:\nPS value: %f\nPlain value: %f\nError: %f\nMax allowed error: %f",
					tc.name, psValue, plainValue, error, tc.maxError)
			}
			t.Logf("%s Results:\nInput x: %f\nPS result: %f\nPlain result: %f\nError: %e",
				tc.name, tc.x, psValue, plainValue, error)
		})
	}
}
func TestLogisticApproximation(t *testing.T) {
	params, _ := ckks.NewParametersFromLiteral(ckks.PN14QP438)
	mkParams := mkckks.NewParameters(params)

	kgen := mkckks.NewKeyGenerator(mkParams)
	users := []string{"user1", "user2"}
	skSet := mkrlwe.NewSecretKeySet()
	pkSet := mkrlwe.NewPublicKeyKeySet()
	rlkSet := mkrlwe.NewRelinearizationKeySet(mkParams.Parameters)

	for _, user := range users {
		sk, pk := kgen.GenKeyPair(user)
		rlk := kgen.GenRelinearizationKey(sk)
		skSet.AddSecretKey(sk)
		pkSet.AddPublicKey(pk)
		rlkSet.AddRelinearizationKey(rlk)
	}

	encryptor := mkckks.NewEncryptor(mkParams)
	evaluator := mkckks.NewEvaluator(mkParams)
	decryptor := mkckks.NewDecryptor(mkParams)

	// Need to deliver an encryption of 1 to evaluate constant term
	constMsg := mkckks.NewMessage(mkParams)
	for i := 0; i < constMsg.Slots(); i++ {
		constMsg.Value[i] = complex(1.0, 0)
	}
	constOne := encryptor.EncryptMsgNew(constMsg, pkSet.GetPublicKey("user1"))

	msg1 := mkckks.NewMessage(mkParams)
	msg2 := mkckks.NewMessage(mkParams)

	realValues1 := make([]float64, 8)
	realValues2 := make([]float64, 8)

	for i := 0; i < 8; i++ {
		val1 := -5.0 + rand.Float64()*10.0
		val2 := -5.0 + rand.Float64()*10.0
		realValues1[i] = val1
		realValues2[i] = val2
		msg1.Value[i] = complex(val1, 0)
		msg2.Value[i] = complex(val2, 0)
	}

	ct1 := encryptor.EncryptMsgNew(msg1, pkSet.GetPublicKey("user1"))
	ct2 := encryptor.EncryptMsgNew(msg2, pkSet.GetPublicKey("user2"))

	ctSum := evaluator.AddNew(ct1, ct2)

	decryptedSum := decryptor.Decrypt(ctSum, skSet)
	fmt.Println("\nSum results and expected binary values:")
	for i := 0; i < 8; i++ {
		sum := real(decryptedSum.Value[i])
		expectedBinary := 0.0
		if sum > 0 {
			expectedBinary = 1.0
		}
		fmt.Printf("Sum[%d] = %.6f, Expected: %.0f\n", i, sum, expectedBinary)
	}

	// Precomputed Logistic Coefficients
	coefs := []float64{
		5.0000000000e-01,  // x^0
		2.2096262864e-01,  // x^1
		-3.9670928355e-15, // x^2
		-9.3600405460e-03, // x^3
		5.0178215784e-16,  // x^4
		2.5838970275e-04,  // x^5
		-1.3098791429e-17, // x^6
		-4.0374611259e-06, // x^7
		0,                 // x^8
		3.6299656113e-08,  // x^9
		0,                 // x^10
		-1.8597436477e-10, // x^11
		0,                 // x^12
		5.0380255300e-13,  // x^13
		0,                 // x^14
		-5.5942238141e-16, // x^15
	}

	result, err := EvalPolynomialPS(coefs, ctSum, constOne, mkParams, evaluator, rlkSet)
	if err != nil {
		t.Fatalf("EvaluatePolynomial failed: %v", err)
	}

	decryptedResult := decryptor.Decrypt(result, skSet)
	fmt.Println("\nLogistic approximation results comparison:")
	for i := 0; i < 8; i++ {
		sum := real(decryptedSum.Value[i])
		psApprox := real(decryptedResult.Value[i])
		plainApprox := evaluatePlainPolynomial(sum, coefs)
		expected := 0.0
		if sum > 0 {
			expected = 1.0
		}

		psError := math.Abs(psApprox - expected)
		plainError := math.Abs(plainApprox - expected)
		approxDiff := math.Abs(psApprox - plainApprox)

		fmt.Printf("\nIndex[%d]:\n"+
			"  Input sum = %.6f\n"+
			"  PS result = %.6f\n"+
			"  Plain poly result = %.6f\n"+
			"  Expected = %.0f\n"+
			"  PS vs Expected error = %.6f\n"+
			"  Plain vs Expected error = %.6f\n"+
			"  PS vs Plain difference = %.6f\n",
			i, sum, psApprox, plainApprox, expected,
			psError, plainError, approxDiff)
	}
}

func TestSignF3(t *testing.T) {
	params, _ := ckks.NewParametersFromLiteral(ckks.PN15QP827pq)
	mkParams := mkckks.NewParameters(params)
	kgen := mkckks.NewKeyGenerator(mkParams)

	users := []string{"user1", "user2"}
	skSet := mkrlwe.NewSecretKeySet()
	pkSet := mkrlwe.NewPublicKeyKeySet()
	rlkSet := mkrlwe.NewRelinearizationKeySet(mkParams.Parameters)

	for _, user := range users {
		sk, pk := kgen.GenKeyPair(user)
		rlk := kgen.GenRelinearizationKey(sk)
		skSet.AddSecretKey(sk)
		pkSet.AddPublicKey(pk)
		rlkSet.AddRelinearizationKey(rlk)
	}

	encryptor := mkckks.NewEncryptor(mkParams)
	evaluator := mkckks.NewEvaluator(mkParams)
	decryptor := mkckks.NewDecryptor(mkParams)

	constMsg := mkckks.NewMessage(mkParams)
	for i := 0; i < constMsg.Slots(); i++ {
		constMsg.Value[i] = complex(1.0, 0)
	}
	constOne := encryptor.EncryptMsgNew(constMsg, pkSet.GetPublicKey("user1"))

	msg := mkckks.NewMessage(mkParams)
	realValues := make([]float64, 128)

	fmt.Println("\nInput values:")
	for i := 0; i < 128; i++ {
		realValues[i] = -1.0 + (2.0 * float64(i) / 128.0)
		msg.Value[i] = complex(realValues[i], 0)
		fmt.Printf("Input[%d] = %.6f\n", i, realValues[i])
	}

	ct := encryptor.EncryptMsgNew(msg, pkSet.GetPublicKey("user1"))

	fmt.Printf("\n=== Testing Sign with F3 (iterations: 4) ===\n")
	signResult, err := EvalSign(ct, constOne, mkParams, evaluator, rlkSet, false, 4)
	if err != nil {
		t.Fatalf("EvalSign F3 failed: %v", err)
	}

	decryptedSign := decryptor.Decrypt(signResult, skSet)
	fmt.Println("\nSign function results comparison:")

	for i := 0; i < 128; i++ {
		input := realValues[i]
		result := real(decryptedSign.Value[i])
		expected := 1.0
		if input < 0 {
			expected = -1.0
		}
		error := math.Abs(result - expected)

		fmt.Printf("\nIndex[%d]:\n Input = %.6f\n Sign result = %.6f\n Expected = %.6f\n Error = %.6f\n",
			i, input, result, expected, error)
	}
}

func TestSignF4(t *testing.T) {
	params, _ := ckks.NewParametersFromLiteral(ckks.PN15QP827pq)
	mkParams := mkckks.NewParameters(params)
	kgen := mkckks.NewKeyGenerator(mkParams)

	users := []string{"user1", "user2"}
	skSet := mkrlwe.NewSecretKeySet()
	pkSet := mkrlwe.NewPublicKeyKeySet()
	rlkSet := mkrlwe.NewRelinearizationKeySet(mkParams.Parameters)

	for _, user := range users {
		sk, pk := kgen.GenKeyPair(user)
		rlk := kgen.GenRelinearizationKey(sk)
		skSet.AddSecretKey(sk)
		pkSet.AddPublicKey(pk)
		rlkSet.AddRelinearizationKey(rlk)
	}

	encryptor := mkckks.NewEncryptor(mkParams)
	evaluator := mkckks.NewEvaluator(mkParams)
	decryptor := mkckks.NewDecryptor(mkParams)

	constMsg := mkckks.NewMessage(mkParams)
	for i := 0; i < constMsg.Slots(); i++ {
		constMsg.Value[i] = complex(1.0, 0)
	}
	constOne := encryptor.EncryptMsgNew(constMsg, pkSet.GetPublicKey("user1"))

	msg := mkckks.NewMessage(mkParams)
	realValues := make([]float64, 16)

	fmt.Println("\nInput values:")
	for i := 0; i < 16; i++ {
		realValues[i] = -1.0 + (2.0 * float64(i) / 15.0)
		msg.Value[i] = complex(realValues[i], 0)
		fmt.Printf("Input[%d] = %.6f\n", i, realValues[i])
	}

	ct := encryptor.EncryptMsgNew(msg, pkSet.GetPublicKey("user1"))

	fmt.Printf("\n=== Testing Sign with F4 (iterations: 1) ===\n")
	signResult, err := EvalSign(ct, constOne, mkParams, evaluator, rlkSet, true, 3)
	if err != nil {
		t.Fatalf("EvalSign F4 failed: %v", err)
	}

	decryptedSign := decryptor.Decrypt(signResult, skSet)
	fmt.Println("\nSign function results comparison:")

	for i := 0; i < 16; i++ {
		input := realValues[i]
		result := real(decryptedSign.Value[i])
		expected := 1.0
		if input < 0 {
			expected = -1.0
		}
		error := math.Abs(result - expected)

		fmt.Printf("\nIndex[%d]:\n Input = %.6f\n Sign result = %.6f\n Expected = %.6f\n Error = %.6f\n",
			i, input, result, expected, error)
	}
}

func TestReLUF3(t *testing.T) {
	params, _ := ckks.NewParametersFromLiteral(ckks.PN15QP827pq)
	mkParams := mkckks.NewParameters(params)
	kgen := mkckks.NewKeyGenerator(mkParams)

	users := []string{"user1", "user2"}
	skSet := mkrlwe.NewSecretKeySet()
	pkSet := mkrlwe.NewPublicKeyKeySet()
	rlkSet := mkrlwe.NewRelinearizationKeySet(mkParams.Parameters)

	for _, user := range users {
		sk, pk := kgen.GenKeyPair(user)
		rlk := kgen.GenRelinearizationKey(sk)
		skSet.AddSecretKey(sk)
		pkSet.AddPublicKey(pk)
		rlkSet.AddRelinearizationKey(rlk)
	}

	encryptor := mkckks.NewEncryptor(mkParams)
	evaluator := mkckks.NewEvaluator(mkParams)
	decryptor := mkckks.NewDecryptor(mkParams)

	constMsg := mkckks.NewMessage(mkParams)
	for i := 0; i < constMsg.Slots(); i++ {
		constMsg.Value[i] = complex(1.0, 0)
	}
	constOne := encryptor.EncryptMsgNew(constMsg, pkSet.GetPublicKey("user1"))

	msg := mkckks.NewMessage(mkParams)
	realValues := make([]float64, 16)

	fmt.Println("\nInput values:")
	for i := 0; i < 16; i++ {
		realValues[i] = -1.0 + (2.0 * float64(i) / 15.0)
		msg.Value[i] = complex(realValues[i], 0)
		fmt.Printf("Input[%d] = %.6f\n", i, realValues[i])
	}

	ct := encryptor.EncryptMsgNew(msg, pkSet.GetPublicKey("user1"))

	fmt.Printf("\n=== Testing ReLU with F3 (iterations: 1) ===\n")
	reluResult, err := EvalReLU(ct, constOne, mkParams, evaluator, rlkSet, false, 3)
	if err != nil {
		t.Fatalf("EvalReLU F3 failed: %v", err)
	}

	decryptedReLU := decryptor.Decrypt(reluResult, skSet)
	fmt.Println("\nReLU function results comparison:")

	for i := 0; i < 16; i++ {
		input := realValues[i]
		result := real(decryptedReLU.Value[i])
		expected := math.Max(0, input)
		error := math.Abs(result - expected)

		fmt.Printf("\nIndex[%d]:\n Input = %.6f\n ReLU result = %.6f\n Expected = %.6f\n Error = %.6f\n",
			i, input, result, expected, error)
	}
}

func TestReLUF4(t *testing.T) {
	params, _ := ckks.NewParametersFromLiteral(ckks.PN15QP827pq)
	mkParams := mkckks.NewParameters(params)
	kgen := mkckks.NewKeyGenerator(mkParams)

	users := []string{"user1", "user2"}
	skSet := mkrlwe.NewSecretKeySet()
	pkSet := mkrlwe.NewPublicKeyKeySet()
	rlkSet := mkrlwe.NewRelinearizationKeySet(mkParams.Parameters)

	for _, user := range users {
		sk, pk := kgen.GenKeyPair(user)
		rlk := kgen.GenRelinearizationKey(sk)
		skSet.AddSecretKey(sk)
		pkSet.AddPublicKey(pk)
		rlkSet.AddRelinearizationKey(rlk)
	}

	encryptor := mkckks.NewEncryptor(mkParams)
	evaluator := mkckks.NewEvaluator(mkParams)
	decryptor := mkckks.NewDecryptor(mkParams)

	constMsg := mkckks.NewMessage(mkParams)
	for i := 0; i < constMsg.Slots(); i++ {
		constMsg.Value[i] = complex(1.0, 0)
	}
	constOne := encryptor.EncryptMsgNew(constMsg, pkSet.GetPublicKey("user1"))

	msg := mkckks.NewMessage(mkParams)
	realValues := make([]float64, 32)

	fmt.Println("\nInput values:")
	for i := 0; i < 32; i++ {
		realValues[i] = -1.0 + (2.0 * float64(i) / 32.0)
		msg.Value[i] = complex(realValues[i], 0)
		fmt.Printf("Input[%d] = %.6f\n", i, realValues[i])
	}

	ct := encryptor.EncryptMsgNew(msg, pkSet.GetPublicKey("user1"))

	fmt.Printf("\n=== Testing ReLU with F4 (iterations: 1) ===\n")
	reluResult, err := EvalReLU(ct, constOne, mkParams, evaluator, rlkSet, true, 3)
	if err != nil {
		t.Fatalf("EvalReLU F4 failed: %v", err)
	}

	decryptedReLU := decryptor.Decrypt(reluResult, skSet)
	fmt.Println("\nReLU function results comparison:")

	for i := 0; i < 16; i++ {
		input := realValues[i]
		result := real(decryptedReLU.Value[i])
		expected := math.Max(0, input)
		error := math.Abs(result - expected)

		fmt.Printf("\nIndex[%d]:\n Input = %.6f\n ReLU result = %.6f\n Expected = %.6f\n Error = %.6f\n",
			i, input, result, expected, error)
	}
}

func TestScaledReLU(t *testing.T) {
	params, _ := ckks.NewParametersFromLiteral(ckks.PN15QP827pq)
	mkParams := mkckks.NewParameters(params)
	kgen := mkckks.NewKeyGenerator(mkParams)

	users := []string{"user1", "user2"}
	skSet := mkrlwe.NewSecretKeySet()
	pkSet := mkrlwe.NewPublicKeyKeySet()
	rlkSet := mkrlwe.NewRelinearizationKeySet(mkParams.Parameters)

	for _, user := range users {
		sk, pk := kgen.GenKeyPair(user)
		rlk := kgen.GenRelinearizationKey(sk)
		skSet.AddSecretKey(sk)
		pkSet.AddPublicKey(pk)
		rlkSet.AddRelinearizationKey(rlk)
	}

	encryptor := mkckks.NewEncryptor(mkParams)
	evaluator := mkckks.NewEvaluator(mkParams)
	decryptor := mkckks.NewDecryptor(mkParams)

	constMsg := mkckks.NewMessage(mkParams)
	for i := 0; i < constMsg.Slots(); i++ {
		constMsg.Value[i] = complex(1.0, 0)
	}
	constOne := encryptor.EncryptMsgNew(constMsg, pkSet.GetPublicKey("user1"))

	msg := mkckks.NewMessage(mkParams)
	realValues := make([]float64, 128)

	scale := 10.0
	fmt.Printf("\nTesting ScaledReLU with scale factor: %.1f\n", scale)
	fmt.Println("\nInput values:")
	for i := 0; i < 128; i++ {
		// Generate values from -10 to 10
		realValues[i] = -10.0 + (20.0 * float64(i) / 127.0)
		msg.Value[i] = complex(realValues[i], 0)
		fmt.Printf("Input[%d] = %.6f\n", i, realValues[i])
	}

	ct := encryptor.EncryptMsgNew(msg, pkSet.GetPublicKey("user1"))

	fmt.Printf("\n=== Testing ScaledReLU (scale: %.1f) ===\n", scale)
	reluResult, err := EvalScaledReLU(ct, constOne, mkParams, evaluator, rlkSet, scale)
	if err != nil {
		t.Fatalf("EvalScaledReLU failed: %v", err)
	}

	decryptedReLU := decryptor.Decrypt(reluResult, skSet)
	fmt.Println("\nScaledReLU function results comparison:")

	maxError := 0.0
	avgError := 0.0
	totalSamples := 0

	for i := 0; i < 128; i++ {
		input := realValues[i]
		result := real(decryptedReLU.Value[i])
		expected := math.Max(0, input)
		error := math.Abs(result - expected)
		maxError = math.Max(maxError, error)
		avgError += error
		totalSamples++

		fmt.Printf("\nIndex[%d]:\n"+
			"  Input = %.6f\n"+
			"  ScaledReLU result = %.6f\n"+
			"  Expected = %.6f\n"+
			"  Error = %.6f\n",
			i, input, result, expected, error)
	}

	avgError /= float64(totalSamples)
	fmt.Printf("\nError Analysis:\n"+
		"Maximum Error: %.6f\n"+
		"Average Error: %.6f\n",
		maxError, avgError)

	if maxError > 1e-2 {
		t.Errorf("Maximum error (%.6f) exceeds threshold (1e-2)", maxError)
	}
}

// go test -v -run TestReLUF4
