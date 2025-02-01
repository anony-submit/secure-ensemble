package activation

import (
	"fmt"
	"log"
	"math/bits"

	"github.com/anony-submit/snu-mghe/mkckks"
	"github.com/anony-submit/snu-mghe/mkrlwe"
)

type PowerBasis struct {
	Value map[int]*mkckks.Ciphertext // Stores only powers of 2 up to largest power <= degree
}

func NewPowerBasis(ct *mkckks.Ciphertext, constOne *mkckks.Ciphertext, evaluator *mkckks.Evaluator) PowerBasis {
	pb := PowerBasis{
		Value: make(map[int]*mkckks.Ciphertext),
	}
	pb.Value[1] = ct.CopyNew()
	pb.Value[0] = constOne.CopyNew()
	return pb
}

func (pb *PowerBasis) GenPower(n int, evaluator *mkckks.Evaluator, rlkSet *mkrlwe.RelinearizationKeySet) error {
	if pb.Value[n] != nil {
		return nil
	}

	if n <= 1 {
		return nil
	}

	// Only generate powers of 2
	if n&(n-1) != 0 {
		return fmt.Errorf("only power of 2 is supported: %d", n)
	}

	half := n / 2
	if err := pb.GenPower(half, evaluator, rlkSet); err != nil {
		return fmt.Errorf("power of 2 generation failed at %d: %w", half, err)
	}

	// log.Printf("=== Generating power basis x^%d ===", n)
	// log.Printf("Operation: Multiplying x^%d * x^%d", half, half)
	// log.Printf("Before multiplication - Level of x^%d: %d", half, pb.Value[half].Level())

	pb.Value[n] = evaluator.MulRelinNew(pb.Value[half], pb.Value[half], rlkSet)

	// log.Printf("After multiplication - Level of x^%d: %d\n", n, pb.Value[n].Level())
	return nil
}

func SplitDegree(n int) (a, b int) {
	if n <= 1 {
		return n, 0
	}
	if n&(n-1) == 0 {
		return n / 2, n / 2
	}
	k := bits.Len64(uint64(n-1)) - 1
	base := 1 << k
	if n-base < base/2 {
		a = base / 2
		b = n - base/2
	} else {
		a = base - 1
		b = n - (base - 1)
	}
	return
}

// EvalPolynomialNaive evaluate polynomials sequentially
func EvalPolynomialNaive(coefs []float64, ct *mkckks.Ciphertext, constOne *mkckks.Ciphertext,
	evaluator *mkckks.Evaluator, rlkSet *mkrlwe.RelinearizationKeySet) (*mkckks.Ciphertext, error) {

	log.Printf("\nStarting naive polynomial evaluation with coefficients: %v", coefs)

	if len(coefs) == 0 {
		return nil, fmt.Errorf("empty coefficient array")
	}

	var result *mkckks.Ciphertext

	if coefs[0] != 0 {
		result = constOne.CopyNew()
		evaluator.MultByConst(result, complex(coefs[0], 0), result)
	}

	if len(coefs) > 1 && coefs[1] != 0 {
		term := ct.CopyNew()
		evaluator.MultByConst(term, complex(coefs[1], 0), term)

		if result == nil {
			result = term
		} else {
			result = evaluator.AddNew(result, term)
		}
	}

	currentPower := ct.CopyNew()
	for i := 2; i < len(coefs); i++ {
		currentPower = evaluator.MulRelinNew(currentPower, ct, rlkSet)
		if currentPower == nil {
			return nil, fmt.Errorf("multiplication failed at power %d", i)
		}

		if coefs[i] == 0 {
			continue
		}

		term := currentPower.CopyNew()
		evaluator.MultByConst(term, complex(coefs[i], 0), term)

		if result == nil {
			result = term
		} else {
			result = evaluator.AddNew(result, term)
			if result == nil {
				return nil, fmt.Errorf("addition failed at term %d", i)
			}
		}
	}

	if result == nil {
		result = constOne.CopyNew()
		evaluator.MultByConst(result, complex(0, 0), result)
	}

	return result, nil
}

func EvalPolynomialPS(coefs []float64, ct *mkckks.Ciphertext, constOne *mkckks.Ciphertext, params mkckks.Parameters,
	evaluator *mkckks.Evaluator, rlkSet *mkrlwe.RelinearizationKeySet) (*mkckks.Ciphertext, error) {

	if len(coefs) == 0 {
		return nil, fmt.Errorf("empty coefficient array")
	}

	degree := len(coefs) - 1
	for degree >= 0 && coefs[degree] == 0 {
		degree--
	}

	if degree < 0 {
		result := constOne.CopyNew()
		evaluator.MultByConst(result, complex(0, 0), result)
		if err := evaluator.Rescale(result, params.Scale(), result); err != nil {
			return nil, fmt.Errorf("rescale failed after MultByConst: %w", err)
		}
		return result, nil
	}

	logDegree := bits.Len64(uint64(degree))
	m := (logDegree + 1) / 2
	splitSize := 1 << m

	// log.Printf("PS polynomial evaluation - degree: %d, splitSize: %d", degree, splitSize)

	// Generate power basis (only powers of 2)
	pb := NewPowerBasis(ct, constOne, evaluator)
	maxPowerOf2 := 1 << bits.Len64(uint64(degree))
	for i := 2; i < maxPowerOf2; i *= 2 {
		if err := pb.GenPower(i, evaluator, rlkSet); err != nil {
			return nil, fmt.Errorf("power generation failed at %d: %w", i, err)
		}
	}

	var result *mkckks.Ciphertext

	// Baby-step Giant-step
	for i := 0; i <= degree/splitSize; i++ {
		var subSum *mkckks.Ciphertext
		firstInSplit := true

		// Evaluate Baby-step
		for j := 0; j < splitSize && i*splitSize+j <= degree; j++ {
			coefIdx := i*splitSize + j
			if coefs[coefIdx] == 0 {
				continue
			}

			// log.Printf("======= Processing term: %fx^%d =======", coefs[coefIdx], i*splitSize+j)

			// Find the largest power of 2 less than or equal to j
			power := uint(bits.Len64(uint64(j))) - 1
			base := pb.Value[1<<power]
			term := base.CopyNew()
			remaining := j - (1 << power)

			if remaining > 0 {
				// If j is not a power of 2
				remainingPower := bits.Len64(uint64(remaining)) - 1
				remainingBase := pb.Value[1<<remainingPower]
				cterm := remainingBase.CopyNew()

				evaluator.MultByConst(cterm, complex(coefs[coefIdx], 0), cterm)
				if err := evaluator.Rescale(cterm, params.Scale(), cterm); err != nil {
					return nil, fmt.Errorf("rescale failed after MultByConst: %w", err)
				}
				// log.Printf("After constant multiplication and rescale - Level: %d", cterm.Level())
				// log.Printf("Multiplying Levels cterm, term: %d, %d", cterm.Level(), term.Level())
				term = evaluator.MulRelinNew(term, cterm, rlkSet)
				// log.Printf("After remaining multiplication - Level: %d", term.Level())

			} else {
				// log.Printf("Before constant multiplication - Level: %d", term.Level())
				evaluator.MultByConst(term, complex(coefs[coefIdx], 0), term)

				if err := evaluator.Rescale(term, params.Scale(), term); err != nil {
					return nil, fmt.Errorf("rescale failed after MultByConst: %w", err)
				}
				// log.Printf("After constant multiplication and rescale - Level: %d", term.Level())
			}

			if firstInSplit {
				subSum = term
				firstInSplit = false
			} else {
				// log.Printf("Adding terms - Levels: %d, %d", subSum.Level(), term.Level())
				subSum = evaluator.AddNew(subSum, term)
				if subSum == nil {
					return nil, fmt.Errorf("failed to add term in split %d", i)
				}
			}
			// log.Print(("============================================="))
		}

		if subSum == nil {
			continue
		}

		// Giant-step using closest power of 2
		if i > 0 {
			powerOf2 := uint(bits.Len64(uint64(i*splitSize))) - 1
			giant := pb.Value[1<<powerOf2]

			// log.Printf("Giant step multiplication - Levels: %d, %d", subSum.Level(), giant.Level())
			subSum = evaluator.MulRelinNew(subSum, giant, rlkSet)
			// log.Printf("After giant step - Level: %d", subSum.Level())

			if subSum == nil {
				return nil, fmt.Errorf("failed giant step multiplication at split %d", i)
			}
		}

		if result == nil {
			result = subSum
		} else {
			// log.Printf("Adding split results - Levels: %d, %d", result.Level(), subSum.Level())
			result = evaluator.AddNew(result, subSum)
			if result == nil {
				return nil, fmt.Errorf("failed to add split result at %d", i)
			}
		}
	}

	if result == nil {
		result = constOne.CopyNew()
		evaluator.MultByConst(result, complex(0, 0), result)
		if err := evaluator.Rescale(result, params.Scale(), result); err != nil {
			return nil, fmt.Errorf("rescale failed after MultByConst: %w", err)
		}
	}

	return result, nil

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Activation Functions ////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

func EvalLogistic(ct *mkckks.Ciphertext, constOne *mkckks.Ciphertext, params mkckks.Parameters,
	evaluator *mkckks.Evaluator, rlkSet *mkrlwe.RelinearizationKeySet) (*mkckks.Ciphertext, error) {

	// Initialize coefficients array with zeros
	// coefs := make([]float64, 8)  // Size 16 to accommodate x^7
	coefs := make([]float64, 16) // Size 16 to accommodate x^15

	// Set the coefficients

	// degree of 7
	// coefs[0] = 5.0000000000e-01  // x^0
	// coefs[1] = 1.9686870706e-01  // x^1
	// coefs[2] = -7.3914582501e-17 // x^2
	// coefs[3] = 5.4411280489e-03  // x^3
	// coefs[4] = 2.1557781782e-18  // x^4
	// coefs[5] = 7.4959264610e-05  // x^5
	// coefs[6] = 0                 // x^6
	// coefs[7] = -3.5735824382e-07 // x^7

	// degree of 15
	coefs[0] = 5.0000000000e-01   // x^0
	coefs[1] = 2.4250510146e-01   // x^1
	coefs[2] = 1.3290526344e-14   // x^2
	coefs[3] = -1.5524259607e-02  // x^3
	coefs[4] = -7.7605063158e-16  // x^4
	coefs[5] = 7.8715742808e-04   // x^5
	coefs[6] = 1.2932654458e-17   // x^6
	coefs[7] = -2.4803847179e-05  // x^7
	coefs[8] = 0                  // x^8
	coefs[9] = 4.7027883543e-07   // x^9
	coefs[10] = 0                 // x^10
	coefs[11] = -5.2000211423e-09 // x^11
	coefs[12] = 0                 // x^12
	coefs[13] = 3.0802408045e-11  // x^13
	coefs[14] = 0                 // x^14
	coefs[15] = -7.5384645302e-14 // x^15

	// Evaluate the polynomial using Paterson-Stockmeyer algorithm
	result, err := EvalPolynomialPS(coefs, ct, constOne, params, evaluator, rlkSet)
	if err != nil {
		return nil, fmt.Errorf("failed to evaluate logistic polynomial: %w", err)
	}

	return result, nil
}

func EvalF3(ct *mkckks.Ciphertext, constOne *mkckks.Ciphertext, params mkckks.Parameters,
	evaluator *mkckks.Evaluator, rlkSet *mkrlwe.RelinearizationKeySet) (*mkckks.Ciphertext, error) {

	// Initialize coefficients array with zeros
	// coefs := make([]float64, 8)  // Size 16 to accommodate x^7
	coefs := make([]float64, 8) // Size 16 to accommodate x^15

	// Set the coefficients
	// degree of 7
	coefs[0] = 0             // x^0
	coefs[1] = 2.1875000000  // x^1
	coefs[2] = 0             // x^2
	coefs[3] = -2.1875000000 // x^3
	coefs[4] = 0             // x^4
	coefs[5] = 1.3125000000  // x^5
	coefs[6] = 0             // x^6
	coefs[7] = -0.3125000000 // x^7

	// Evaluate the polynomial using Paterson-Stockmeyer algorithm
	result, err := EvalPolynomialPS(coefs, ct, constOne, params, evaluator, rlkSet)
	if err != nil {
		return nil, fmt.Errorf("failed to evaluate F3 polynomial: %w", err)
	}

	return result, nil
}

func EvalF4(ct *mkckks.Ciphertext, constOne *mkckks.Ciphertext, params mkckks.Parameters,
	evaluator *mkckks.Evaluator, rlkSet *mkrlwe.RelinearizationKeySet) (*mkckks.Ciphertext, error) {

	// Initialize coefficients array with zeros
	// coefs := make([]float64, 8)  // Size 16 to accommodate x^7
	coefs := make([]float64, 16) // Size 16 to accommodate x^15

	// Set the coefficients
	// degree of 15
	coefs[0] = 0              // x^0
	coefs[1] = 3.1420898438   // x^1
	coefs[2] = 0              // x^2
	coefs[3] = -7.3315429688  // x^3
	coefs[4] = 0              // x^4
	coefs[5] = 13.1967773438  // x^5
	coefs[6] = 0              // x^6
	coefs[7] = -15.7104492188 // x^7
	coefs[8] = 0              // x^8
	coefs[9] = 12.2192382813  // x^9
	coefs[10] = 0             // x^10
	coefs[11] = -5.9985351563 // x^11
	coefs[12] = 0             // x^12
	coefs[13] = 1.6918945313  // x^13
	coefs[14] = 0             // x^14
	coefs[15] = -0.2094726563 // x^15

	// Evaluate the polynomial using Paterson-Stockmeyer algorithm
	result, err := EvalPolynomialPS(coefs, ct, constOne, params, evaluator, rlkSet)
	if err != nil {
		return nil, fmt.Errorf("failed to evaluate F4 polynomial: %w", err)
	}

	return result, nil
}

func EvalSign(ct *mkckks.Ciphertext, constOne *mkckks.Ciphertext, params mkckks.Parameters,
	evaluator *mkckks.Evaluator, rlkSet *mkrlwe.RelinearizationKeySet,
	useF4 bool, iterations int) (*mkckks.Ciphertext, error) {

	result := ct.CopyNew()
	var err error

	for i := 0; i < iterations; i++ {
		if useF4 {
			result, err = EvalF4(result, constOne, params, evaluator, rlkSet)
		} else {
			result, err = EvalF3(result, constOne, params, evaluator, rlkSet)
		}

		if err != nil {
			return nil, fmt.Errorf("iteration %d failed: %w", i, err)
		}
	}

	return result, nil
}

func EvalScaledReLU(ct *mkckks.Ciphertext, constOne *mkckks.Ciphertext, params mkckks.Parameters,
	evaluator *mkckks.Evaluator, rlkSet *mkrlwe.RelinearizationKeySet, scale float64) (*mkckks.Ciphertext, error) {

	// Scale down the input
	scaledCt := ct.CopyNew()
	evaluator.MultByConst(scaledCt, complex(1.0/scale, 0), scaledCt)
	if err := evaluator.Rescale(scaledCt, params.Scale(), scaledCt); err != nil {
		return nil, fmt.Errorf("rescale failed after scaling down: %w", err)
	}

	// Apply ReLU with fixed f3 and 3 iterations
	ctHalf := ct.CopyNew()
	evaluator.MultByConst(ctHalf, complex(0.5, 0), ctHalf)
	if err := evaluator.Rescale(ctHalf, params.Scale(), ctHalf); err != nil {
		return nil, fmt.Errorf("rescale failed after MultByConst: %w", err)
	}

	log.Printf("ScaledReLU - Before Sign - Level: %d", ctHalf.Level())

	signResult, err := EvalSign(scaledCt, constOne, params, evaluator, rlkSet, false, 3)
	if err != nil {
		return nil, fmt.Errorf("sign evaluation failed: %w", err)
	}

	log.Printf("ScaledReLU - After Sign - Level: %d", signResult.Level())

	result := evaluator.AddNew(constOne, signResult)
	if result == nil {
		return nil, fmt.Errorf("addition failed in ScaledReLU")
	}
	result = evaluator.MulRelinNew(result, ctHalf, rlkSet)

	return result, nil
}

func EvalReLU(ct *mkckks.Ciphertext, constOne *mkckks.Ciphertext, params mkckks.Parameters,
	evaluator *mkckks.Evaluator, rlkSet *mkrlwe.RelinearizationKeySet,
	useF4 bool, iterations int) (*mkckks.Ciphertext, error) {

	ctHalf := ct.CopyNew()
	evaluator.MultByConst(ctHalf, complex(0.5, 0), ctHalf)
	if err := evaluator.Rescale(ctHalf, params.Scale(), ctHalf); err != nil {
		return nil, fmt.Errorf("rescale failed after MultByConst: %w", err)
	}

	log.Printf("ReLU - Before Sign - Level: %d", ctHalf.Level())

	signResult, err := EvalSign(ct, constOne, params, evaluator, rlkSet, useF4, iterations)
	if err != nil {
		return nil, fmt.Errorf("sign evaluation failed: %w", err)
	}

	log.Printf("ReLU - After Sign - Level: %d", signResult.Level())

	result := evaluator.AddNew(constOne, signResult)
	if result == nil {
		return nil, fmt.Errorf("addition failed in ReLU")
	}
	result = evaluator.MulRelinNew(result, ctHalf, rlkSet)

	return result, nil
}
