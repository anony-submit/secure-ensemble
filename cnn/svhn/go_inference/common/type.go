// common/types.go
package common

import (
	"github.com/anony-submit/snu-mghe/mkckks"
	"github.com/anony-submit/snu-mghe/mkrlwe"
)

type EncryptedModel struct {
	Conv1Weights [6]*mkckks.Ciphertext
	Conv1Bias    *mkckks.Ciphertext
	Conv2Weights [64]*mkckks.Ciphertext
	Conv2Bias    *mkckks.Ciphertext
	FC1Weights   [16]*mkckks.Ciphertext
	FC1Bias      *mkckks.Ciphertext
	FC2Weights   *mkckks.Ciphertext
	FC2Bias      *mkckks.Ciphertext
}

type CryptoKeys struct {
	Params mkckks.Parameters
	SKSet  *mkrlwe.SecretKeySet
	PKSet  *mkrlwe.PublicKeySet
	RLKSet *mkrlwe.RelinearizationKeySet
	RTKSet *mkrlwe.RotationKeySet
}

type ConvParams struct {
	Conv1 Conv1Params `json:"conv1"`
	Conv2 Conv2Params `json:"conv2"`
}

type Conv1Params struct {
	Weight [][][][]float64 `json:"weight"`
	Bias   []float64       `json:"bias"`
}

type Conv2Params struct {
	Weight [][][][]float64 `json:"weight"`
	Bias   []float64       `json:"bias"`
}

type ClassifierParams struct {
	FC1 FC1Params `json:"fc1"`
	FC2 FC2Params `json:"fc2"`
}

type FC1Params struct {
	Weight [][]float64 `json:"weight"`
	Bias   []float64   `json:"bias"`
}

type FC2Params struct {
	Weight [][]float64 `json:"weight"`
	Bias   []float64   `json:"bias"`
}

type ModelParams struct {
	ConvParams       ConvParams       `json:"conv_params"`
	ClassifierParams ClassifierParams `json:"classifier_params"`
}

type StoredModel struct {
	Conv1Weights [][]byte
	Conv1Bias    []byte
	Conv2Weights [][]byte
	Conv2Bias    []byte
	FC1Weights   [][]byte
	FC1Bias      []byte
	FC2Weights   []byte
	FC2Bias      []byte
}

type StoredKeys struct {
	PublicKey          []byte
	RelinearizationKey []byte
	RotationKeys       [][]byte
}
