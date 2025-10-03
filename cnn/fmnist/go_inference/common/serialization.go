package common

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"fmt"
	ser "secure-ensemble/pkg/serialization"

	"github.com/anony-submit/snu-mghe/mkckks"
	"github.com/ldsec/lattigo/v2/ring"
)

type FMNISTModelBytes struct {
	ConvWeights []byte    `json:"conv_weights"`
	ConvBias    []byte    `json:"conv_bias"`
	FC1Weights  [8][]byte `json:"fc1_weights"`
	FC1Bias     []byte    `json:"fc1_bias"`
	FC2Weights  []byte    `json:"fc2_weights"`
	FC2Bias     []byte    `json:"fc2_bias"`
}

func SerializeFMNISTModel(convW, convB *mkckks.Ciphertext, fc1w [8]*mkckks.Ciphertext, fc1b, fc2w, fc2b *mkckks.Ciphertext) ([]byte, error) {
	modelBytes := FMNISTModelBytes{}

	var err error
	if modelBytes.ConvWeights, err = ser.SerializeCiphertext(convW); err != nil {
		return nil, fmt.Errorf("conv weights serialization failed: %v", err)
	}

	if modelBytes.ConvBias, err = ser.SerializeCiphertext(convB); err != nil {
		return nil, fmt.Errorf("conv bias serialization failed: %v", err)
	}

	for i := 0; i < 8; i++ {
		if modelBytes.FC1Weights[i], err = ser.SerializeCiphertext(fc1w[i]); err != nil {
			return nil, fmt.Errorf("FC1 weight serialization failed at index %d: %v", i, err)
		}
	}

	if modelBytes.FC1Bias, err = ser.SerializeCiphertext(fc1b); err != nil {
		return nil, fmt.Errorf("FC1 bias serialization failed: %v", err)
	}

	if modelBytes.FC2Weights, err = ser.SerializeCiphertext(fc2w); err != nil {
		return nil, fmt.Errorf("FC2 weights serialization failed: %v", err)
	}

	if modelBytes.FC2Bias, err = ser.SerializeCiphertext(fc2b); err != nil {
		return nil, fmt.Errorf("FC2 bias serialization failed: %v", err)
	}

	return json.Marshal(modelBytes)
}

func DeserializeFMNISTModel(data []byte, params mkckks.Parameters) (*mkckks.Ciphertext, *mkckks.Ciphertext, [8]*mkckks.Ciphertext, *mkckks.Ciphertext, *mkckks.Ciphertext, *mkckks.Ciphertext, error) {
	var modelBytes FMNISTModelBytes
	if err := json.Unmarshal(data, &modelBytes); err != nil {
		var emptyFC1Weights [8]*mkckks.Ciphertext
		return nil, nil, emptyFC1Weights, nil, nil, nil, fmt.Errorf("model bytes unmarshal failed: %v", err)
	}

	convWeights, err := ser.DeserializeCiphertext(modelBytes.ConvWeights, params)
	if err != nil {
		var emptyFC1Weights [8]*mkckks.Ciphertext
		return nil, nil, emptyFC1Weights, nil, nil, nil, fmt.Errorf("conv weights deserialization failed: %v", err)
	}

	convBias, err := ser.DeserializeCiphertext(modelBytes.ConvBias, params)
	if err != nil {
		var emptyFC1Weights [8]*mkckks.Ciphertext
		return nil, nil, emptyFC1Weights, nil, nil, nil, fmt.Errorf("conv bias deserialization failed: %v", err)
	}

	fc1Weights := [8]*mkckks.Ciphertext{}
	for i := 0; i < 8; i++ {
		fc1Weights[i], err = ser.DeserializeCiphertext(modelBytes.FC1Weights[i], params)
		if err != nil {
			var emptyFC1Weights [8]*mkckks.Ciphertext
			return nil, nil, emptyFC1Weights, nil, nil, nil, fmt.Errorf("FC1 weight deserialization failed at index %d: %v", i, err)
		}
	}

	fc1Bias, err := ser.DeserializeCiphertext(modelBytes.FC1Bias, params)
	if err != nil {
		var emptyFC1Weights [8]*mkckks.Ciphertext
		return nil, nil, emptyFC1Weights, nil, nil, nil, fmt.Errorf("FC1 bias deserialization failed: %v", err)
	}

	fc2Weights, err := ser.DeserializeCiphertext(modelBytes.FC2Weights, params)
	if err != nil {
		var emptyFC1Weights [8]*mkckks.Ciphertext
		return nil, nil, emptyFC1Weights, nil, nil, nil, fmt.Errorf("FC2 weights deserialization failed: %v", err)
	}

	fc2Bias, err := ser.DeserializeCiphertext(modelBytes.FC2Bias, params)
	if err != nil {
		var emptyFC1Weights [8]*mkckks.Ciphertext
		return nil, nil, emptyFC1Weights, nil, nil, nil, fmt.Errorf("FC2 bias deserialization failed: %v", err)
	}

	return convWeights, convBias, fc1Weights, fc1Bias, fc2Weights, fc2Bias, nil
}

// PolyBytes represents the serialized form of a ring.Poly
type PolyBytes struct {
	Value []byte
}

// SerializePoly serializes a ring.Poly
func SerializePoly(p *ring.Poly) ([]byte, error) {
	data := make([]byte, p.GetDataLen(true))
	if _, err := p.WriteTo(data); err != nil {
		return nil, fmt.Errorf("failed to serialize poly: %v", err)
	}

	polyBytes := &PolyBytes{Value: data}

	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(polyBytes); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// DeserializePoly deserializes bytes into a ring.Poly
func DeserializePoly(data []byte, ringQ *ring.Ring) (*ring.Poly, error) {
	var polyBytes PolyBytes
	if err := gob.NewDecoder(bytes.NewBuffer(data)).Decode(&polyBytes); err != nil {
		return nil, err
	}

	p := ringQ.NewPoly()
	if _, err := p.DecodePolyNew(polyBytes.Value); err != nil {
		return nil, fmt.Errorf("failed to deserialize poly: %v", err)
	}

	return p, nil
}
