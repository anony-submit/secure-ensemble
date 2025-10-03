package common

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"secure-ensemble/pkg/serialization"

	"github.com/anony-submit/snu-mghe/mkckks"
	"github.com/ldsec/lattigo/v2/ring"
)

type MNISTModelBytes struct {
	FC1Weights [8][]byte
	FC1Bias    []byte
	FC2Weights []byte
	FC2Bias    []byte
}

func SerializeMNISTModel(fc1w [8]*mkckks.Ciphertext, fc1b, fc2w, fc2b *mkckks.Ciphertext) ([]byte, error) {
	modelBytes := MNISTModelBytes{}

	for i := 0; i < 8; i++ {
		bytes, err := serialization.SerializeCiphertext(fc1w[i])
		if err != nil {
			return nil, fmt.Errorf("FC1 weight serialization failed at index %d: %v", i, err)
		}
		modelBytes.FC1Weights[i] = bytes
	}

	var err error
	if modelBytes.FC1Bias, err = serialization.SerializeCiphertext(fc1b); err != nil {
		return nil, fmt.Errorf("FC1 bias serialization failed: %v", err)
	}

	if modelBytes.FC2Weights, err = serialization.SerializeCiphertext(fc2w); err != nil {
		return nil, fmt.Errorf("FC2 weights serialization failed: %v", err)
	}

	if modelBytes.FC2Bias, err = serialization.SerializeCiphertext(fc2b); err != nil {
		return nil, fmt.Errorf("FC2 bias serialization failed: %v", err)
	}

	return json.Marshal(modelBytes)
}

func DeserializeMNISTModel(data []byte, params mkckks.Parameters) ([8]*mkckks.Ciphertext, *mkckks.Ciphertext, *mkckks.Ciphertext, *mkckks.Ciphertext, error) {
	var modelBytes MNISTModelBytes
	if err := json.Unmarshal(data, &modelBytes); err != nil {
		var emptyFC1Weights [8]*mkckks.Ciphertext
		return emptyFC1Weights, nil, nil, nil, fmt.Errorf("model bytes unmarshal failed: %v", err)
	}

	fc1Weights := [8]*mkckks.Ciphertext{}
	for i := 0; i < 8; i++ {
		ct, err := serialization.DeserializeCiphertext(modelBytes.FC1Weights[i], params)
		if err != nil {
			var emptyFC1Weights [8]*mkckks.Ciphertext
			return emptyFC1Weights, nil, nil, nil, fmt.Errorf("FC1 weight deserialization failed at index %d: %v", i, err)
		}
		fc1Weights[i] = ct
	}

	fc1Bias, err := serialization.DeserializeCiphertext(modelBytes.FC1Bias, params)
	if err != nil {
		var emptyFC1Weights [8]*mkckks.Ciphertext
		return emptyFC1Weights, nil, nil, nil, fmt.Errorf("FC1 bias deserialization failed: %v", err)
	}

	fc2Weights, err := serialization.DeserializeCiphertext(modelBytes.FC2Weights, params)
	if err != nil {
		var emptyFC1Weights [8]*mkckks.Ciphertext
		return emptyFC1Weights, nil, nil, nil, fmt.Errorf("FC2 weights deserialization failed: %v", err)
	}

	fc2Bias, err := serialization.DeserializeCiphertext(modelBytes.FC2Bias, params)
	if err != nil {
		var emptyFC1Weights [8]*mkckks.Ciphertext
		return emptyFC1Weights, nil, nil, nil, fmt.Errorf("FC2 bias deserialization failed: %v", err)
	}

	return fc1Weights, fc1Bias, fc2Weights, fc2Bias, nil
}

func SerializeMNISTResult(result *mkckks.Ciphertext) ([]byte, error) {
	return serialization.SerializeCiphertext(result)
}

func DeserializeMNISTResult(data []byte, params mkckks.Parameters) (*mkckks.Ciphertext, error) {
	return serialization.DeserializeCiphertext(data, params)
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
