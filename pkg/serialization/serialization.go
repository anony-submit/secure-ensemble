package serialization

import (
	"bytes"
	"encoding/gob"
	"fmt"

	"github.com/anony-submit/snu-mghe/mkckks"
	"github.com/anony-submit/snu-mghe/mkrlwe"
	"github.com/ldsec/lattigo/v2/ring"
)

// CiphertextBytes represents the serialized form of a Ciphertext
type CiphertextBytes struct {
	Scale     float64
	PolyBytes map[string][]byte // serialized ring.Poly using MarshalBinary
	IDSet     []string
}

// MessageBytes represents the serialized form of a Message
type MessageBytes struct {
	Value []complex128
}

// VotingResultBytes represents the serialized form of multiple voting results
type VotingResultBytes struct {
	SoftVoting         CiphertextBytes
	LogitSoftVoting    CiphertextBytes
	VarianceMaskVoting CiphertextBytes
}

// SerializeCiphertext converts a Ciphertext to bytes
func SerializeCiphertext(ct *mkckks.Ciphertext) ([]byte, error) {
	idSlice := make([]string, 0)
	for id := range ct.IDSet().Value {
		idSlice = append(idSlice, id)
	}

	ctBytes := CiphertextBytes{
		Scale:     ct.Scale,
		PolyBytes: make(map[string][]byte),
		IDSet:     idSlice,
	}

	// Serialize each ring.Poly in the map using MarshalBinary
	for id, poly := range ct.Value {
		polyBytes, err := poly.MarshalBinary()
		if err != nil {
			return nil, fmt.Errorf("failed to serialize poly for id %s: %v", id, err)
		}
		ctBytes.PolyBytes[id] = polyBytes
	}

	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(ctBytes); err != nil {
		return nil, fmt.Errorf("failed to encode CiphertextBytes: %v", err)
	}
	return buf.Bytes(), nil
}

// DeserializeCiphertext converts bytes back to a Ciphertext
func DeserializeCiphertext(data []byte, params mkckks.Parameters) (*mkckks.Ciphertext, error) {
	var ctBytes CiphertextBytes
	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)
	if err := dec.Decode(&ctBytes); err != nil {
		return nil, fmt.Errorf("failed to decode CiphertextBytes: %v", err)
	}

	// Create IDSet from the keys
	idset := mkrlwe.NewIDSet()
	for _, id := range ctBytes.IDSet {
		idset.Add(id)
	}

	// Get the level from the first poly (they should all be the same level)
	firstPoly := new(ring.Poly)
	if err := firstPoly.UnmarshalBinary(ctBytes.PolyBytes["0"]); err != nil {
		return nil, fmt.Errorf("failed to deserialize first poly: %v", err)
	}
	level := firstPoly.Level()

	// Create new ciphertext with appropriate parameters
	ct := mkckks.NewCiphertext(params, idset, level, ctBytes.Scale)

	// Fill in the values
	for id, polyBytes := range ctBytes.PolyBytes {
		poly := new(ring.Poly)
		if err := poly.UnmarshalBinary(polyBytes); err != nil {
			return nil, fmt.Errorf("failed to deserialize poly for id %s: %v", id, err)
		}
		ct.Value[id] = poly
	}

	return ct, nil
}

// SerializeMessage converts a Message to bytes
func SerializeMessage(msg *mkckks.Message) ([]byte, error) {
	msgBytes := MessageBytes{
		Value: msg.Value,
	}
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(msgBytes); err != nil {
		return nil, fmt.Errorf("failed to encode MessageBytes: %v", err)
	}
	return buf.Bytes(), nil
}

// DeserializeMessage converts bytes back to a Message
func DeserializeMessage(data []byte, params mkckks.Parameters) (*mkckks.Message, error) {
	var msgBytes MessageBytes
	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)
	if err := dec.Decode(&msgBytes); err != nil {
		return nil, fmt.Errorf("failed to decode MessageBytes: %v", err)
	}

	msg := mkckks.NewMessage(params)
	msg.Value = msgBytes.Value
	return msg, nil
}

// SerializeVotingResults serializes voting results from all methods
func SerializeVotingResults(soft, logit, variance *mkckks.Ciphertext) ([]byte, error) {
	resultBytes := VotingResultBytes{}

	if soft != nil {
		idSlice := make([]string, 0)
		for id := range soft.IDSet().Value {
			idSlice = append(idSlice, id)
		}
		resultBytes.SoftVoting = CiphertextBytes{
			Scale:     soft.Scale,
			PolyBytes: make(map[string][]byte),
			IDSet:     idSlice,
		}
		for id, poly := range soft.Value {
			polyBytes, err := poly.MarshalBinary()
			if err != nil {
				return nil, fmt.Errorf("failed to serialize poly for soft voting id %s: %v", id, err)
			}
			resultBytes.SoftVoting.PolyBytes[id] = polyBytes
		}
	}

	if logit != nil {
		idSlice := make([]string, 0)
		for id := range logit.IDSet().Value {
			idSlice = append(idSlice, id)
		}
		resultBytes.LogitSoftVoting = CiphertextBytes{
			Scale:     logit.Scale,
			PolyBytes: make(map[string][]byte),
			IDSet:     idSlice,
		}
		for id, poly := range logit.Value {
			polyBytes, err := poly.MarshalBinary()
			if err != nil {
				return nil, fmt.Errorf("failed to serialize poly for logit voting id %s: %v", id, err)
			}
			resultBytes.LogitSoftVoting.PolyBytes[id] = polyBytes
		}
	}

	if variance != nil {
		idSlice := make([]string, 0)
		for id := range logit.IDSet().Value {
			idSlice = append(idSlice, id)
		}
		resultBytes.LogitSoftVoting = CiphertextBytes{
			Scale:     logit.Scale,
			PolyBytes: make(map[string][]byte),
			IDSet:     idSlice,
		}
		for id, poly := range logit.Value {
			polyBytes, err := poly.MarshalBinary()
			if err != nil {
				return nil, fmt.Errorf("failed to serialize poly for logit voting id %s: %v", id, err)
			}
			resultBytes.LogitSoftVoting.PolyBytes[id] = polyBytes
		}
	}

	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(resultBytes); err != nil {
		return nil, fmt.Errorf("failed to encode voting results: %v", err)
	}
	return buf.Bytes(), nil
}

// DeserializeVotingResults deserializes all voting results
func DeserializeVotingResults(data []byte, params mkckks.Parameters) (soft, logit, variance *mkckks.Ciphertext, err error) {
	var resultBytes VotingResultBytes
	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)
	if err = dec.Decode(&resultBytes); err != nil {
		return nil, nil, nil, fmt.Errorf("failed to decode voting results: %v", err)
	}

	if len(resultBytes.SoftVoting.PolyBytes) > 0 {
		idset := mkrlwe.NewIDSet()
		for _, id := range resultBytes.SoftVoting.IDSet {
			idset.Add(id)
		}

		firstPoly := new(ring.Poly)
		if err := firstPoly.UnmarshalBinary(resultBytes.SoftVoting.PolyBytes["0"]); err != nil {
			return nil, nil, nil, fmt.Errorf("failed to deserialize first poly for soft voting: %v", err)
		}
		level := firstPoly.Level()

		soft = mkckks.NewCiphertext(params, idset, level, resultBytes.SoftVoting.Scale)
		for id, polyBytes := range resultBytes.SoftVoting.PolyBytes {
			poly := new(ring.Poly)
			if err := poly.UnmarshalBinary(polyBytes); err != nil {
				return nil, nil, nil, fmt.Errorf("failed to deserialize poly for soft voting id %s: %v", id, err)
			}
			soft.Value[id] = poly
		}
	}

	if len(resultBytes.LogitSoftVoting.PolyBytes) > 0 {
		idset := mkrlwe.NewIDSet()
		for _, id := range resultBytes.LogitSoftVoting.IDSet {
			idset.Add(id)
		}

		firstPoly := new(ring.Poly)
		if err := firstPoly.UnmarshalBinary(resultBytes.LogitSoftVoting.PolyBytes["0"]); err != nil {
			return nil, nil, nil, fmt.Errorf("failed to deserialize first poly for logit voting: %v", err)
		}
		level := firstPoly.Level()

		logit = mkckks.NewCiphertext(params, idset, level, resultBytes.LogitSoftVoting.Scale)
		for id, polyBytes := range resultBytes.LogitSoftVoting.PolyBytes {
			poly := new(ring.Poly)
			if err := poly.UnmarshalBinary(polyBytes); err != nil {
				return nil, nil, nil, fmt.Errorf("failed to deserialize poly for logit voting id %s: %v", id, err)
			}
			logit.Value[id] = poly
		}
	}

	if len(resultBytes.VarianceMaskVoting.PolyBytes) > 0 {
		idset := mkrlwe.NewIDSet()
		for _, id := range resultBytes.LogitSoftVoting.IDSet {
			idset.Add(id)
		}

		firstPoly := new(ring.Poly)
		if err := firstPoly.UnmarshalBinary(resultBytes.LogitSoftVoting.PolyBytes["0"]); err != nil {
			return nil, nil, nil, fmt.Errorf("failed to deserialize first poly for logit voting: %v", err)
		}
		level := firstPoly.Level()

		logit = mkckks.NewCiphertext(params, idset, level, resultBytes.LogitSoftVoting.Scale)
		for id, polyBytes := range resultBytes.LogitSoftVoting.PolyBytes {
			poly := new(ring.Poly)
			if err := poly.UnmarshalBinary(polyBytes); err != nil {
				return nil, nil, nil, fmt.Errorf("failed to deserialize poly for logit voting id %s: %v", id, err)
			}
			logit.Value[id] = poly
		}
	}

	return soft, logit, variance, nil
}
