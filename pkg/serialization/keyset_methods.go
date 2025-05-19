package serialization

import (
	"bytes"
	"encoding/gob"
	"fmt"

	"github.com/anony-submit/snu-mghe/mkrlwe"
)

// PublicKeyBytes represents the serialized form of a PublicKey
type PublicKeyBytes struct {
	Value []byte // serialized PolyQP using WriteTo
	ID    string
}

// SwitchingKeyBytes represents the serialized form of a SwitchingKey
type SwitchingKeyBytes struct {
	Value []byte
}

// RelinearizationKeyBytes represents the serialized form of a RelinearizationKey
type RelinearizationKeyBytes struct {
	Value [3][]byte // serialized SwitchingKey for each component
	ID    string
}

// RotationKeyBytes represents the serialized form of a RotationKey
type RotationKeyBytes struct {
	Value  []byte // serialized SwitchingKey
	ID     string
	RotIdx uint
}

// SerializePublicKey serializes a PublicKey
func SerializePublicKey(pk *mkrlwe.PublicKey) ([]byte, error) {
	// Allocate buffer with the required size
	dataLen := pk.Value[0].GetDataLen(true) + pk.Value[1].GetDataLen(true)
	data := make([]byte, dataLen)

	// Write both PolyQP values
	pt := 0
	for i := 0; i < 2; i++ {
		inc, err := pk.Value[i].WriteTo(data[pt:])
		if err != nil {
			return nil, fmt.Errorf("failed to serialize value %d: %v", i, err)
		}
		pt += inc
	}

	pkBytes := &PublicKeyBytes{
		Value: data,
		ID:    pk.ID,
	}

	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(pkBytes); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// DeserializePublicKey deserializes a PublicKey
func DeserializePublicKey(data []byte, params mkrlwe.Parameters) (*mkrlwe.PublicKey, error) {
	var pkBytes PublicKeyBytes
	if err := gob.NewDecoder(bytes.NewBuffer(data)).Decode(&pkBytes); err != nil {
		return nil, err
	}

	pk := mkrlwe.NewPublicKey(params, pkBytes.ID)
	pt := 0
	for i := 0; i < 2; i++ {
		inc, err := pk.Value[i].DecodePolyNew(pkBytes.Value[pt:])
		if err != nil {
			return nil, fmt.Errorf("failed to deserialize value %d: %v", i, err)
		}
		pt += inc
	}

	return pk, nil
}

// SerializeRelinearizationKey serializes a RelinearizationKey
func SerializeRelinearizationKey(rlk *mkrlwe.RelinearizationKey) ([]byte, error) {
	dataLen := 0
	for i := 0; i < 3; i++ {
		beta := len(rlk.Value[i].Value)
		for j := 0; j < beta; j++ {
			dataLen += rlk.Value[i].Value[j].GetDataLen(true)
		}
	}

	rlkBytes := &RelinearizationKeyBytes{
		ID:    rlk.ID,
		Value: [3][]byte{},
	}

	// Serialize each switching key component
	for i := 0; i < 3; i++ {
		beta := len(rlk.Value[i].Value)
		componentData := make([]byte, 0)
		pt := 0

		for j := 0; j < beta; j++ {
			data := make([]byte, rlk.Value[i].Value[j].GetDataLen(true))
			inc, err := rlk.Value[i].Value[j].WriteTo(data)
			if err != nil {
				return nil, fmt.Errorf("failed to serialize value[%d][%d]: %v", i, j, err)
			}
			componentData = append(componentData, data[:inc]...)
			pt += inc
		}
		rlkBytes.Value[i] = componentData
	}

	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(rlkBytes); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// DeserializeRelinearizationKey deserializes a RelinearizationKey
func DeserializeRelinearizationKey(data []byte, params mkrlwe.Parameters) (*mkrlwe.RelinearizationKey, error) {
	var rlkBytes RelinearizationKeyBytes
	if err := gob.NewDecoder(bytes.NewBuffer(data)).Decode(&rlkBytes); err != nil {
		return nil, err
	}

	rlk := mkrlwe.NewRelinearizationKey(params, rlkBytes.ID)
	for i := 0; i < 3; i++ {
		pt := 0
		beta := len(rlk.Value[i].Value)

		for j := 0; j < beta; j++ {
			inc, err := rlk.Value[i].Value[j].DecodePolyNew(rlkBytes.Value[i][pt:])
			if err != nil {
				return nil, fmt.Errorf("failed to deserialize value[%d][%d]: %v", i, j, err)
			}
			pt += inc
		}
	}

	return rlk, nil
}

// SerializeRotationKey serializes a RotationKey
func SerializeRotationKey(rtk *mkrlwe.RotationKey) ([]byte, error) {
	dataLen := 0
	beta := len(rtk.Value.Value)
	for i := 0; i < beta; i++ {
		dataLen += rtk.Value.Value[i].GetDataLen(true)
	}

	data := make([]byte, dataLen)
	pt := 0

	// Serialize switching key
	for i := 0; i < beta; i++ {
		inc, err := rtk.Value.Value[i].WriteTo(data[pt:])
		if err != nil {
			return nil, fmt.Errorf("failed to serialize value[%d]: %v", i, err)
		}
		pt += inc
	}

	rtkBytes := &RotationKeyBytes{
		Value:  data,
		ID:     rtk.ID,
		RotIdx: rtk.RotIdx,
	}

	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(rtkBytes); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// DeserializeRotationKey deserializes a RotationKey
func DeserializeRotationKey(data []byte, params mkrlwe.Parameters) (*mkrlwe.RotationKey, error) {
	var rtkBytes RotationKeyBytes
	if err := gob.NewDecoder(bytes.NewBuffer(data)).Decode(&rtkBytes); err != nil {
		return nil, err
	}

	rtk := mkrlwe.NewRotationKey(params, rtkBytes.RotIdx, rtkBytes.ID)
	pt := 0
	beta := len(rtk.Value.Value)

	for i := 0; i < beta; i++ {
		inc, err := rtk.Value.Value[i].DecodePolyNew(rtkBytes.Value[pt:])
		if err != nil {
			return nil, fmt.Errorf("failed to deserialize value[%d]: %v", i, err)
		}
		pt += inc
	}

	return rtk, nil
}

// Helper functions remain the same
func AddPublicKeyFromBytes(ks *mkrlwe.PublicKeySet, data []byte, params mkrlwe.Parameters) error {
	pk, err := DeserializePublicKey(data, params)
	if err != nil {
		return err
	}
	ks.AddPublicKey(pk)
	return nil
}

func AddRelinKeyFromBytes(ks *mkrlwe.RelinearizationKeySet, data []byte, params mkrlwe.Parameters) error {
	rlk, err := DeserializeRelinearizationKey(data, params)
	if err != nil {
		return err
	}
	ks.AddRelinearizationKey(rlk)
	return nil
}

func AddRotationKeyFromBytes(ks *mkrlwe.RotationKeySet, data []byte, params mkrlwe.Parameters) error {
	rtk, err := DeserializeRotationKey(data, params)
	if err != nil {
		return err
	}
	ks.AddRotationKey(rtk)
	return nil
}
