package mkrlwe

import (
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

type Decryptor struct {
	params  Parameters
	ringQ   *ring.Ring
	pool    *ring.Poly
	sk      *SecretKey
	sampler ring.Sampler
}

// NewDecryptor instantiates a new generic RLWE Decryptor.
func NewDecryptor(params Parameters) *Decryptor {

	return &Decryptor{
		params: params,
		ringQ:  params.RingQ(),
		pool:   params.RingQ().NewPoly(),
	}
}

func NewDecryptorWithGaussianNoise(params Parameters, sigma float64, bound int) *Decryptor {
	prng, _ := utils.NewPRNG()
	sampler := ring.NewGaussianSampler(prng, params.RingQ(), sigma, bound)
	return &Decryptor{
		params:  params,
		ringQ:   params.RingQ(),
		pool:    params.RingQ().NewPoly(),
		sampler: sampler,
	}
}

// PartialDecrypt partially decrypts the ct with single secretkey sk and update result inplace
func (decryptor *Decryptor) PartialDecrypt(ct *Ciphertext, sk *SecretKey) {
	ringQ := decryptor.ringQ
	id := sk.ID
	level := ct.Level()

	if !ct.Value[id].IsNTT {
		ringQ.NTTLvl(level, ct.Value[id], ct.Value[id])
	}

	ringQ.MulCoeffsMontgomeryLvl(level, ct.Value[id], sk.Value.Q, ct.Value[id])

	if !ct.Value[id].IsNTT {
		ringQ.InvNTTLvl(level, ct.Value[id], ct.Value[id])
	}

	ringQ.AddLvl(level, ct.Value["0"], ct.Value[id], ct.Value["0"])
	delete(ct.Value, id)
}

// GenShare computes noisy partial decryption share for one party
func (decryptor *Decryptor) GenShare(ct *Ciphertext, sk *SecretKey) *ring.Poly {
	ringQ := decryptor.ringQ
	level := ct.Level()

	tmp := ringQ.NewPoly()
	if ct.Value[sk.ID].IsNTT {
		ringQ.MulCoeffsMontgomeryLvl(level, ct.Value[sk.ID], sk.Value.Q, tmp)
	} else {
		ringQ.NTTLvl(level, ct.Value[sk.ID], tmp)
		ringQ.MulCoeffsMontgomeryLvl(level, tmp, sk.Value.Q, tmp)
		ringQ.InvNTTLvl(level, tmp, tmp)
	}

	noise := ringQ.NewPoly()
	decryptor.sampler.Read(noise)

	if ct.Value[sk.ID].IsNTT {
		ringQ.NTTLvl(level, noise, noise)
	}

	ringQ.AddLvl(level, tmp, noise, tmp)

	return tmp
}

func (decryptor *Decryptor) AggregateSharesAndDrop(ct *Ciphertext, shares map[string]*ring.Poly) {
	ringQ := decryptor.ringQ
	level := ct.Level()

	agg := ringQ.NewPoly()
	for _, sh := range shares {
		ringQ.AddLvl(level, agg, sh, agg)
	}
	// add aggregated share to c0
	ringQ.AddLvl(level, ct.Value["0"], agg, ct.Value["0"])

	// drop only the terms for which we added c_i*s_i (i.e., keys in shares)
	for id := range shares {
		if id != "0" {
			delete(ct.Value, id)
		}
	}
}

// Decrypt decrypts the ciphertext with given secretkey set and write the result in ptOut.
// The level of the output plaintext is min(ciphertext.Level(), plaintext.Level())
// Output domain will match plaintext.Value.IsNTT value.
func (decryptor *Decryptor) Decrypt(ciphertext *Ciphertext, skSet *SecretKeySet, plaintext *rlwe.Plaintext) {
	ringQ := decryptor.ringQ
	level := utils.MinInt(ciphertext.Level(), plaintext.Level())
	plaintext.Value.Coeffs = plaintext.Value.Coeffs[:level+1]

	ctTmp := ciphertext.CopyNew()
	idset := ctTmp.IDSet()
	for _, sk := range skSet.Value {
		if idset.Has(sk.ID) {
			decryptor.PartialDecrypt(ctTmp, sk)
		}
	}

	if len(ctTmp.Value) > 1 {
		panic("Cannot Decrypt: there is a missing secretkey")
	}

	ringQ.ReduceLvl(level, ctTmp.Value["0"], plaintext.Value)
}
