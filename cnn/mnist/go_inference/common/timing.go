package common

import (
	"math"
	"time"
)

type TimingStats struct {
	Mean    time.Duration
	StdDev  time.Duration
	Samples []time.Duration
}

func (ts *TimingStats) AddSample(duration time.Duration) {
	ts.Samples = append(ts.Samples, duration)
	ts.updateStats()
}

func (ts *TimingStats) updateStats() {
	var sum time.Duration
	for _, sample := range ts.Samples {
		sum += sample
	}
	ts.Mean = sum / time.Duration(len(ts.Samples))

	var sumSquaredDiff float64
	for _, sample := range ts.Samples {
		diff := float64(sample - ts.Mean)
		sumSquaredDiff += diff * diff
	}
	stdDev := math.Sqrt(sumSquaredDiff / float64(len(ts.Samples)))
	ts.StdDev = time.Duration(stdDev)
}

type TimingInfo struct {
	// Client timings
	ClientKeyGenStats    TimingStats
	DataEncryptionStats  TimingStats
	FinalDecryptionStats TimingStats
	TotalDecryptionStats TimingStats

	// Data owner statistics
	DataOwnerKeyGenStats  TimingStats
	ModelEncryptionStats  TimingStats
	PartialDecryptionTime time.Duration

	// CSP timings
	InferenceStats        TimingStats
	EnsembleStats         TimingStats
	TotalComputeStats     TimingStats
	ClientTransferStats   TimingStats
	DataOwnerTransferTime time.Duration
}
