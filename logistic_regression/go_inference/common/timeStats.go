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
	ts.calculateStats()
}

func (ts *TimingStats) calculateStats() {
	if len(ts.Samples) == 0 {
		return
	}

	var sum time.Duration
	for _, d := range ts.Samples {
		sum += d
	}
	ts.Mean = sum / time.Duration(len(ts.Samples))

	var sqSum float64
	mean := float64(ts.Mean.Nanoseconds())
	for _, d := range ts.Samples {
		diff := float64(d.Nanoseconds()) - mean
		sqSum += diff * diff
	}
	stdDev := math.Sqrt(sqSum / float64(len(ts.Samples)))
	ts.StdDev = time.Duration(stdDev)
}
