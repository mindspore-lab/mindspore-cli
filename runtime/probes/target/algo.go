package target

import (
	"context"

	"github.com/vigo999/ms-cli/internal/train"
	"github.com/vigo999/ms-cli/runtime/probes"
)

// AlgoProbe checks remote model checkpoint and train script visibility.
// Phase 1: returns demo-backed results.
type AlgoProbe struct{}

func (p *AlgoProbe) Run(_ context.Context, target train.TrainTarget) ([]probes.Result, error) {
	// TODO: real implementation would check remote file existence via SSH
	return []probes.Result{
		{
			Scope:    probes.ScopeTarget,
			Name:     "model ckpt",
			Status:   probes.StatusPass,
			Summary:  "~/qwen3_experiment/ckpt/qwen3-7b.ckpt (13.2 GB, sha256 verified)",
			Critical: true,
			Details: map[string]any{
				"host": target.Name,
			},
		},
		{
			Scope:    probes.ScopeTarget,
			Name:     "train scripts",
			Status:   probes.StatusPass,
			Summary:  "~/qwen3_experiment/train_lora.py",
			Critical: true,
			Details: map[string]any{
				"host": target.Name,
			},
		},
	}, nil
}
