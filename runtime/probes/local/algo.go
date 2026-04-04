package local

import (
	"context"
	"fmt"

	"github.com/vigo999/mindspore-cli/internal/train"
	"github.com/vigo999/mindspore-cli/runtime/probes"
)

// AlgoProbe checks local repository path, train script, and config hints.
// Phase 1: returns demo-backed results.
type AlgoProbe struct{}

func (p *AlgoProbe) Run(_ context.Context, req train.Request) ([]probes.Result, error) {
	// TODO: real implementation would check git status, script existence, etc.
	return []probes.Result{
		{
			Scope:    probes.ScopeLocal,
			Name:     "repo",
			Status:   probes.StatusPass,
			Summary:  fmt.Sprintf("~/qwen3_experiment | github.com/user/%s", req.Model),
			Critical: false,
		},
		{
			Scope:    probes.ScopeLocal,
			Name:     "model ckpt",
			Status:   probes.StatusPass,
			Summary:  fmt.Sprintf("~/qwen3_experiment/ckpt/%s-7b.ckpt (13.2 GB, sha256 verified)", req.Model),
			Critical: false,
		},
		{
			Scope:    probes.ScopeLocal,
			Name:     "train scripts",
			Status:   probes.StatusPass,
			Summary:  fmt.Sprintf("~/qwen3_experiment/train_%s.py", req.Method),
			Critical: false,
		},
	}, nil
}
