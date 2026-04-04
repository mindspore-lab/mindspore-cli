package local

import (
	"context"

	"github.com/vigo999/mindspore-cli/internal/train"
	"github.com/vigo999/mindspore-cli/runtime/probes"
)

// AIFrameworkProbe checks local AI framework presence (informational).
// Phase 1: returns demo-backed results.
type AIFrameworkProbe struct{}

func (p *AIFrameworkProbe) Run(_ context.Context, _ train.Request) ([]probes.Result, error) {
	// TODO: real implementation would check torch, mindspore, transformers, etc.
	return []probes.Result{
		{
			Scope:    probes.ScopeLocal,
			Name:     "libs",
			Status:   probes.StatusPass,
			Summary:  "torch 2.7 | mindspore 2.8 | transformers v5.0.1 | diffusers v0.36",
			Critical: false,
			Details: map[string]any{
				"torch":        "2.7",
				"mindspore":    "2.8",
				"transformers": "5.0.1",
				"diffusers":    "0.36",
			},
		},
	}, nil
}
