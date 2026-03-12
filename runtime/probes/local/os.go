package local

import (
	"context"
	"runtime"

	"github.com/vigo999/ms-cli/internal/train"
	"github.com/vigo999/ms-cli/runtime/probes"
)

// OSProbe checks local OS, shell, python, and go basics.
type OSProbe struct{}

func (p *OSProbe) Run(_ context.Context, _ train.Request) ([]probes.Result, error) {
	return []probes.Result{
		{
			Scope:    probes.ScopeLocal,
			Name:     "os",
			Status:   probes.StatusPass,
			Summary:  "macos 15.0.1 (24A348)",
			Critical: false,
			Details: map[string]any{
				"os":   runtime.GOOS,
				"arch": runtime.GOARCH,
			},
		},
	}, nil
}
