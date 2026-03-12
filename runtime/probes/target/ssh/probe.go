// Package ssh provides the SSH connectivity probe for remote targets.
package ssh

import (
	"context"
	"fmt"

	"github.com/vigo999/ms-cli/internal/train"
	"github.com/vigo999/ms-cli/runtime/probes"
)

// Probe checks SSH connectivity to the remote target.
// Phase 1: returns demo-backed results.
type Probe struct{}

func (p *Probe) Run(_ context.Context, target train.TrainTarget) ([]probes.Result, error) {
	// TODO: real implementation would attempt SSH connection
	host := target.Name
	addr, _ := target.Config["address"].(string)
	if addr == "" {
		addr = "unknown"
	}
	if flaky, _ := target.Config["demo_ssh_flaky"].(bool); flaky {
		return []probes.Result{
			{
				Scope:    probes.ScopeTarget,
				Name:     "ssh",
				Status:   probes.StatusFail,
				Summary:  fmt.Sprintf("%s (%s) key auth timed out during first probe", host, addr),
				Critical: true,
				Details: map[string]any{
					"host":    host,
					"address": addr,
					"auth":    "key",
				},
			},
		}, nil
	}

	return []probes.Result{
		{
			Scope:    probes.ScopeTarget,
			Name:     "ssh",
			Status:   probes.StatusPass,
			Summary:  fmt.Sprintf("weizheng@%s", addr),
			Critical: true,
			Details: map[string]any{
				"host":    host,
				"address": addr,
				"auth":    "key",
			},
		},
	}, nil
}
