package target

import (
	"context"

	"github.com/vigo999/mindspore-cli/internal/train"
	"github.com/vigo999/mindspore-cli/runtime/probes"
)

// DeviceProbe checks accelerator device on the remote target.
// Phase 1: returns demo-backed results.
type DeviceProbe struct{}

func (p *DeviceProbe) Run(_ context.Context, target train.TrainTarget) ([]probes.Result, error) {
	return []probes.Result{
		{
			Scope:    probes.ScopeTarget,
			Name:     "device",
			Status:   probes.StatusPass,
			Summary:  "ascend 910B",
			Critical: true,
			Details: map[string]any{
				"host": target.Name,
			},
		},
	}, nil
}
