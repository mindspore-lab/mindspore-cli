// Package local defines the interface for local-side readiness probes.
package local

import (
	"context"

	"github.com/vigo999/mindspore-cli/internal/train"
	"github.com/vigo999/mindspore-cli/runtime/probes"
)

// Probe checks local machine readiness before training.
type Probe interface {
	Run(ctx context.Context, req train.Request) ([]probes.Result, error)
}
