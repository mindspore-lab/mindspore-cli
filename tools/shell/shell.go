package shell

import (
	"context"
	"time"

	"github.com/vigo999/ms-cli/executor"
)

// Tool wraps shell execution for runtime.
type Tool struct {
	runner *executor.BashRunner
}

func NewTool(workDir string, timeout time.Duration) *Tool {
	return &Tool{
		runner: executor.NewBashRunner(workDir, timeout),
	}
}

// Run executes a shell command via bash -lc.
func (t *Tool) Run(ctx context.Context, command string) (string, int, error) {
	res := t.runner.Run(ctx, command)
	return res.Output, res.ExitCode, res.Err
}
