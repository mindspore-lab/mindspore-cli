package executor

import (
	"github.com/vigo999/ms-cli/agent/loop"
)

// Run executes a task using the real engine.
// This function is called by the engine itself in real mode.
func Run(task loop.Task) string {
	// In real mode, this is handled by the Engine.Run method
	// This function is kept for backward compatibility with the old executor interface
	return "Task submitted to engine: " + task.Description
}
