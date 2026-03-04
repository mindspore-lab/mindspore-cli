package shell

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestToolRun(t *testing.T) {
	tool := NewTool(".", 2*time.Second)
	out, code, err := tool.Run(context.Background(), "echo ms-cli")
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if code != 0 {
		t.Fatalf("unexpected exit code: %d", code)
	}
	if !strings.Contains(out, "ms-cli") {
		t.Fatalf("unexpected output: %q", out)
	}
}
