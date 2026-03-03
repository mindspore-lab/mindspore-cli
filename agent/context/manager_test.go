package context

import (
	"strings"
	"testing"
)

func TestManagerDedupAdjacentEntries(t *testing.T) {
	m := NewManager(2000, 0.8, 10)
	m.Add("shell", "cmd: ls -la")
	m.Add("shell", "cmd: ls -la")
	m.Add("read", "main.go")

	out := m.Render()
	if strings.Count(out, "cmd: ls -la") != 1 {
		t.Fatalf("expected adjacent duplicate to be compacted, got: %q", out)
	}
}

func TestManagerCompactsByBudget(t *testing.T) {
	m := NewManager(60, 0.75, 50)
	for i := 0; i < 20; i++ {
		m.Add("grep", strings.Repeat("very-long-line-", 20))
	}

	out := m.Render()
	if !strings.Contains(out, "[context compacted]") {
		t.Fatalf("expected context compaction marker, got: %q", out)
	}
}
