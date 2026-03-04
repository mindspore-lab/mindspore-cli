package trace

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestJSONLWriter_Write(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trace.jsonl")
	w, err := NewJSONLWriter(path)
	if err != nil {
		t.Fatalf("NewJSONLWriter failed: %v", err)
	}
	if err := w.Write("test_event", map[string]any{"ok": true}); err != nil {
		t.Fatalf("Write failed: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile failed: %v", err)
	}
	if !strings.Contains(string(data), `"type":"test_event"`) {
		t.Fatalf("unexpected trace output: %s", string(data))
	}
}
