package permission

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestFilePermissionStore_SaveUsesClaudeStyleSettingsShape(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "permissions.json")
	store, err := NewFilePermissionStore(path)
	if err != nil {
		t.Fatalf("NewFilePermissionStore() err = %v", err)
	}

	if err := store.SaveDecision(PermissionDecision{
		Tool:      "shell",
		Action:    "find:*",
		Level:     PermissionAllowSession,
		Timestamp: time.Now(),
	}); err != nil {
		t.Fatalf("SaveDecision() err = %v", err)
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile() err = %v", err)
	}
	text := string(raw)
	if !strings.Contains(text, `"permissions"`) || !strings.Contains(text, `"allow"`) {
		t.Fatalf("saved file = %s, want permissions.allow JSON object", text)
	}
	if strings.HasPrefix(strings.TrimSpace(text), "[") {
		t.Fatalf("saved file should not be legacy array: %s", text)
	}
}

func TestFilePermissionStore_LoadRejectsLegacyArray(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "permissions.json")
	legacy := `[
  {
    "Tool": "shell",
    "Action": "ls -la",
    "Path": "",
    "Level": 3,
    "Timestamp": "2026-03-21T15:19:57.660921+08:00"
  }
]`
	if err := os.WriteFile(path, []byte(legacy), 0644); err != nil {
		t.Fatalf("WriteFile() err = %v", err)
	}

	if _, err := NewFilePermissionStore(path); err == nil {
		t.Fatal("NewFilePermissionStore() err = nil, want format error")
	}
}

func TestFilePermissionStore_LoadRejectsLowercaseToolName(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "permissions.json")
	invalid := `{
  "permissions": {
    "allow": [
      "helloworld"
    ]
  }
}`
	if err := os.WriteFile(path, []byte(invalid), 0644); err != nil {
		t.Fatalf("WriteFile() err = %v", err)
	}

	_, err := NewFilePermissionStore(path)
	if err == nil {
		t.Fatal("NewFilePermissionStore() err = nil, want validation error")
	}
	if !strings.Contains(err.Error(), "permissions.allow[0]") {
		t.Fatalf("err = %v, want location hint permissions.allow[0]", err)
	}
}
