package project

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadWeeklyUpdateFromFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "weekly.md")
	content := `---
week: "2026-W10"
date: "2026-03-01"
metrics:
  milestones_done: 1
  milestones_total: 2
  progress_pct: 50
---

# Weekly Update
Body line.
`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	wu, err := LoadWeeklyUpdateFromFile(path)
	if err != nil {
		t.Fatalf("LoadWeeklyUpdateFromFile failed: %v", err)
	}
	if wu.Week != "2026-W10" || wu.Metrics.ProgressPct != 50 {
		t.Fatalf("unexpected weekly update: %+v", wu)
	}
	if wu.Body == "" {
		t.Fatalf("expected markdown body")
	}
}
