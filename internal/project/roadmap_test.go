package project

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLoadRoadmapAndComputeStatus(t *testing.T) {
	path := filepath.Join(t.TempDir(), "roadmap.yaml")
	content := `
version: 1
target_date: "2026-06-30"
phases:
  - id: p1
    name: Foundation
    start: "2026-03-01"
    end: "2026-03-31"
    milestones:
      - id: m1
        title: setup
        status: done
      - id: m2
        title: polish
        status: in_progress
`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	rm, err := LoadRoadmapFromFile(path)
	if err != nil {
		t.Fatalf("LoadRoadmapFromFile failed: %v", err)
	}
	status, err := ComputeRoadmapStatus(rm, time.Date(2026, 3, 1, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatalf("ComputeRoadmapStatus failed: %v", err)
	}
	if status.Overall.Total != 2 || status.Overall.Done != 1 || status.Overall.Pct != 50 {
		t.Fatalf("unexpected overall progress: %+v", status.Overall)
	}
	if len(status.Phases) != 1 || status.Phases[0].InProg != 1 {
		t.Fatalf("unexpected phase status: %+v", status.Phases)
	}
}
