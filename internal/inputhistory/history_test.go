package inputhistory

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestLoadWorkdirReturnsOldestFirstRecentWindow(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	for i := 0; i < HistoryLoadMax+5; i++ {
		if err := Append("/repo/a", "prompt-"+strconv.Itoa(i)); err != nil {
			t.Fatalf("append history: %v", err)
		}
	}
	if err := Append("/repo/b", "other-workdir"); err != nil {
		t.Fatalf("append history for other workdir: %v", err)
	}

	got, err := LoadWorkdir("/repo/a")
	if err != nil {
		t.Fatalf("load history: %v", err)
	}
	if len(got) != HistoryLoadMax {
		t.Fatalf("expected %d entries, got %d", HistoryLoadMax, len(got))
	}
	if got[0] != "prompt-5" {
		t.Fatalf("expected oldest kept entry prompt-5, got %q", got[0])
	}
	if got[len(got)-1] != "prompt-"+strconv.Itoa(HistoryLoadMax+4) {
		t.Fatalf("expected newest entry prompt-104, got %q", got[len(got)-1])
	}
}

func TestAppendTrimKeepsNewestTail(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	chunk := strings.Repeat("x", int(HistoryMaxBytes/4))
	for i := 0; i < 6; i++ {
		if err := Append("/repo/a", chunk+"-"+strconv.Itoa(i)); err != nil {
			t.Fatalf("append history %d: %v", i, err)
		}
	}

	path, err := FilePath()
	if err != nil {
		t.Fatalf("history path: %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat history: %v", err)
	}
	if info.Size() > HistoryMaxBytes {
		t.Fatalf("expected trimmed history at or below %d bytes, got %d", HistoryMaxBytes, info.Size())
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read trimmed history: %v", err)
	}
	content := string(data)
	if !strings.Contains(content, chunk+"-5") {
		t.Fatal("expected newest entry to remain after trim")
	}
	if strings.Contains(content, chunk+"-0") {
		t.Fatal("expected oldest entry to be trimmed")
	}
}

func TestLoadWorkdirSkipsMalformedLines(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	path, err := FilePath()
	if err != nil {
		t.Fatalf("history path: %v", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir history dir: %v", err)
	}
	content := "{bad json}\n" +
		"{\"display\":\"alpha\",\"workdir\":\"/repo/a\",\"timestamp\":1}\n" +
		"{\"display\":\"beta\",\"workdir\":\"/repo/b\",\"timestamp\":2}\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write history file: %v", err)
	}

	got, err := LoadWorkdir("/repo/a")
	if err != nil {
		t.Fatalf("load history: %v", err)
	}
	if len(got) != 1 || got[0] != "alpha" {
		t.Fatalf("expected only alpha for /repo/a, got %#v", got)
	}
}
