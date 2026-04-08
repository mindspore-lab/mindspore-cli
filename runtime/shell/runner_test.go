package shell

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestRunnerRunStream_EmitsOutputWhileCollectingResult(t *testing.T) {
	runner := NewRunner(Config{
		WorkDir: ".",
		Timeout: 2 * time.Second,
	})

	var (
		mu     sync.Mutex
		chunks []OutputChunk
	)
	result, err := runner.RunStream(context.Background(), "printf 'out-1\\nout-2\\n'; printf 'err-1\\n' >&2", func(chunk OutputChunk) {
		mu.Lock()
		chunks = append(chunks, chunk)
		mu.Unlock()
	})
	if err != nil {
		t.Fatalf("RunStream failed: %v", err)
	}

	if got, want := result.Stdout, "out-1\nout-2"; got != want {
		t.Fatalf("stdout = %q, want %q", got, want)
	}
	if got, want := result.Stderr, "err-1"; got != want {
		t.Fatalf("stderr = %q, want %q", got, want)
	}

	var sawStdout, sawStderr bool
	for _, chunk := range chunks {
		if chunk.Stream == StreamStdout && strings.Contains(chunk.Text, "out-1") {
			sawStdout = true
		}
		if chunk.Stream == StreamStderr && strings.Contains(chunk.Text, "err-1") {
			sawStderr = true
		}
	}
	if !sawStdout {
		t.Fatalf("expected streamed stdout chunk, got %#v", chunks)
	}
	if !sawStderr {
		t.Fatalf("expected streamed stderr chunk, got %#v", chunks)
	}
}

func TestReadCapped_KeepsMostRecentOutputWhenTruncated(t *testing.T) {
	input := strings.NewReader("line-1\nline-2\nline-3\nline-4\n")

	got, err := readCapped(input, len("line-3\nline-4"), nil)
	if err != nil {
		t.Fatalf("readCapped failed: %v", err)
	}

	if !strings.Contains(got, "line-3") || !strings.Contains(got, "line-4") {
		t.Fatalf("expected recent lines kept, got:\n%s", got)
	}
	if strings.Contains(got, "line-1") || strings.Contains(got, "line-2") {
		t.Fatalf("expected old lines dropped from truncated output, got:\n%s", got)
	}
	if !strings.Contains(got, "[output truncated]") {
		t.Fatalf("expected truncated marker, got:\n%s", got)
	}
}

func TestReadCapped_ContinuesEmittingAfterWindowExceeded(t *testing.T) {
	input := strings.NewReader("line-1\nline-2\nline-3\nline-4\n")

	var emitted []string
	got, err := readCapped(input, len("line-3\nline-4"), func(line string) {
		emitted = append(emitted, line)
	})
	if err != nil && err != io.EOF {
		t.Fatalf("readCapped failed: %v", err)
	}

	if !strings.Contains(got, "line-4") {
		t.Fatalf("expected final output window to include latest line, got:\n%s", got)
	}
	if len(emitted) < 4 {
		t.Fatalf("expected all lines to be emitted despite truncation, got %#v", emitted)
	}
	if gotLast := emitted[len(emitted)-1]; gotLast != "line-4" {
		t.Fatalf("last emitted line = %q, want line-4", gotLast)
	}
}

func TestRunnerRunStream_CancelStopsForegroundChildProcess(t *testing.T) {
	runner := NewRunner(Config{
		WorkDir: ".",
		Timeout: 5 * time.Second,
	})

	outfile := filepath.Join(t.TempDir(), "survived.txt")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	done := make(chan struct{})
	var (
		result *Result
		err    error
	)
	go func() {
		result, err = runner.RunStream(ctx, "printf 'ready\\n'; sh -c 'trap \"\" TERM; sleep 1; echo survived > "+outfile+"'", func(chunk OutputChunk) {
			if chunk.Stream == StreamStdout && strings.Contains(chunk.Text, "ready") {
				cancel()
			}
		})
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for RunStream to return after cancellation")
	}
	if err != nil {
		t.Fatalf("RunStream failed: %v", err)
	}
	if result == nil {
		t.Fatal("expected non-nil result")
	}

	time.Sleep(1500 * time.Millisecond)
	data, readErr := os.ReadFile(outfile)
	if readErr == nil && strings.Contains(string(data), "survived") {
		t.Fatalf("expected canceled child process to stop before writing marker, got %q", string(data))
	}
	if readErr != nil && !os.IsNotExist(readErr) {
		t.Fatalf("read outfile: %v", readErr)
	}
}
