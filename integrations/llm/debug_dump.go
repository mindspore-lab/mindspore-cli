package llm

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"
)

type debugContextKey string

const debugDumperKey debugContextKey = "llm_debug_dumper"

// DebugDumper persists raw HTTP request/response payloads for LLM calls.
type DebugDumper struct {
	dir string
	seq atomic.Uint64
}

// NewDebugDumper creates a dumper rooted at the supplied directory.
func NewDebugDumper(dir string) *DebugDumper {
	trimmed := strings.TrimSpace(dir)
	if trimmed == "" {
		return nil
	}
	return &DebugDumper{dir: trimmed}
}

// WithDebugDumper annotates a request context so DoJSON emits raw dumps.
func WithDebugDumper(ctx context.Context, dumper *DebugDumper) context.Context {
	if ctx == nil || dumper == nil {
		return ctx
	}
	return context.WithValue(ctx, debugDumperKey, dumper)
}

func debugDumperFromContext(ctx context.Context) *DebugDumper {
	if ctx == nil {
		return nil
	}
	dumper, _ := ctx.Value(debugDumperKey).(*DebugDumper)
	return dumper
}

func (d *DebugDumper) dumpRequest(req *http.Request) (string, string, error) {
	base, err := d.nextBasePath()
	if err != nil {
		return "", "", err
	}

	requestPath := base + ".request.http"
	responsePath := base + ".response.http"

	data, err := httputil.DumpRequestOut(req, true)
	if err != nil {
		return "", "", fmt.Errorf("dump request: %w", err)
	}
	if err := os.WriteFile(requestPath, data, 0o600); err != nil {
		return "", "", fmt.Errorf("write request dump: %w", err)
	}

	return requestPath, responsePath, nil
}

func (d *DebugDumper) wrapResponse(resp *http.Response, responsePath string) (*http.Response, error) {
	if resp == nil {
		return nil, fmt.Errorf("response is nil")
	}

	file, err := os.OpenFile(responsePath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return nil, fmt.Errorf("open response dump: %w", err)
	}

	headerDump, err := httputil.DumpResponse(resp, false)
	if err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("dump response headers: %w", err)
	}
	if _, err := file.Write(headerDump); err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("write response headers: %w", err)
	}

	if resp.Body == nil {
		if err := file.Close(); err != nil {
			return nil, fmt.Errorf("close response dump: %w", err)
		}
		return resp, nil
	}

	resp.Body = &teeReadCloser{
		reader: io.TeeReader(resp.Body, file),
		body:   resp.Body,
		extra:  file,
	}
	return resp, nil
}

func (d *DebugDumper) writeError(responsePath string, err error) error {
	if strings.TrimSpace(responsePath) == "" || err == nil {
		return nil
	}
	content := []byte(fmt.Sprintf("request failed: %v\n", err))
	if writeErr := os.WriteFile(responsePath, content, 0o600); writeErr != nil {
		return fmt.Errorf("write response dump: %w", writeErr)
	}
	return nil
}

func (d *DebugDumper) nextBasePath() (string, error) {
	if d == nil {
		return "", fmt.Errorf("debug dumper is nil")
	}
	if err := os.MkdirAll(d.dir, 0o755); err != nil {
		return "", fmt.Errorf("create debug dump dir: %w", err)
	}

	seq := d.seq.Add(1)
	stamp := time.Now().UTC().Format("20060102-150405-000000000")
	name := fmt.Sprintf("llm_%s_%04d", stamp, seq)
	return filepath.Join(d.dir, name), nil
}

type teeReadCloser struct {
	reader io.Reader
	body   io.Closer
	extra  io.Closer
}

func (t *teeReadCloser) Read(p []byte) (int, error) {
	return t.reader.Read(p)
}

func (t *teeReadCloser) Close() error {
	var firstErr error
	if t.body != nil {
		if err := t.body.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if t.extra != nil {
		if err := t.extra.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}
