package llm

import (
	"context"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type debugHTTPClient struct {
	do func(*http.Request) (*http.Response, error)
}

func (c debugHTTPClient) Do(req *http.Request) (*http.Response, error) {
	return c.do(req)
}

func TestDoJSONDebugDumpsRequestAndResponse(t *testing.T) {
	t.Helper()

	var sentBody string
	client := debugHTTPClient{
		do: func(req *http.Request) (*http.Response, error) {
			body, err := io.ReadAll(req.Body)
			if err != nil {
				t.Fatalf("read request body: %v", err)
			}
			sentBody = string(body)

			return &http.Response{
				Status:        "200 OK",
				StatusCode:    http.StatusOK,
				Proto:         "HTTP/1.1",
				ProtoMajor:    1,
				ProtoMinor:    1,
				Header:        http.Header{"Content-Type": []string{"application/json"}},
				Body:          io.NopCloser(strings.NewReader(`{"ok":true}`)),
				ContentLength: int64(len(`{"ok":true}`)),
			}, nil
		},
	}

	debugDir := t.TempDir()
	ctx := WithDebugDumper(context.Background(), NewDebugDumper(debugDir))
	resp, err := DoJSON(ctx, client, http.MethodPost, "https://example.com/v1/responses", map[string]string{
		"Authorization": "Bearer test-key",
	}, map[string]string{"hello": "world"})
	if err != nil {
		t.Fatalf("DoJSON() error = %v", err)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read response body: %v", err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("close response body: %v", err)
	}

	if sentBody != `{"hello":"world"}` {
		t.Fatalf("sent request body = %q, want %q", sentBody, `{"hello":"world"}`)
	}
	if string(body) != `{"ok":true}` {
		t.Fatalf("response body = %q, want %q", string(body), `{"ok":true}`)
	}

	requestDump, responseDump := readDebugDumpPair(t, debugDir)
	if !strings.Contains(requestDump, "POST /v1/responses HTTP/1.1") {
		t.Fatalf("request dump missing request line: %q", requestDump)
	}
	if !strings.Contains(requestDump, `"hello":"world"`) {
		t.Fatalf("request dump missing body: %q", requestDump)
	}
	if !strings.Contains(responseDump, "200 OK") {
		t.Fatalf("response dump missing status line: %q", responseDump)
	}
	if !strings.Contains(responseDump, `{"ok":true}`) {
		t.Fatalf("response dump missing body: %q", responseDump)
	}
}

func TestDoJSONDebugWritesErrorDumpOnRequestFailure(t *testing.T) {
	debugDir := t.TempDir()
	ctx := WithDebugDumper(context.Background(), NewDebugDumper(debugDir))
	wantErr := errors.New("boom")

	_, err := DoJSON(ctx, debugHTTPClient{
		do: func(req *http.Request) (*http.Response, error) {
			return nil, wantErr
		},
	}, http.MethodPost, "https://example.com/v1/responses", nil, map[string]string{"hello": "world"})
	if err == nil {
		t.Fatal("DoJSON() error = nil, want error")
	}
	if !strings.Contains(err.Error(), "boom") {
		t.Fatalf("DoJSON() error = %q, want boom", err)
	}

	_, responseDump := readDebugDumpPair(t, debugDir)
	if !strings.Contains(responseDump, "request failed: boom") {
		t.Fatalf("response dump = %q, want request failure", responseDump)
	}
}

func readDebugDumpPair(t *testing.T, dir string) (string, string) {
	t.Helper()

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read debug dir: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("debug file count = %d, want 2", len(entries))
	}

	var requestPath string
	var responsePath string
	for _, entry := range entries {
		name := entry.Name()
		switch {
		case strings.HasSuffix(name, ".request.http"):
			requestPath = filepath.Join(dir, name)
		case strings.HasSuffix(name, ".response.http"):
			responsePath = filepath.Join(dir, name)
		}
	}
	if requestPath == "" || responsePath == "" {
		t.Fatalf("missing request/response dump files in %q", dir)
	}

	requestDump, err := os.ReadFile(requestPath)
	if err != nil {
		t.Fatalf("read request dump: %v", err)
	}
	responseDump, err := os.ReadFile(responsePath)
	if err != nil {
		t.Fatalf("read response dump: %v", err)
	}
	return string(requestDump), string(responseDump)
}
