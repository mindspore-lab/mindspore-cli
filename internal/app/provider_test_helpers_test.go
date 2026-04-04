package app

import (
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/vigo999/mindspore-cli/integrations/llm"
)

type stubHTTPClient struct {
	do func(*http.Request) (*http.Response, error)
}

func (c stubHTTPClient) Do(req *http.Request) (*http.Response, error) {
	return c.do(req)
}

func newOpenAICompletionTestProvider(t *testing.T, cfg llm.ResolvedConfig, handle func(*http.Request)) llm.Provider {
	t.Helper()

	provider, err := llm.NewOpenAICompletionProviderWithHTTPClient(cfg, stubHTTPClient{
		do: func(req *http.Request) (*http.Response, error) {
			if handle != nil {
				handle(req)
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body: io.NopCloser(strings.NewReader(
					`{"id":"cmpl-test","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`,
				)),
			}, nil
		},
	})
	if err != nil {
		t.Fatalf("NewOpenAICompletionProviderWithHTTPClient() error = %v", err)
	}
	return provider
}
