// Package openai provides a compatibility shim over the unified provider implementation.
package openai

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/vigo999/ms-cli/integrations/llm"
	providerpkg "github.com/vigo999/ms-cli/integrations/llm/provider"
)

const (
	defaultEndpoint = "https://api.openai.com/v1"
	defaultTimeout  = 180 * time.Second
)

// Config holds the OpenAI client configuration.
type Config struct {
	Key        string
	URL        string
	Model      string
	Timeout    time.Duration
	HTTPClient *http.Client
}

// Client implements the llm.Provider interface for OpenAI.
type Client struct {
	provider llm.Provider
}

// NewClient creates a new OpenAI client.
func NewClient(cfg Config) (*Client, error) {
	apiKey := strings.TrimSpace(cfg.Key)
	if apiKey == "" {
		return nil, fmt.Errorf("key is required")
	}

	endpoint := strings.TrimSpace(cfg.URL)
	if endpoint == "" {
		endpoint = defaultEndpoint
	}

	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = defaultTimeout
	}

	provider, err := providerpkg.NewOpenAIProviderWithHTTPClient(providerpkg.ResolvedConfig{
		Kind:           providerpkg.ProviderOpenAI,
		Model:          strings.TrimSpace(cfg.Model),
		BaseURL:        endpoint,
		APIKey:         apiKey,
		AuthHeaderName: "Authorization",
		Headers: map[string]string{
			"Authorization": "Bearer " + apiKey,
		},
		Timeout: timeout,
	}, cfg.HTTPClient)
	if err != nil {
		return nil, err
	}

	return &Client{provider: provider}, nil
}

// Name returns the provider name.
func (c *Client) Name() string {
	return c.provider.Name()
}

// SupportsTools returns whether the provider supports tool calls.
func (c *Client) SupportsTools() bool {
	return c.provider.SupportsTools()
}

// Complete performs a non-streaming completion request.
func (c *Client) Complete(ctx context.Context, req *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	return c.provider.Complete(ctx, req)
}

// CompleteStream performs a streaming completion request.
func (c *Client) CompleteStream(ctx context.Context, req *llm.CompletionRequest) (llm.StreamIterator, error) {
	return c.provider.CompleteStream(ctx, req)
}

// AvailableModels returns the list of available models.
func (c *Client) AvailableModels() []llm.ModelInfo {
	return c.provider.AvailableModels()
}
