package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/vigo999/ms-cli/integrations/llm"
)

const anthropicDefaultTimeout = 180 * time.Second

type anthropicClient struct {
	name       string
	baseURL    string
	model      string
	headers    map[string]string
	httpClient HTTPClient
	codec      *anthropicCodec
}

// NewAnthropicProvider builds the Anthropic Messages API provider implementation.
func NewAnthropicProvider(cfg ResolvedConfig) (llm.Provider, error) {
	return newAnthropicClient(cfg, string(ProviderAnthropic), nil)
}

func newAnthropicClient(cfg ResolvedConfig, name string, httpClient HTTPClient) (*anthropicClient, error) {
	apiKey := strings.TrimSpace(cfg.APIKey)
	if apiKey == "" {
		return nil, fmt.Errorf("key is required")
	}

	baseURL := strings.TrimSpace(cfg.BaseURL)
	if baseURL == "" {
		baseURL = defaultAnthropicBaseURL
	}

	headers := copyHeaders(cfg.Headers)
	ensureAuthHeader(headers, "x-api-key", apiKey)
	ensureRequiredHeader(headers, "anthropic-version", anthropicVersionHeader)

	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = anthropicDefaultTimeout
	}
	if httpClient == nil {
		httpClient = &http.Client{Timeout: timeout}
	}

	if strings.TrimSpace(name) == "" {
		name = string(cfg.Kind)
	}

	return &anthropicClient{
		name:       name,
		baseURL:    strings.TrimRight(baseURL, "/"),
		model:      strings.TrimSpace(cfg.Model),
		headers:    headers,
		httpClient: httpClient,
		codec:      newAnthropicCodec(cfg.Model),
	}, nil
}

func (c *anthropicClient) Name() string {
	return c.name
}

func (c *anthropicClient) Complete(ctx context.Context, req *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	body, err := c.codec.encodeRequest(req, false)
	if err != nil {
		return nil, fmt.Errorf("build request body: %w", err)
	}

	resp, err := DoJSON(ctx, c.httpClient, http.MethodPost, c.baseURL+"/v1/messages", c.headers, body)
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return nil, fmt.Errorf("request timeout: the operation took too long (>%v). Try reducing context size or increasing timeout", c.requestTimeout())
		}
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, parseAnthropicError(resp)
	}

	var decoded anthropicMessagesResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return nil, fmt.Errorf("response timeout: server took too long to respond. Try with a shorter conversation or increase timeout")
		}
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return c.codec.decodeCompletionResponse(decoded), nil
}

func (c *anthropicClient) CompleteStream(ctx context.Context, req *llm.CompletionRequest) (llm.StreamIterator, error) {
	body, err := c.codec.encodeRequest(req, true)
	if err != nil {
		return nil, fmt.Errorf("build request body: %w", err)
	}

	resp, err := DoJSON(ctx, c.httpClient, http.MethodPost, c.baseURL+"/v1/messages", c.headers, body)
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return nil, fmt.Errorf("request timeout: the operation took too long (>%v). Try reducing context size or increasing timeout", c.requestTimeout())
		}
		return nil, fmt.Errorf("request failed: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, parseAnthropicError(resp)
	}

	return c.codec.newStreamIterator(resp.Body), nil
}

func (c *anthropicClient) SupportsTools() bool {
	return true
}

func (c *anthropicClient) AvailableModels() []llm.ModelInfo {
	if c.model == "" {
		return nil
	}
	return []llm.ModelInfo{
		{ID: c.model, Provider: c.name},
	}
}

func (c *anthropicClient) requestTimeout() time.Duration {
	httpClient, ok := c.httpClient.(*http.Client)
	if !ok {
		return anthropicDefaultTimeout
	}
	if httpClient.Timeout == 0 {
		return anthropicDefaultTimeout
	}
	return httpClient.Timeout
}

func ensureRequiredHeader(headers map[string]string, key, value string) {
	if headers == nil {
		return
	}

	keys := matchingHeaderKeys(headers, key)
	if len(keys) == 0 {
		headers[key] = value
		return
	}

	chosenKey := keys[0]
	for _, existingKey := range keys {
		if existingKey == key {
			chosenKey = existingKey
			break
		}
	}
	for _, existingKey := range keys {
		if existingKey == chosenKey {
			continue
		}
		delete(headers, existingKey)
	}
	headers[chosenKey] = value
}

func parseAnthropicError(resp *http.Response) error {
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1024*1024))
	if err != nil {
		return fmt.Errorf("API error (status %d): failed to read error body: %w", resp.StatusCode, err)
	}
	if len(body) == 0 {
		return fmt.Errorf("API error (status %d): empty response", resp.StatusCode)
	}

	var decoded struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &decoded); err == nil && decoded.Error.Message != "" {
		return fmt.Errorf("API error (status %d, %s): %s", resp.StatusCode, decoded.Error.Type, decoded.Error.Message)
	}

	return fmt.Errorf("API error (status %d): %s", resp.StatusCode, strings.TrimSpace(string(body)))
}
