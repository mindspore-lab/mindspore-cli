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

const openAIDefaultTimeout = 180 * time.Second

type openAIClient struct {
	name       string
	baseURL    string
	model      string
	headers    map[string]string
	httpClient HTTPClient
	codec      *openAICodec
}

// NewOpenAIProvider builds the native OpenAI provider implementation.
func NewOpenAIProvider(cfg ResolvedConfig) (llm.Provider, error) {
	return newOpenAIClient(cfg, string(ProviderOpenAI), nil)
}

// NewOpenAIProviderWithHTTPClient builds the native OpenAI provider with a supplied HTTP client.
func NewOpenAIProviderWithHTTPClient(cfg ResolvedConfig, httpClient HTTPClient) (llm.Provider, error) {
	return newOpenAIClient(cfg, string(ProviderOpenAI), httpClient)
}

func newOpenAIClient(cfg ResolvedConfig, name string, httpClient HTTPClient) (*openAIClient, error) {
	apiKey := strings.TrimSpace(cfg.APIKey)
	if apiKey == "" {
		return nil, fmt.Errorf("key is required")
	}

	baseURL := strings.TrimSpace(cfg.BaseURL)
	if baseURL == "" {
		baseURL = defaultOpenAIBaseURL
	}

	headers := copyHeaders(cfg.Headers)
	authHeaderName := strings.TrimSpace(cfg.AuthHeaderName)
	if authHeaderName == "" {
		authHeaderName = "Authorization"
	}
	if _, ok := headers[authHeaderName]; !ok {
		headers[authHeaderName] = "Bearer " + apiKey
	}

	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = openAIDefaultTimeout
	}
	if httpClient == nil {
		httpClient = &http.Client{Timeout: timeout}
	}

	if strings.TrimSpace(name) == "" {
		name = string(cfg.Kind)
	}

	return &openAIClient{
		name:       name,
		baseURL:    strings.TrimRight(baseURL, "/"),
		model:      strings.TrimSpace(cfg.Model),
		headers:    headers,
		httpClient: httpClient,
		codec:      newOpenAICodec(cfg.Model),
	}, nil
}

func (c *openAIClient) Name() string {
	return c.name
}

func (c *openAIClient) Complete(ctx context.Context, req *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	body, err := c.codec.encodeRequest(req, false)
	if err != nil {
		return nil, fmt.Errorf("build request body: %w", err)
	}

	resp, err := DoJSON(ctx, c.httpClient, http.MethodPost, c.baseURL+"/chat/completions", c.headers, body)
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return nil, fmt.Errorf("request timeout: the operation took too long (>%v). Try reducing context size or increasing timeout", c.requestTimeout())
		}
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, parseOpenAIError(resp)
	}

	var decoded openAIChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return nil, fmt.Errorf("response timeout: server took too long to respond. Try with a shorter conversation or increase timeout")
		}
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return c.codec.decodeCompletionResponse(decoded), nil
}

func (c *openAIClient) CompleteStream(ctx context.Context, req *llm.CompletionRequest) (llm.StreamIterator, error) {
	body, err := c.codec.encodeRequest(req, true)
	if err != nil {
		return nil, fmt.Errorf("build request body: %w", err)
	}

	resp, err := DoJSON(ctx, c.httpClient, http.MethodPost, c.baseURL+"/chat/completions", c.headers, body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, parseOpenAIError(resp)
	}

	return c.codec.newStreamIterator(resp.Body), nil
}

func (c *openAIClient) SupportsTools() bool {
	return true
}

func (c *openAIClient) AvailableModels() []llm.ModelInfo {
	return []llm.ModelInfo{
		{ID: "gpt-4o", Provider: c.name, MaxTokens: 128000},
		{ID: "gpt-4o-mini", Provider: c.name, MaxTokens: 128000},
		{ID: "gpt-4-turbo", Provider: c.name, MaxTokens: 128000},
		{ID: "gpt-4", Provider: c.name, MaxTokens: 8192},
		{ID: "gpt-3.5-turbo", Provider: c.name, MaxTokens: 16385},
	}
}

func (c *openAIClient) requestTimeout() time.Duration {
	httpClient, ok := c.httpClient.(*http.Client)
	if !ok {
		return openAIDefaultTimeout
	}
	if httpClient.Timeout == 0 {
		return openAIDefaultTimeout
	}
	return httpClient.Timeout
}

func copyHeaders(headers map[string]string) map[string]string {
	if len(headers) == 0 {
		return map[string]string{}
	}

	cloned := make(map[string]string, len(headers))
	for key, value := range headers {
		if strings.TrimSpace(key) == "" {
			continue
		}
		cloned[key] = value
	}
	return cloned
}

func parseOpenAIError(resp *http.Response) error {
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1024*1024))
	if err != nil {
		return fmt.Errorf("API error (status %d): failed to read error body: %w", resp.StatusCode, err)
	}

	if len(body) == 0 {
		return fmt.Errorf("API error (status %d): empty response", resp.StatusCode)
	}

	var decoded struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &decoded); err == nil && decoded.Error.Message != "" {
		return fmt.Errorf("API error (status %d, %s): %s", resp.StatusCode, decoded.Error.Type, decoded.Error.Message)
	}

	return fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
}
