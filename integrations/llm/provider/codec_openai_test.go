package provider

import (
	"testing"

	"github.com/vigo999/ms-cli/integrations/llm"
)

func TestOpenAICodec_EncodeMessagesWithToolCalls(t *testing.T) {
	codec := newOpenAICodec("gpt-default")

	req, err := codec.encodeRequest(&llm.CompletionRequest{
		Messages: []llm.Message{
			llm.NewSystemMessage("system prompt"),
			{
				Role:    "assistant",
				Content: "",
				ToolCalls: []llm.ToolCall{
					{
						ID:   "call_weather",
						Type: "function",
						Function: llm.ToolCallFunc{
							Name:      "lookup_weather",
							Arguments: []byte(`{"city":"Shanghai"}`),
						},
					},
				},
			},
			llm.NewToolMessage("call_weather", `{"ok":true}`),
		},
		Tools: []llm.Tool{
			{
				Type: "function",
				Function: llm.ToolFunction{
					Name:        "lookup_weather",
					Description: "Look up the weather",
					Parameters: llm.ToolSchema{
						Type: "object",
						Properties: map[string]llm.Property{
							"city": {Type: "string"},
						},
						Required: []string{"city"},
					},
				},
			},
		},
	}, false)
	if err != nil {
		t.Fatalf("encodeRequest() error = %v", err)
	}

	if req.Model != "gpt-default" {
		t.Fatalf("model = %q, want %q", req.Model, "gpt-default")
	}
	if got := len(req.Messages); got != 3 {
		t.Fatalf("len(messages) = %d, want 3", got)
	}

	if req.Messages[1].Role != "assistant" {
		t.Fatalf("assistant role = %q, want assistant", req.Messages[1].Role)
	}
	if got := len(req.Messages[1].ToolCalls); got != 1 {
		t.Fatalf("assistant tool_calls len = %d, want 1", got)
	}
	if req.Messages[1].ToolCalls[0].Function.Arguments != `{"city":"Shanghai"}` {
		t.Fatalf("assistant tool args = %q", req.Messages[1].ToolCalls[0].Function.Arguments)
	}
	if req.Messages[2].ToolCallID != "call_weather" {
		t.Fatalf("tool_call_id = %q, want %q", req.Messages[2].ToolCallID, "call_weather")
	}
	if got := len(req.Tools); got != 1 {
		t.Fatalf("len(tools) = %d, want 1", got)
	}
	if req.Tools[0].Function.Name != "lookup_weather" {
		t.Fatalf("tool name = %q, want %q", req.Tools[0].Function.Name, "lookup_weather")
	}
}

func TestOpenAICodec_ParseCompletionResponseWithToolCalls(t *testing.T) {
	codec := newOpenAICodec("gpt-default")

	resp := codec.decodeCompletionResponse(openAIChatCompletionResponse{
		ID:    "resp_123",
		Model: "gpt-4o-mini",
		Choices: []openAIChoice{
			{
				Index: 0,
				Message: openAIMessage{
					Role:    "assistant",
					Content: "",
					ToolCalls: []openAIToolCall{
						{
							ID:   "call_weather",
							Type: "function",
							Function: openAIToolCallFunction{
								Name:      "lookup_weather",
								Arguments: `{"city":"Shanghai"}`,
							},
						},
					},
				},
				FinishReason: string(llm.FinishStop),
			},
		},
		Usage: openAIUsage{
			PromptTokens:     11,
			CompletionTokens: 7,
			TotalTokens:      18,
		},
	})

	if resp == nil {
		t.Fatal("decodeCompletionResponse() returned nil response")
	}
	if resp.ID != "resp_123" {
		t.Fatalf("id = %q, want %q", resp.ID, "resp_123")
	}
	if resp.FinishReason != llm.FinishToolCalls {
		t.Fatalf("finish_reason = %q, want %q", resp.FinishReason, llm.FinishToolCalls)
	}
	if got := len(resp.ToolCalls); got != 1 {
		t.Fatalf("len(tool_calls) = %d, want 1", got)
	}
	if resp.ToolCalls[0].Function.Name != "lookup_weather" {
		t.Fatalf("tool call name = %q, want %q", resp.ToolCalls[0].Function.Name, "lookup_weather")
	}
	if string(resp.ToolCalls[0].Function.Arguments) != `{"city":"Shanghai"}` {
		t.Fatalf("tool call args = %q", string(resp.ToolCalls[0].Function.Arguments))
	}
	if resp.Usage.TotalTokens != 18 {
		t.Fatalf("total_tokens = %d, want 18", resp.Usage.TotalTokens)
	}
}

func TestOpenAICodec_StreamAssembleToolCallsFromDeltas(t *testing.T) {
	iter := &openAIStreamIterator{}

	iter.applyToolCallDelta([]openAIStreamToolCall{
		{
			Index: intPtr(0),
			ID:    "call_weather",
			Type:  "function",
			Function: openAIStreamToolCallFields{
				Name:      "lookup_weather",
				Arguments: `{"city"`,
			},
		},
		{
			Index: intPtr(1),
			ID:    "call_time",
			Type:  "function",
			Function: openAIStreamToolCallFields{
				Name:      "lookup_time",
				Arguments: `{"zone"`,
			},
		},
	})

	if got := string(iter.accumulated.ToolCalls[0].Function.Arguments); got != `{"city"` {
		t.Fatalf("partial tool 0 args = %q, want %q", got, `{"city"`)
	}

	iter.applyToolCallDelta([]openAIStreamToolCall{
		{
			Index: intPtr(0),
			Function: openAIStreamToolCallFields{
				Arguments: `:"Shanghai"}`,
			},
		},
		{
			Index: intPtr(1),
			Function: openAIStreamToolCallFields{
				Arguments: `:"Asia/Shanghai"}`,
			},
		},
	})

	lastToolChunk := iter.accumulated
	if got := len(lastToolChunk.ToolCalls); got != 2 {
		t.Fatalf("len(tool_calls) = %d, want 2", got)
	}
	if string(lastToolChunk.ToolCalls[0].Function.Arguments) != `{"city":"Shanghai"}` {
		t.Fatalf("tool 0 args = %q", string(lastToolChunk.ToolCalls[0].Function.Arguments))
	}
	if string(lastToolChunk.ToolCalls[1].Function.Arguments) != `{"zone":"Asia/Shanghai"}` {
		t.Fatalf("tool 1 args = %q", string(lastToolChunk.ToolCalls[1].Function.Arguments))
	}
}

func intPtr(v int) *int {
	return &v
}

func TestOpenAICompatibleBuilder_SharesOpenAIProtocolPath(t *testing.T) {
	provider, err := NewOpenAICompatibleProvider(ResolvedConfig{
		Kind:    ProviderOpenAICompatible,
		Model:   "gpt-4o-mini",
		BaseURL: "https://example.invalid/v1",
		APIKey:  "test-key",
		Headers: map[string]string{"Authorization": "Bearer test-key"},
	})
	if err != nil {
		t.Fatalf("NewOpenAICompatibleProvider() error = %v", err)
	}

	if provider.Name() != string(ProviderOpenAICompatible) {
		t.Fatalf("provider.Name() = %q, want %q", provider.Name(), ProviderOpenAICompatible)
	}
}

func TestOpenAIBuilder_UsesConfiguredBaseURLAndHeaders(t *testing.T) {
	provider, err := NewOpenAIProvider(ResolvedConfig{
		Kind:    ProviderOpenAI,
		Model:   "gpt-4o-mini",
		BaseURL: "https://example.invalid/custom",
		APIKey:  "test-key",
		Headers: map[string]string{"Authorization": "Bearer test-key", "X-Trace-ID": "trace-123"},
	})
	if err != nil {
		t.Fatalf("NewOpenAIProvider() error = %v", err)
	}

	client, ok := provider.(*openAIClient)
	if !ok {
		t.Fatalf("provider type = %T, want *openAIClient", provider)
	}

	if client.baseURL != "https://example.invalid/custom" {
		t.Fatalf("baseURL = %q, want %q", client.baseURL, "https://example.invalid/custom")
	}
	if client.headers["X-Trace-ID"] != "trace-123" {
		t.Fatalf("X-Trace-ID = %q, want %q", client.headers["X-Trace-ID"], "trace-123")
	}
}
