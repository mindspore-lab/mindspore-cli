package llm

import (
	"encoding/json"
	"io"
	"strings"
	"testing"
)

func TestAnthropicEncodeRequestDefaultsMaxTokens(t *testing.T) {
	req, err := newAnthropicCodec("claude-test").encodeRequest(&CompletionRequest{
		Messages: []Message{{Role: "user", Content: "ping"}},
	}, false)
	if err != nil {
		t.Fatalf("encodeRequest() error = %v", err)
	}

	if req.MaxTokens == nil {
		t.Fatal("req.MaxTokens = nil, want default value")
	}
	if got, want := *req.MaxTokens, anthropicDefaultMaxTokens; got != want {
		t.Fatalf("req.MaxTokens = %d, want %d", got, want)
	}
}

func TestAnthropicEncodeRequestPreservesExplicitMaxTokens(t *testing.T) {
	maxTokens := 2048
	req, err := newAnthropicCodec("claude-test").encodeRequest(&CompletionRequest{
		Messages:  []Message{{Role: "user", Content: "ping"}},
		MaxTokens: &maxTokens,
	}, false)
	if err != nil {
		t.Fatalf("encodeRequest() error = %v", err)
	}

	if req.MaxTokens == nil {
		t.Fatal("req.MaxTokens = nil, want explicit value")
	}
	if got, want := *req.MaxTokens, 2048; got != want {
		t.Fatalf("req.MaxTokens = %d, want %d", got, want)
	}
}

func TestAnthropicStreamIteratorAccumulatesToolUseJSONWithoutBuilderCopyPanic(t *testing.T) {
	stream := strings.Join([]string{
		mustAnthropicSSEEvent(t, "message_start", anthropicStreamMessageStartEvent{
			Message: struct {
				ID    string         `json:"id"`
				Model string         `json:"model"`
				Usage anthropicUsage `json:"usage"`
			}{
				ID:    "msg_123",
				Model: "claude-test",
				Usage: anthropicUsage{InputTokens: 12},
			},
		}),
		mustAnthropicSSEEvent(t, "content_block_start", anthropicStreamContentBlockStartEvent{
			Index: 0,
			ContentBlock: anthropicContentBlock{
				Type: "tool_use",
				ID:   "toolu_123",
				Name: "read_file",
			},
		}),
		mustAnthropicSSEEvent(t, "content_block_delta", anthropicStreamContentBlockDeltaEvent{
			Index: 0,
			Delta: struct {
				Type        string `json:"type"`
				Text        string `json:"text,omitempty"`
				PartialJSON string `json:"partial_json,omitempty"`
			}{
				Type:        "input_json_delta",
				PartialJSON: `{"path":"README.md",`,
			},
		}),
		mustAnthropicSSEEvent(t, "content_block_delta", anthropicStreamContentBlockDeltaEvent{
			Index: 0,
			Delta: struct {
				Type        string `json:"type"`
				Text        string `json:"text,omitempty"`
				PartialJSON string `json:"partial_json,omitempty"`
			}{
				Type:        "input_json_delta",
				PartialJSON: `"limit":25}`,
			},
		}),
		mustAnthropicSSEEvent(t, "content_block_stop", anthropicStreamContentBlockStopEvent{
			Index: 0,
		}),
		mustAnthropicSSEEvent(t, "message_delta", anthropicStreamMessageDeltaEvent{
			Delta: struct {
				StopReason string `json:"stop_reason"`
			}{
				StopReason: "tool_use",
			},
			Usage: anthropicUsage{OutputTokens: 7},
		}),
		"event: message_stop\n\n",
	}, "")

	it := newAnthropicCodec("").newStreamIterator(io.NopCloser(strings.NewReader(stream)))
	t.Cleanup(func() {
		if err := it.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
	})

	var firstChunk *StreamChunk
	var finalChunk *StreamChunk

	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("Next() panicked while appending tool JSON deltas: %v", r)
			}
		}()

		var err error
		firstChunk, err = it.Next()
		if err != nil {
			t.Fatalf("first Next() error = %v", err)
		}

		finalChunk, err = it.Next()
		if err != nil {
			t.Fatalf("second Next() error = %v", err)
		}

		_, err = it.Next()
		if err != io.EOF {
			t.Fatalf("third Next() error = %v, want EOF", err)
		}
	}()

	if firstChunk == nil {
		t.Fatal("first chunk = nil, want tool call chunk")
	}
	if got, want := len(firstChunk.ToolCalls), 1; got != want {
		t.Fatalf("len(firstChunk.ToolCalls) = %d, want %d", got, want)
	}
	if got, want := firstChunk.ToolCalls[0].ID, "toolu_123"; got != want {
		t.Fatalf("firstChunk.ToolCalls[0].ID = %q, want %q", got, want)
	}
	if got, want := firstChunk.ToolCalls[0].Function.Name, "read_file"; got != want {
		t.Fatalf("firstChunk.ToolCalls[0].Function.Name = %q, want %q", got, want)
	}
	if got, want := string(firstChunk.ToolCalls[0].Function.Arguments), `{"path":"README.md","limit":25}`; got != want {
		t.Fatalf("firstChunk.ToolCalls[0].Function.Arguments = %s, want %s", got, want)
	}

	if finalChunk == nil {
		t.Fatal("final chunk = nil, want finish chunk")
	}
	if got, want := finalChunk.FinishReason, FinishToolCalls; got != want {
		t.Fatalf("finalChunk.FinishReason = %q, want %q", got, want)
	}
	if finalChunk.Usage == nil {
		t.Fatal("finalChunk.Usage = nil, want usage")
	}
	if got, want := finalChunk.Usage.PromptTokens, 12; got != want {
		t.Fatalf("finalChunk.Usage.PromptTokens = %d, want %d", got, want)
	}
	if got, want := finalChunk.Usage.CompletionTokens, 7; got != want {
		t.Fatalf("finalChunk.Usage.CompletionTokens = %d, want %d", got, want)
	}
	if got, want := finalChunk.Usage.TotalTokens, 19; got != want {
		t.Fatalf("finalChunk.Usage.TotalTokens = %d, want %d", got, want)
	}
	if got, want := len(finalChunk.ToolCalls), 1; got != want {
		t.Fatalf("len(finalChunk.ToolCalls) = %d, want %d", got, want)
	}
	if got, want := string(finalChunk.ToolCalls[0].Function.Arguments), `{"path":"README.md","limit":25}`; got != want {
		t.Fatalf("finalChunk.ToolCalls[0].Function.Arguments = %s, want %s", got, want)
	}
}

func TestAnthropicStreamIteratorUsesStartBlockInputWithoutJSONDeltas(t *testing.T) {
	stream := strings.Join([]string{
		mustAnthropicSSEEvent(t, "message_start", anthropicStreamMessageStartEvent{
			Message: struct {
				ID    string         `json:"id"`
				Model string         `json:"model"`
				Usage anthropicUsage `json:"usage"`
			}{
				ID:    "msg_234",
				Model: "claude-test",
				Usage: anthropicUsage{InputTokens: 5},
			},
		}),
		mustAnthropicSSEEvent(t, "content_block_start", anthropicStreamContentBlockStartEvent{
			Index: 0,
			ContentBlock: anthropicContentBlock{
				Type:  "tool_use",
				ID:    "toolu_234",
				Name:  "read_file",
				Input: json.RawMessage(`{"path":"docs/arch.md"}`),
			},
		}),
		mustAnthropicSSEEvent(t, "content_block_stop", anthropicStreamContentBlockStopEvent{
			Index: 0,
		}),
		mustAnthropicSSEEvent(t, "message_delta", anthropicStreamMessageDeltaEvent{
			Delta: struct {
				StopReason string `json:"stop_reason"`
			}{
				StopReason: "tool_use",
			},
			Usage: anthropicUsage{OutputTokens: 3},
		}),
		"event: message_stop\n\n",
	}, "")

	firstChunk, finalChunk := runAnthropicStreamAndAssertNoPanic(t, stream)

	if firstChunk == nil {
		t.Fatal("first chunk = nil, want tool call chunk")
	}
	if got, want := len(firstChunk.ToolCalls), 1; got != want {
		t.Fatalf("len(firstChunk.ToolCalls) = %d, want %d", got, want)
	}
	if got, want := string(firstChunk.ToolCalls[0].Function.Arguments), `{"path":"docs/arch.md"}`; got != want {
		t.Fatalf("firstChunk.ToolCalls[0].Function.Arguments = %s, want %s", got, want)
	}
	if got, want := finalChunk.FinishReason, FinishToolCalls; got != want {
		t.Fatalf("finalChunk.FinishReason = %q, want %q", got, want)
	}
	if got, want := string(finalChunk.ToolCalls[0].Function.Arguments), `{"path":"docs/arch.md"}`; got != want {
		t.Fatalf("finalChunk.ToolCalls[0].Function.Arguments = %s, want %s", got, want)
	}
}

func TestAnthropicStreamIteratorSignalsBackgroundWorkWhenToolJSONFollowsText(t *testing.T) {
	stream := strings.Join([]string{
		mustAnthropicSSEEvent(t, "message_start", anthropicStreamMessageStartEvent{
			Message: struct {
				ID    string         `json:"id"`
				Model string         `json:"model"`
				Usage anthropicUsage `json:"usage"`
			}{
				ID:    "msg_456",
				Model: "claude-test",
				Usage: anthropicUsage{InputTokens: 9},
			},
		}),
		mustAnthropicSSEEvent(t, "content_block_delta", anthropicStreamContentBlockDeltaEvent{
			Index: 0,
			Delta: struct {
				Type        string `json:"type"`
				Text        string `json:"text,omitempty"`
				PartialJSON string `json:"partial_json,omitempty"`
			}{
				Type: "text_delta",
				Text: "好的，我来处理。",
			},
		}),
		mustAnthropicSSEEvent(t, "content_block_start", anthropicStreamContentBlockStartEvent{
			Index: 1,
			ContentBlock: anthropicContentBlock{
				Type: "tool_use",
				ID:   "toolu_456",
				Name: "write",
			},
		}),
		mustAnthropicSSEEvent(t, "content_block_delta", anthropicStreamContentBlockDeltaEvent{
			Index: 1,
			Delta: struct {
				Type        string `json:"type"`
				Text        string `json:"text,omitempty"`
				PartialJSON string `json:"partial_json,omitempty"`
			}{
				Type:        "input_json_delta",
				PartialJSON: `{"path":"README.md"`,
			},
		}),
		mustAnthropicSSEEvent(t, "content_block_delta", anthropicStreamContentBlockDeltaEvent{
			Index: 1,
			Delta: struct {
				Type        string `json:"type"`
				Text        string `json:"text,omitempty"`
				PartialJSON string `json:"partial_json,omitempty"`
			}{
				Type:        "input_json_delta",
				PartialJSON: `,"content":"hello"}`,
			},
		}),
		mustAnthropicSSEEvent(t, "content_block_stop", anthropicStreamContentBlockStopEvent{
			Index: 1,
		}),
		mustAnthropicSSEEvent(t, "message_delta", anthropicStreamMessageDeltaEvent{
			Delta: struct {
				StopReason string `json:"stop_reason"`
			}{
				StopReason: "tool_use",
			},
			Usage: anthropicUsage{OutputTokens: 5},
		}),
		"event: message_stop\n\n",
	}, "")

	it := newAnthropicCodec("").newStreamIterator(io.NopCloser(strings.NewReader(stream)))
	t.Cleanup(func() {
		if err := it.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
	})

	textChunk, err := it.Next()
	if err != nil {
		t.Fatalf("first Next() error = %v", err)
	}
	if textChunk == nil || textChunk.Content != "好的，我来处理。" {
		t.Fatalf("first chunk = %#v, want streamed text content", textChunk)
	}

	backgroundChunk, err := it.Next()
	if err != nil {
		t.Fatalf("second Next() error = %v", err)
	}
	if backgroundChunk == nil || !backgroundChunk.BackgroundWork {
		t.Fatalf("second chunk = %#v, want BackgroundWork signal", backgroundChunk)
	}

	toolChunk, err := it.Next()
	if err != nil {
		t.Fatalf("third Next() error = %v", err)
	}
	if toolChunk == nil || len(toolChunk.ToolCalls) != 1 {
		t.Fatalf("third chunk = %#v, want tool call chunk", toolChunk)
	}
}

func TestAnthropicStreamIteratorAccumulatesSingleToolUseJSONDelta(t *testing.T) {
	stream := strings.Join([]string{
		mustAnthropicSSEEvent(t, "message_start", anthropicStreamMessageStartEvent{
			Message: struct {
				ID    string         `json:"id"`
				Model string         `json:"model"`
				Usage anthropicUsage `json:"usage"`
			}{
				ID:    "msg_345",
				Model: "claude-test",
				Usage: anthropicUsage{InputTokens: 8},
			},
		}),
		mustAnthropicSSEEvent(t, "content_block_start", anthropicStreamContentBlockStartEvent{
			Index: 0,
			ContentBlock: anthropicContentBlock{
				Type: "tool_use",
				ID:   "toolu_345",
				Name: "list_dir",
			},
		}),
		mustAnthropicSSEEvent(t, "content_block_delta", anthropicStreamContentBlockDeltaEvent{
			Index: 0,
			Delta: struct {
				Type        string `json:"type"`
				Text        string `json:"text,omitempty"`
				PartialJSON string `json:"partial_json,omitempty"`
			}{
				Type:        "input_json_delta",
				PartialJSON: `{"path":"."}`,
			},
		}),
		mustAnthropicSSEEvent(t, "content_block_stop", anthropicStreamContentBlockStopEvent{
			Index: 0,
		}),
		mustAnthropicSSEEvent(t, "message_delta", anthropicStreamMessageDeltaEvent{
			Delta: struct {
				StopReason string `json:"stop_reason"`
			}{
				StopReason: "tool_use",
			},
			Usage: anthropicUsage{OutputTokens: 4},
		}),
		"event: message_stop\n\n",
	}, "")

	firstChunk, finalChunk := runAnthropicStreamAndAssertNoPanic(t, stream)

	if firstChunk == nil {
		t.Fatal("first chunk = nil, want tool call chunk")
	}
	if got, want := len(firstChunk.ToolCalls), 1; got != want {
		t.Fatalf("len(firstChunk.ToolCalls) = %d, want %d", got, want)
	}
	if got, want := string(firstChunk.ToolCalls[0].Function.Arguments), `{"path":"."}`; got != want {
		t.Fatalf("firstChunk.ToolCalls[0].Function.Arguments = %s, want %s", got, want)
	}
	if got, want := finalChunk.FinishReason, FinishToolCalls; got != want {
		t.Fatalf("finalChunk.FinishReason = %q, want %q", got, want)
	}
	if got, want := string(finalChunk.ToolCalls[0].Function.Arguments), `{"path":"."}`; got != want {
		t.Fatalf("finalChunk.ToolCalls[0].Function.Arguments = %s, want %s", got, want)
	}
}

func TestAnthropicStreamIteratorFallsBackToStartInputWhenPartialJSONInvalid(t *testing.T) {
	stream := strings.Join([]string{
		mustAnthropicSSEEvent(t, "message_start", anthropicStreamMessageStartEvent{
			Message: struct {
				ID    string         `json:"id"`
				Model string         `json:"model"`
				Usage anthropicUsage `json:"usage"`
			}{
				ID:    "msg_456",
				Model: "claude-test",
				Usage: anthropicUsage{InputTokens: 9},
			},
		}),
		mustAnthropicSSEEvent(t, "content_block_start", anthropicStreamContentBlockStartEvent{
			Index: 2,
			ContentBlock: anthropicContentBlock{
				Type:  "tool_use",
				ID:    "toolu_456",
				Name:  "read_file",
				Input: json.RawMessage(`{"path":"docs/arch.md"}`),
			},
		}),
		mustAnthropicSSEEvent(t, "content_block_delta", anthropicStreamContentBlockDeltaEvent{
			Index: 2,
			Delta: struct {
				Type        string `json:"type"`
				Text        string `json:"text,omitempty"`
				PartialJSON string `json:"partial_json,omitempty"`
			}{
				Type:        "input_json_delta",
				PartialJSON: `{"path":"docs/arch.md"`,
			},
		}),
		mustAnthropicSSEEvent(t, "content_block_stop", anthropicStreamContentBlockStopEvent{
			Index: 2,
		}),
		mustAnthropicSSEEvent(t, "message_delta", anthropicStreamMessageDeltaEvent{
			Delta: struct {
				StopReason string `json:"stop_reason"`
			}{
				StopReason: "tool_use",
			},
			Usage: anthropicUsage{OutputTokens: 2},
		}),
		"event: message_stop\n\n",
	}, "")

	it := newAnthropicCodec("").newStreamIterator(io.NopCloser(strings.NewReader(stream)))
	t.Cleanup(func() {
		if err := it.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
	})

	firstChunk, err := it.Next()
	if err != nil {
		t.Fatalf("first Next() error = %v", err)
	}
	if firstChunk == nil {
		t.Fatal("first chunk = nil, want tool call chunk")
	}
	if got, want := len(firstChunk.ToolCalls), 1; got != want {
		t.Fatalf("len(firstChunk.ToolCalls) = %d, want %d", got, want)
	}
	if got, want := string(firstChunk.ToolCalls[0].Function.Arguments), `{"path":"docs/arch.md"}`; got != want {
		t.Fatalf("firstChunk.ToolCalls[0].Function.Arguments = %s, want %s", got, want)
	}
}

func runAnthropicStreamAndAssertNoPanic(t *testing.T, stream string) (*StreamChunk, *StreamChunk) {
	t.Helper()

	it := newAnthropicCodec("").newStreamIterator(io.NopCloser(strings.NewReader(stream)))
	t.Cleanup(func() {
		if err := it.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
	})

	var firstChunk *StreamChunk
	var finalChunk *StreamChunk

	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("Next() panicked while consuming anthropic stream: %v", r)
			}
		}()

		var err error
		firstChunk, err = it.Next()
		if err != nil {
			t.Fatalf("first Next() error = %v", err)
		}

		finalChunk, err = it.Next()
		if err != nil {
			t.Fatalf("second Next() error = %v", err)
		}

		_, err = it.Next()
		if err != io.EOF {
			t.Fatalf("third Next() error = %v, want EOF", err)
		}
	}()

	if finalChunk == nil {
		t.Fatal("final chunk = nil, want finish chunk")
	}
	if finalChunk.Usage == nil {
		t.Fatal("finalChunk.Usage = nil, want usage")
	}

	return firstChunk, finalChunk
}

func mustAnthropicSSEEvent(t *testing.T, event string, payload any) string {
	t.Helper()

	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	return "event: " + event + "\n" +
		"data: " + string(data) + "\n\n"
}
