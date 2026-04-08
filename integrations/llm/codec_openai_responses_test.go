package llm

import (
	"io"
	"strings"
	"testing"
)

func TestOpenAIResponsesStreamIteratorSignalsBackgroundWorkWhenToolArgsFollowText(t *testing.T) {
	stream := strings.Join([]string{
		`data: {"type":"response.output_text.delta","delta":"好的，我来处理。","output_index":0}`,
		`data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_123","call_id":"call_123","name":"write"}}`,
		`data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"path\":\"README.md\""}`,
		`data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":",\"content\":\"hello\"}"}`,
		`data: {"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"fc_123","call_id":"call_123","name":"write","arguments":"{\"path\":\"README.md\",\"content\":\"hello\"}"}}`,
		`data: {"type":"response.completed","response":{"id":"resp_123","model":"gpt-test","usage":{"input_tokens":10,"output_tokens":4,"total_tokens":14}}}`,
		`data: [DONE]`,
	}, "\n") + "\n"

	it := newOpenAIResponsesCodec("").newStreamIterator(io.NopCloser(strings.NewReader(stream)))
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
