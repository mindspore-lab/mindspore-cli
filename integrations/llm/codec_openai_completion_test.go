package llm

import (
	"io"
	"strings"
	"testing"
)

func TestOpenAIEncodeStreamRequestIncludesUsage(t *testing.T) {
	codec := newOpenAICodec("gpt-4o")
	req := &CompletionRequest{
		Messages: []Message{NewUserMessage("hello")},
	}

	body, err := codec.encodeRequest(req, true)
	if err != nil {
		t.Fatalf("encodeRequest failed: %v", err)
	}

	if body.StreamOptions == nil {
		t.Fatal("StreamOptions = nil, want include_usage enabled")
	}
	if !body.StreamOptions.IncludeUsage {
		t.Fatal("StreamOptions.IncludeUsage = false, want true")
	}
}

func TestOpenAIStreamIteratorCapturesUsageAfterFinishReason(t *testing.T) {
	stream := strings.Join([]string{
		`data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
		"",
		`data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		"",
		`data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":11,"completion_tokens":2,"total_tokens":13,"prompt_tokens_details":{"cached_tokens":3}}}`,
		"",
		`data: [DONE]`,
		"",
	}, "\n")

	iter := (&openAICodec{}).newStreamIterator(io.NopCloser(strings.NewReader(stream)))

	var (
		content      strings.Builder
		finishReason FinishReason
		usage        *Usage
	)

	for {
		chunk, err := iter.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Next failed: %v", err)
		}
		if chunk == nil {
			continue
		}
		content.WriteString(chunk.Content)
		if chunk.FinishReason != "" {
			finishReason = chunk.FinishReason
		}
		if chunk.Usage != nil {
			copied := *chunk.Usage
			usage = &copied
		}
	}

	if got, want := content.String(), "hi"; got != want {
		t.Fatalf("streamed content = %q, want %q", got, want)
	}
	if got, want := finishReason, FinishStop; got != want {
		t.Fatalf("finish reason = %q, want %q", got, want)
	}
	if usage == nil {
		t.Fatal("usage = nil, want usage chunk")
	}
	if got, want := usage.PromptTokens, 11; got != want {
		t.Fatalf("usage.PromptTokens = %d, want %d", got, want)
	}
	if got, want := usage.CompletionTokens, 2; got != want {
		t.Fatalf("usage.CompletionTokens = %d, want %d", got, want)
	}
	if got, want := usage.TotalTokens, 13; got != want {
		t.Fatalf("usage.TotalTokens = %d, want %d", got, want)
	}
	if got, want := string(usage.Raw), `{"prompt_tokens":11,"completion_tokens":2,"total_tokens":13,"prompt_tokens_details":{"cached_tokens":3}}`; got != want {
		t.Fatalf("usage.Raw = %s, want %s", got, want)
	}
}
