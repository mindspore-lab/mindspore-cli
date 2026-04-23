package llm

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

type openAICodec struct {
	defaultModel string
}

func newOpenAICodec(defaultModel string) *openAICodec {
	return &openAICodec{defaultModel: strings.TrimSpace(defaultModel)}
}

func (c *openAICodec) encodeRequest(req *CompletionRequest, stream bool) (openAIChatCompletionRequest, error) {
	if req == nil {
		return openAIChatCompletionRequest{}, fmt.Errorf("request is nil")
	}

	model := strings.TrimSpace(req.Model)
	if model == "" {
		model = c.defaultModel
	}
	if model == "" {
		model = "gpt-4o-mini"
	}

	var streamOptions *openAIStreamOptions
	if stream {
		streamOptions = &openAIStreamOptions{IncludeUsage: true}
	}

	return openAIChatCompletionRequest{
		Model:         model,
		Messages:      c.encodeMessages(req.Messages),
		Temperature:   req.Temperature,
		MaxTokens:     req.MaxTokens,
		TopP:          req.TopP,
		Stop:          req.Stop,
		Tools:         c.encodeTools(req.Tools),
		Stream:        stream,
		StreamOptions: streamOptions,
	}, nil
}

func (c *openAICodec) encodeMessages(msgs []Message) []openAIMessage {
	if len(msgs) == 0 {
		return nil
	}

	result := make([]openAIMessage, len(msgs))
	for i, m := range msgs {
		result[i] = openAIMessage{
			Role:       m.Role,
			Content:    m.Content,
			ToolCalls:  c.encodeToolCalls(m.ToolCalls),
			ToolCallID: m.ToolCallID,
		}
	}
	return result
}

func (c *openAICodec) encodeToolCalls(calls []ToolCall) []openAIToolCall {
	if len(calls) == 0 {
		return nil
	}

	result := make([]openAIToolCall, len(calls))
	for i, tc := range calls {
		result[i] = openAIToolCall{
			ID:   tc.ID,
			Type: tc.Type,
			Function: openAIToolCallFunction{
				Name:      tc.Function.Name,
				Arguments: string(tc.Function.Arguments),
			},
		}
	}
	return result
}

func (c *openAICodec) encodeTools(tools []Tool) []openAITool {
	if len(tools) == 0 {
		return nil
	}

	result := make([]openAITool, len(tools))
	for i, tool := range tools {
		result[i] = openAITool{
			Type: tool.Type,
			Function: openAIToolFunction{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  tool.Function.Parameters,
			},
		}
	}
	return result
}

func (c *openAICodec) decodeCompletionResponse(resp openAIChatCompletionResponse) *CompletionResponse {
	if len(resp.Choices) == 0 {
		return &CompletionResponse{
			ID:    resp.ID,
			Model: resp.Model,
		}
	}

	choice := resp.Choices[0]
	result := &CompletionResponse{
		ID:           resp.ID,
		Model:        resp.Model,
		Content:      choice.Message.Content,
		FinishReason: FinishReason(choice.FinishReason),
		Usage:        resp.Usage.toUsage(),
	}

	if len(choice.Message.ToolCalls) == 0 {
		return result
	}

	result.ToolCalls = make([]ToolCall, len(choice.Message.ToolCalls))
	for i, tc := range choice.Message.ToolCalls {
		result.ToolCalls[i] = ToolCall{
			ID:   tc.ID,
			Type: tc.Type,
			Function: ToolCallFunc{
				Name:      tc.Function.Name,
				Arguments: json.RawMessage(tc.Function.Arguments),
			},
		}
	}
	result.FinishReason = FinishToolCalls

	return result
}

func (c *openAICodec) newStreamIterator(body io.ReadCloser) StreamIterator {
	return &openAIStreamIterator{
		reader: bufio.NewReader(body),
		closer: body,
	}
}

type openAIMessage struct {
	Role       string           `json:"role"`
	Content    string           `json:"content"`
	ToolCalls  []openAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

type openAITool struct {
	Type     string             `json:"type"`
	Function openAIToolFunction `json:"function"`
}

type openAIToolFunction struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Parameters  ToolSchema `json:"parameters"`
}

type openAIToolCall struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Function openAIToolCallFunction `json:"function"`
}

type openAIToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openAIChatCompletionRequest struct {
	Model         string               `json:"model"`
	Messages      []openAIMessage      `json:"messages"`
	Temperature   *float32             `json:"temperature,omitempty"`
	MaxTokens     *int                 `json:"max_tokens,omitempty"`
	TopP          float32              `json:"top_p,omitempty"`
	Stop          []string             `json:"stop,omitempty"`
	Tools         []openAITool         `json:"tools,omitempty"`
	Stream        bool                 `json:"stream"`
	StreamOptions *openAIStreamOptions `json:"stream_options,omitempty"`
}

type openAIStreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

type openAIChatCompletionResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []openAIChoice `json:"choices"`
	Usage   openAIUsage    `json:"usage"`
}

type openAIChoice struct {
	Index        int           `json:"index"`
	Message      openAIMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type openAIUsage struct {
	PromptTokens     int             `json:"prompt_tokens"`
	CompletionTokens int             `json:"completion_tokens"`
	TotalTokens      int             `json:"total_tokens"`
	Raw              json.RawMessage `json:"-"`
}

type openAIUsagePayload struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func (u *openAIUsage) UnmarshalJSON(data []byte) error {
	var payload openAIUsagePayload
	if err := json.Unmarshal(data, &payload); err != nil {
		return err
	}
	u.PromptTokens = payload.PromptTokens
	u.CompletionTokens = payload.CompletionTokens
	u.TotalTokens = payload.TotalTokens
	u.Raw = cloneRawJSON(data)
	return nil
}

func (u openAIUsage) toUsage() Usage {
	return Usage{
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
		Raw:              cloneRawJSON(u.Raw),
	}
}

type openAIStreamResponse struct {
	ID      string               `json:"id"`
	Object  string               `json:"object"`
	Created int64                `json:"created"`
	Model   string               `json:"model"`
	Choices []openAIStreamChoice `json:"choices"`
	Usage   *openAIUsage         `json:"usage,omitempty"`
}

type openAIStreamChoice struct {
	Index        int               `json:"index"`
	Delta        openAIStreamDelta `json:"delta"`
	FinishReason *string           `json:"finish_reason"`
}

type openAIStreamDelta struct {
	Role      string                 `json:"role,omitempty"`
	Content   string                 `json:"content,omitempty"`
	ToolCalls []openAIStreamToolCall `json:"tool_calls,omitempty"`
}

type openAIStreamToolCall struct {
	Index    *int                       `json:"index,omitempty"`
	ID       string                     `json:"id,omitempty"`
	Type     string                     `json:"type,omitempty"`
	Function openAIStreamToolCallFields `json:"function,omitempty"`
}

type openAIStreamToolCallFields struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type openAIStreamIterator struct {
	reader      *bufio.Reader
	closer      io.Closer
	done        bool
	accumulated StreamChunk
	toolState   map[int]openAIToolCall
	toolOrder   []int
}

func (it *openAIStreamIterator) Next() (*StreamChunk, error) {
	if it.done {
		return nil, io.EOF
	}

	for {
		line, err := it.reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				it.done = true
				return nil, io.EOF
			}
			return nil, err
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if line == "data: [DONE]" {
			it.done = true
			return nil, io.EOF
		}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		var resp openAIStreamResponse
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &resp); err != nil {
			continue
		}
		if resp.Usage != nil && len(resp.Choices) == 0 {
			return &StreamChunk{
				Usage: ptrUsage(resp.Usage.toUsage()),
			}, nil
		}
		if len(resp.Choices) == 0 {
			continue
		}

		choice := resp.Choices[0]
		chunk := &StreamChunk{
			Content: choice.Delta.Content,
		}
		if chunk.Content != "" {
			it.accumulated.Content += chunk.Content
		}
		if len(choice.Delta.ToolCalls) > 0 {
			it.applyToolCallDelta(choice.Delta.ToolCalls)
			chunk.ToolCalls = make([]ToolCall, len(it.accumulated.ToolCalls))
			copy(chunk.ToolCalls, it.accumulated.ToolCalls)
		}
		if choice.FinishReason != nil {
			chunk.FinishReason = FinishReason(*choice.FinishReason)
		}
		if resp.Usage != nil {
			chunk.Usage = ptrUsage(resp.Usage.toUsage())
		}

		return chunk, nil
	}
}

func (it *openAIStreamIterator) Close() error {
	if it.closer == nil {
		return nil
	}
	return it.closer.Close()
}

func (it *openAIStreamIterator) applyToolCallDelta(calls []openAIStreamToolCall) {
	if it.toolState == nil {
		it.toolState = make(map[int]openAIToolCall)
	}

	for _, delta := range calls {
		idx := len(it.toolOrder)
		if delta.Index != nil {
			idx = *delta.Index
		}

		toolCall, ok := it.toolState[idx]
		if !ok {
			it.toolOrder = append(it.toolOrder, idx)
		}

		if delta.ID != "" {
			toolCall.ID = delta.ID
		}
		if delta.Type != "" {
			toolCall.Type = delta.Type
		}
		if delta.Function.Name != "" {
			toolCall.Function.Name = delta.Function.Name
		}
		if delta.Function.Arguments != "" {
			toolCall.Function.Arguments += delta.Function.Arguments
		}

		it.toolState[idx] = toolCall
	}

	ordered := make([]ToolCall, 0, len(it.toolOrder))
	for _, idx := range it.toolOrder {
		toolCall, ok := it.toolState[idx]
		if !ok {
			continue
		}
		ordered = append(ordered, ToolCall{
			ID:   toolCall.ID,
			Type: toolCall.Type,
			Function: ToolCallFunc{
				Name:      toolCall.Function.Name,
				Arguments: json.RawMessage(toolCall.Function.Arguments),
			},
		})
	}
	it.accumulated.ToolCalls = ordered
}
