package provider

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/vigo999/ms-cli/integrations/llm"
)

type openAICodec struct {
	defaultModel string
}

func newOpenAICodec(defaultModel string) *openAICodec {
	return &openAICodec{defaultModel: strings.TrimSpace(defaultModel)}
}

func (c *openAICodec) encodeRequest(req *llm.CompletionRequest, stream bool) (openAIChatCompletionRequest, error) {
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

	return openAIChatCompletionRequest{
		Model:       model,
		Messages:    c.encodeMessages(req.Messages),
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		TopP:        req.TopP,
		Stop:        req.Stop,
		Tools:       c.encodeTools(req.Tools),
		Stream:      stream,
	}, nil
}

func (c *openAICodec) encodeMessages(msgs []llm.Message) []openAIMessage {
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

func (c *openAICodec) encodeToolCalls(calls []llm.ToolCall) []openAIToolCall {
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

func (c *openAICodec) encodeTools(tools []llm.Tool) []openAITool {
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

func (c *openAICodec) decodeCompletionResponse(resp openAIChatCompletionResponse) *llm.CompletionResponse {
	if len(resp.Choices) == 0 {
		return &llm.CompletionResponse{
			ID:    resp.ID,
			Model: resp.Model,
		}
	}

	choice := resp.Choices[0]
	result := &llm.CompletionResponse{
		ID:           resp.ID,
		Model:        resp.Model,
		Content:      choice.Message.Content,
		FinishReason: llm.FinishReason(choice.FinishReason),
		Usage: llm.Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}

	if len(choice.Message.ToolCalls) == 0 {
		return result
	}

	result.ToolCalls = make([]llm.ToolCall, len(choice.Message.ToolCalls))
	for i, tc := range choice.Message.ToolCalls {
		result.ToolCalls[i] = llm.ToolCall{
			ID:   tc.ID,
			Type: tc.Type,
			Function: llm.ToolCallFunc{
				Name:      tc.Function.Name,
				Arguments: json.RawMessage(tc.Function.Arguments),
			},
		}
	}
	result.FinishReason = llm.FinishToolCalls

	return result
}

func (c *openAICodec) newStreamIterator(body io.ReadCloser) llm.StreamIterator {
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
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  llm.ToolSchema `json:"parameters"`
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
	Model       string          `json:"model"`
	Messages    []openAIMessage `json:"messages"`
	Temperature float32         `json:"temperature"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	TopP        float32         `json:"top_p,omitempty"`
	Stop        []string        `json:"stop,omitempty"`
	Tools       []openAITool    `json:"tools,omitempty"`
	Stream      bool            `json:"stream"`
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
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
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
	accumulated llm.StreamChunk
	toolState   map[int]openAIToolCall
	toolOrder   []int
}

func (it *openAIStreamIterator) Next() (*llm.StreamChunk, error) {
	if it.done {
		return nil, io.EOF
	}

	for {
		line, err := it.reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				it.done = true
				if it.accumulated.Content != "" || len(it.accumulated.ToolCalls) > 0 {
					return &llm.StreamChunk{
						Content:   it.accumulated.Content,
						ToolCalls: it.accumulated.ToolCalls,
					}, io.EOF
				}
			}
			return nil, err
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if line == "data: [DONE]" {
			it.done = true
			if it.accumulated.Content != "" || len(it.accumulated.ToolCalls) > 0 {
				return &llm.StreamChunk{
					Content:   it.accumulated.Content,
					ToolCalls: it.accumulated.ToolCalls,
				}, nil
			}
			return nil, io.EOF
		}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		var resp openAIStreamResponse
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &resp); err != nil {
			continue
		}
		if len(resp.Choices) == 0 {
			continue
		}

		choice := resp.Choices[0]
		chunk := &llm.StreamChunk{
			Content: choice.Delta.Content,
		}
		if chunk.Content != "" {
			it.accumulated.Content += chunk.Content
		}
		if len(choice.Delta.ToolCalls) > 0 {
			it.applyToolCallDelta(choice.Delta.ToolCalls)
			chunk.ToolCalls = make([]llm.ToolCall, len(it.accumulated.ToolCalls))
			copy(chunk.ToolCalls, it.accumulated.ToolCalls)
		}
		if choice.FinishReason != nil {
			chunk.FinishReason = llm.FinishReason(*choice.FinishReason)
			it.done = true
		}
		if resp.Usage != nil {
			chunk.Usage = &llm.Usage{
				PromptTokens:     resp.Usage.PromptTokens,
				CompletionTokens: resp.Usage.CompletionTokens,
				TotalTokens:      resp.Usage.TotalTokens,
			}
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

	ordered := make([]llm.ToolCall, 0, len(it.toolOrder))
	for _, idx := range it.toolOrder {
		toolCall, ok := it.toolState[idx]
		if !ok {
			continue
		}
		ordered = append(ordered, llm.ToolCall{
			ID:   toolCall.ID,
			Type: toolCall.Type,
			Function: llm.ToolCallFunc{
				Name:      toolCall.Function.Name,
				Arguments: json.RawMessage(toolCall.Function.Arguments),
			},
		})
	}
	it.accumulated.ToolCalls = ordered
}
