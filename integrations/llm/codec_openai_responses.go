package llm

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

type openAIResponsesCodec struct {
	defaultModel string
}

func newOpenAIResponsesCodec(defaultModel string) *openAIResponsesCodec {
	return &openAIResponsesCodec{defaultModel: strings.TrimSpace(defaultModel)}
}

func (c *openAIResponsesCodec) encodeRequest(req *CompletionRequest, stream bool, previousResponseID string) (openAIResponsesRequest, error) {
	if req == nil {
		return openAIResponsesRequest{}, fmt.Errorf("request is nil")
	}

	model := strings.TrimSpace(req.Model)
	if model == "" {
		model = c.defaultModel
	}
	if model == "" {
		model = "gpt-4o-mini"
	}

	instructions, input := c.encodeMessages(req.Messages, strings.TrimSpace(previousResponseID) != "")
	encodedTools := c.encodeTools(req.Tools)

	return openAIResponsesRequest{
		Model:              model,
		Input:              input,
		Instructions:       instructions,
		Tools:              encodedTools,
		Temperature:        req.Temperature,
		MaxOutputTokens:    req.MaxTokens,
		TopP:               req.TopP,
		PreviousResponseID: previousResponseID,
		Stream:             stream,
	}, nil
}

func (c *openAIResponsesCodec) encodeMessages(msgs []Message, allowOrphanToolOutputs bool) (string, []openAIResponsesInputItem) {
	if len(msgs) == 0 {
		return "", nil
	}

	systemParts := make([]string, 0, len(msgs))
	items := make([]openAIResponsesInputItem, 0, len(msgs))
	seenCallIDs := make(map[string]struct{})

	for _, msg := range msgs {
		switch msg.Role {
		case "system":
			if text := strings.TrimSpace(msg.Content); text != "" {
				systemParts = append(systemParts, text)
			}
		case "tool":
			callID := strings.TrimSpace(msg.ToolCallID)
			if callID == "" {
				continue
			}
			if !allowOrphanToolOutputs {
				if _, ok := seenCallIDs[callID]; !ok {
					continue
				}
			}
			items = append(items, openAIResponsesInputItem{
				Type:   "function_call_output",
				CallID: callID,
				Output: msg.Content,
			})
		default:
			if text := strings.TrimSpace(msg.Content); text != "" {
				items = append(items, openAIResponsesInputItem{
					Type:    "message",
					Role:    msg.Role,
					Content: msg.Content,
				})
			}
			for _, call := range msg.ToolCalls {
				callID := strings.TrimSpace(call.ID)
				if callID == "" {
					continue
				}
				seenCallIDs[callID] = struct{}{}
				items = append(items, openAIResponsesInputItem{
					Type:      "function_call",
					CallID:    callID,
					Name:      call.Function.Name,
					Arguments: string(call.Function.Arguments),
				})
			}
		}
	}

	return strings.Join(systemParts, "\n\n"), items
}

func (c *openAIResponsesCodec) encodeTools(tools []Tool) []openAIResponsesTool {
	if len(tools) == 0 {
		return nil
	}

	encoded := make([]openAIResponsesTool, 0, len(tools))
	for _, tool := range tools {
		encoded = append(encoded, openAIResponsesTool{
			Type:        tool.Type,
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			Parameters:  tool.Function.Parameters,
			Strict:      true,
		})
	}
	return encoded
}

func (c *openAIResponsesCodec) decodeCompletionResponse(resp openAIResponsesResponse) *CompletionResponse {
	result := &CompletionResponse{
		ID:    resp.ID,
		Model: resp.Model,
		Usage: Usage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}

	if result.Usage.TotalTokens == 0 {
		result.Usage.TotalTokens = result.Usage.PromptTokens + result.Usage.CompletionTokens
	}

	var text strings.Builder
	for _, item := range resp.Output {
		switch item.Type {
		case "message":
			for _, content := range item.Content {
				switch content.Type {
				case "output_text":
					text.WriteString(content.Text)
				case "refusal":
					text.WriteString(content.Refusal)
				}
			}
		case "function_call":
			callID := strings.TrimSpace(item.CallID)
			if callID == "" {
				callID = item.ID
			}
			result.ToolCalls = append(result.ToolCalls, ToolCall{
				ID:   callID,
				Type: "function",
				Function: ToolCallFunc{
					Name:      item.Name,
					Arguments: normalizeRawJSON(json.RawMessage(item.Arguments)),
				},
			})
		}
	}

	result.Content = text.String()
	if len(result.ToolCalls) > 0 {
		result.FinishReason = FinishToolCalls
	} else {
		result.FinishReason = FinishStop
	}

	return result
}

func (c *openAIResponsesCodec) newStreamIterator(body io.ReadCloser) StreamIterator {
	return &openAIResponsesStreamIterator{
		reader: bufio.NewReader(body),
		closer: body,
	}
}

type openAIResponsesRequest struct {
	Model              string                     `json:"model"`
	Input              []openAIResponsesInputItem `json:"input,omitempty"`
	Instructions       string                     `json:"instructions,omitempty"`
	Tools              []openAIResponsesTool      `json:"tools,omitempty"`
	Temperature        *float32                   `json:"temperature,omitempty"`
	MaxOutputTokens    *int                       `json:"max_output_tokens,omitempty"`
	TopP               float32                    `json:"top_p,omitempty"`
	PreviousResponseID string                     `json:"previous_response_id,omitempty"`
	Stream             bool                       `json:"stream,omitempty"`
}

type openAIResponsesInputItem struct {
	Type      string `json:"type,omitempty"`
	Role      string `json:"role,omitempty"`
	Content   any    `json:"content,omitempty"`
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	Output    string `json:"output,omitempty"`
}

type openAIResponsesTool struct {
	Type        string     `json:"type"`
	Name        string     `json:"name"`
	Description string     `json:"description,omitempty"`
	Parameters  ToolSchema `json:"parameters"`
	Strict      bool       `json:"strict"`
}

type openAIResponsesResponseEnvelope struct {
	Response openAIResponsesResponse `json:"response"`
	ID       string                  `json:"id"`
	Model    string                  `json:"model"`
	Output   []openAIResponsesItem   `json:"output"`
	Usage    openAIResponsesUsage    `json:"usage"`
}

func (e openAIResponsesResponseEnvelope) ResponseOrSelf() openAIResponsesResponse {
	if e.Response.ID != "" || len(e.Response.Output) > 0 {
		return e.Response
	}
	return openAIResponsesResponse{
		ID:     e.ID,
		Model:  e.Model,
		Output: e.Output,
		Usage:  e.Usage,
	}
}

type openAIResponsesResponse struct {
	ID     string                `json:"id"`
	Model  string                `json:"model"`
	Output []openAIResponsesItem `json:"output"`
	Usage  openAIResponsesUsage  `json:"usage"`
}

type openAIResponsesItem struct {
	ID        string                       `json:"id"`
	Type      string                       `json:"type"`
	Role      string                       `json:"role,omitempty"`
	Status    string                       `json:"status,omitempty"`
	Name      string                       `json:"name,omitempty"`
	CallID    string                       `json:"call_id,omitempty"`
	Arguments string                       `json:"arguments,omitempty"`
	Content   []openAIResponsesMessagePart `json:"content,omitempty"`
}

type openAIResponsesMessagePart struct {
	Type    string `json:"type"`
	Text    string `json:"text,omitempty"`
	Refusal string `json:"refusal,omitempty"`
}

type openAIResponsesUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

type openAIResponsesStreamEvent struct {
	Type        string                  `json:"type"`
	Delta       string                  `json:"delta,omitempty"`
	OutputIndex int                     `json:"output_index,omitempty"`
	ItemID      string                  `json:"item_id,omitempty"`
	Item        openAIResponsesItem     `json:"item,omitempty"`
	Response    openAIResponsesResponse `json:"response,omitempty"`
}

type openAIResponsesStreamIterator struct {
	reader         *bufio.Reader
	closer         io.Closer
	done           bool
	accumulated    StreamChunk
	text           strings.Builder
	toolState      map[int]ToolCall
	toolOrder      []int
	seenText       bool
	backgroundWork bool
}

func (it *openAIResponsesStreamIterator) Next() (*StreamChunk, error) {
	if it.done {
		return nil, io.EOF
	}

	for {
		line, err := it.reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				it.done = true
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

		var event openAIResponsesStreamEvent
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
			continue
		}

		switch event.Type {
		case "response.output_text.delta":
			it.text.WriteString(event.Delta)
			it.seenText = true
			it.backgroundWork = false
			return &StreamChunk{Content: event.Delta}, nil
		case "response.output_item.added":
			if event.Item.Type == "function_call" {
				it.applyToolCallItem(event.OutputIndex, event.Item, false)
			}
		case "response.function_call_arguments.delta":
			it.applyToolCallArgumentsDelta(event.OutputIndex, event.Delta)
			if it.seenText && !it.backgroundWork {
				it.backgroundWork = true
				return &StreamChunk{BackgroundWork: true}, nil
			}
		case "response.output_item.done":
			if event.Item.Type == "function_call" {
				it.applyToolCallItem(event.OutputIndex, event.Item, true)
				return &StreamChunk{ToolCalls: it.snapshotToolCalls()}, nil
			}
		case "response.completed":
			it.done = true
			resp := event.Response
			final := newOpenAIResponsesCodec("").decodeCompletionResponse(resp)
			toolCalls := it.snapshotToolCalls()
			if len(toolCalls) == 0 {
				toolCalls = final.ToolCalls
			}
			chunk := &StreamChunk{
				ID:           resp.ID,
				Model:        resp.Model,
				ToolCalls:    toolCalls,
				FinishReason: final.FinishReason,
				Usage: &Usage{
					PromptTokens:     resp.Usage.InputTokens,
					CompletionTokens: resp.Usage.OutputTokens,
					TotalTokens:      resp.Usage.TotalTokens,
				},
			}
			if chunk.Usage.TotalTokens == 0 {
				chunk.Usage.TotalTokens = chunk.Usage.PromptTokens + chunk.Usage.CompletionTokens
			}
			if it.text.Len() == 0 {
				chunk.Content = final.Content
			}
			return chunk, nil
		}
	}
}

func (it *openAIResponsesStreamIterator) Close() error {
	if it.closer == nil {
		return nil
	}
	return it.closer.Close()
}

func (it *openAIResponsesStreamIterator) applyToolCallItem(index int, item openAIResponsesItem, replace bool) {
	if it.toolState == nil {
		it.toolState = make(map[int]ToolCall)
	}

	if _, ok := it.toolState[index]; !ok {
		it.toolOrder = append(it.toolOrder, index)
	}

	call := it.toolState[index]
	callID := strings.TrimSpace(item.CallID)
	if callID == "" {
		callID = strings.TrimSpace(item.ID)
	}
	if callID != "" {
		call.ID = callID
	}
	call.Type = "function"
	if item.Name != "" {
		call.Function.Name = item.Name
	}
	if item.Arguments != "" {
		if replace {
			call.Function.Arguments = normalizeRawJSON(json.RawMessage(item.Arguments))
		} else {
			call.Function.Arguments = json.RawMessage(item.Arguments)
		}
	}
	it.toolState[index] = call
}

func (it *openAIResponsesStreamIterator) applyToolCallArgumentsDelta(index int, delta string) {
	if it.toolState == nil {
		it.toolState = make(map[int]ToolCall)
	}
	call := it.toolState[index]
	call.Type = "function"
	call.Function.Arguments = append(call.Function.Arguments, delta...)
	it.toolState[index] = call
}

func (it *openAIResponsesStreamIterator) snapshotToolCalls() []ToolCall {
	if len(it.toolOrder) == 0 {
		return nil
	}

	snapshot := make([]ToolCall, 0, len(it.toolOrder))
	for _, idx := range it.toolOrder {
		call, ok := it.toolState[idx]
		if !ok {
			continue
		}
		call.Function.Arguments = normalizeRawJSON(call.Function.Arguments)
		snapshot = append(snapshot, call)
	}
	return snapshot
}
