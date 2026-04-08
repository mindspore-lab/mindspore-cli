package llm

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

const anthropicDefaultMaxTokens = 64000

type anthropicCodec struct {
	defaultModel string
}

func newAnthropicCodec(defaultModel string) *anthropicCodec {
	return &anthropicCodec{defaultModel: strings.TrimSpace(defaultModel)}
}

func (c *anthropicCodec) encodeRequest(req *CompletionRequest, stream bool) (anthropicMessagesRequest, error) {
	if req == nil {
		return anthropicMessagesRequest{}, fmt.Errorf("request is nil")
	}

	model := strings.TrimSpace(req.Model)
	if model == "" {
		model = c.defaultModel
	}
	if model == "" {
		model = "claude-3-5-haiku-latest"
	}

	system, messages := c.encodeMessages(req.Messages)
	maxTokens := req.MaxTokens
	if maxTokens == nil {
		// Anthropic requests require max_tokens even when mscli has no explicit override.
		defaultMaxTokens := anthropicDefaultMaxTokens
		maxTokens = &defaultMaxTokens
	}

	return anthropicMessagesRequest{
		Model:         model,
		System:        system,
		Messages:      messages,
		Temperature:   req.Temperature,
		MaxTokens:     maxTokens,
		TopP:          req.TopP,
		StopSequences: req.Stop,
		Tools:         c.encodeTools(req.Tools),
		Stream:        stream,
	}, nil
}

func (c *anthropicCodec) encodeMessages(msgs []Message) (string, []anthropicMessage) {
	if len(msgs) == 0 {
		return "", nil
	}

	systemParts := make([]string, 0, len(msgs))
	result := make([]anthropicMessage, 0, len(msgs))
	for _, msg := range msgs {
		switch msg.Role {
		case "system":
			if text := strings.TrimSpace(msg.Content); text != "" {
				systemParts = append(systemParts, text)
			}
		case "tool":
			result = appendToolResultMessage(result, msg)
		default:
			if encoded, ok := c.encodeMessage(msg); ok {
				result = append(result, encoded)
			}
		}
	}

	return strings.Join(systemParts, "\n\n"), result
}

func (c *anthropicCodec) encodeMessage(msg Message) (anthropicMessage, bool) {
	content := make([]anthropicContentBlock, 0, 1+len(msg.ToolCalls))
	if msg.Content != "" {
		content = append(content, anthropicContentBlock{
			Type: "text",
			Text: msg.Content,
		})
	}
	for _, call := range msg.ToolCalls {
		content = append(content, anthropicContentBlock{
			Type:  "tool_use",
			ID:    call.ID,
			Name:  call.Function.Name,
			Input: normalizeRawJSON(call.Function.Arguments),
		})
	}
	if len(content) == 0 {
		return anthropicMessage{}, false
	}

	role := strings.TrimSpace(msg.Role)
	if role == "" {
		role = "user"
	}

	return anthropicMessage{
		Role:    role,
		Content: content,
	}, true
}

func appendToolResultMessage(messages []anthropicMessage, msg Message) []anthropicMessage {
	if strings.TrimSpace(msg.ToolCallID) == "" {
		return messages
	}

	content, isError := decodeAnthropicToolResultPayload(msg.Content)
	block := anthropicContentBlock{
		Type:      "tool_result",
		ToolUseID: msg.ToolCallID,
		Content:   content,
		IsError:   isError,
	}

	return append(messages, anthropicMessage{
		Role:    "user",
		Content: []anthropicContentBlock{block},
	})
}

func decodeAnthropicToolResultPayload(raw string) (string, *bool) {
	var envelope struct {
		Content *string `json:"content"`
		IsError *bool   `json:"is_error"`
	}
	if err := json.Unmarshal([]byte(raw), &envelope); err != nil {
		return raw, nil
	}
	if envelope.IsError == nil {
		return raw, nil
	}
	if envelope.Content != nil {
		return *envelope.Content, envelope.IsError
	}
	return raw, envelope.IsError
}

func (c *anthropicCodec) encodeTools(tools []Tool) []anthropicTool {
	if len(tools) == 0 {
		return nil
	}

	result := make([]anthropicTool, 0, len(tools))
	for _, tool := range tools {
		result = append(result, anthropicTool{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			InputSchema: tool.Function.Parameters,
		})
	}
	return result
}

func (c *anthropicCodec) decodeCompletionResponse(resp anthropicMessagesResponse) *CompletionResponse {
	result := &CompletionResponse{
		ID:           resp.ID,
		Model:        resp.Model,
		FinishReason: mapAnthropicStopReason(resp.StopReason),
		Usage: Usage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
	}

	var text strings.Builder
	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			text.WriteString(block.Text)
		case "tool_use":
			result.ToolCalls = append(result.ToolCalls, ToolCall{
				ID:   block.ID,
				Type: "function",
				Function: ToolCallFunc{
					Name:      block.Name,
					Arguments: normalizeRawJSON(block.Input),
				},
			})
		}
	}
	result.Content = text.String()

	if result.FinishReason == "" && len(result.ToolCalls) > 0 {
		result.FinishReason = FinishToolCalls
	}

	return result
}

func (c *anthropicCodec) newStreamIterator(body io.ReadCloser) StreamIterator {
	return &anthropicStreamIterator{
		reader: bufio.NewReader(body),
		closer: body,
	}
}

type anthropicMessagesRequest struct {
	Model         string             `json:"model"`
	Messages      []anthropicMessage `json:"messages"`
	System        string             `json:"system,omitempty"`
	Temperature   *float32           `json:"temperature,omitempty"`
	MaxTokens     *int               `json:"max_tokens,omitempty"`
	TopP          float32            `json:"top_p,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	Tools         []anthropicTool    `json:"tools,omitempty"`
	Stream        bool               `json:"stream,omitempty"`
}

type anthropicMessage struct {
	Role    string                  `json:"role"`
	Content []anthropicContentBlock `json:"content"`
}

type anthropicContentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   string          `json:"content,omitempty"`
	IsError   *bool           `json:"is_error,omitempty"`
}

type anthropicTool struct {
	Name        string     `json:"name"`
	Description string     `json:"description,omitempty"`
	InputSchema ToolSchema `json:"input_schema"`
}

type anthropicMessagesResponse struct {
	ID         string                  `json:"id"`
	Type       string                  `json:"type,omitempty"`
	Role       string                  `json:"role,omitempty"`
	Model      string                  `json:"model"`
	Content    []anthropicContentBlock `json:"content"`
	StopReason string                  `json:"stop_reason"`
	Usage      anthropicUsage          `json:"usage"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type anthropicStreamIterator struct {
	reader         *bufio.Reader
	closer         io.Closer
	done           bool
	promptTokens   int
	completedCalls []ToolCall
	toolBlocks     map[int]*anthropicStreamToolState
	seenText       bool
	backgroundWork bool
}

type anthropicStreamEvent struct {
	Event string
	Data  []byte
}

type anthropicStreamMessageStartEvent struct {
	Message struct {
		ID    string         `json:"id"`
		Model string         `json:"model"`
		Usage anthropicUsage `json:"usage"`
	} `json:"message"`
}

type anthropicStreamContentBlockStartEvent struct {
	Index        int                   `json:"index"`
	ContentBlock anthropicContentBlock `json:"content_block"`
}

type anthropicStreamContentBlockDeltaEvent struct {
	Index int `json:"index"`
	Delta struct {
		Type        string `json:"type"`
		Text        string `json:"text,omitempty"`
		PartialJSON string `json:"partial_json,omitempty"`
	} `json:"delta"`
}

type anthropicStreamContentBlockStopEvent struct {
	Index int `json:"index"`
}

type anthropicStreamMessageDeltaEvent struct {
	Delta struct {
		StopReason string `json:"stop_reason"`
	} `json:"delta"`
	Usage anthropicUsage `json:"usage"`
}

type anthropicStreamToolState struct {
	ID       string
	Name     string
	Input    json.RawMessage
	Partial  strings.Builder
	Complete bool
}

func (it *anthropicStreamIterator) Next() (*StreamChunk, error) {
	if it.done {
		return nil, io.EOF
	}

	for {
		event, err := it.readEvent()
		if err != nil {
			if err == io.EOF {
				it.done = true
			}
			return nil, err
		}
		if len(event.Data) == 0 && event.Event == "" {
			continue
		}

		switch event.Event {
		case "", "ping":
			continue
		case "error":
			return nil, decodeAnthropicStreamError(event.Data)
		case "message_start":
			var payload anthropicStreamMessageStartEvent
			if err := json.Unmarshal(event.Data, &payload); err != nil {
				return nil, fmt.Errorf("decode message_start: %w", err)
			}
			it.promptTokens = payload.Message.Usage.InputTokens
		case "content_block_start":
			var payload anthropicStreamContentBlockStartEvent
			if err := json.Unmarshal(event.Data, &payload); err != nil {
				return nil, fmt.Errorf("decode content_block_start: %w", err)
			}
			it.startContentBlock(payload)
		case "content_block_delta":
			var payload anthropicStreamContentBlockDeltaEvent
			if err := json.Unmarshal(event.Data, &payload); err != nil {
				return nil, fmt.Errorf("decode content_block_delta: %w", err)
			}
			if chunk := it.applyContentBlockDelta(payload); chunk != nil {
				return chunk, nil
			}
		case "content_block_stop":
			var payload anthropicStreamContentBlockStopEvent
			if err := json.Unmarshal(event.Data, &payload); err != nil {
				return nil, fmt.Errorf("decode content_block_stop: %w", err)
			}
			if chunk, err := it.finishContentBlock(payload.Index); err != nil {
				return nil, err
			} else if chunk != nil {
				return chunk, nil
			}
		case "message_delta":
			var payload anthropicStreamMessageDeltaEvent
			if err := json.Unmarshal(event.Data, &payload); err != nil {
				return nil, fmt.Errorf("decode message_delta: %w", err)
			}
			chunk := &StreamChunk{
				ToolCalls:    it.snapshotCompletedCalls(),
				FinishReason: mapAnthropicStopReason(payload.Delta.StopReason),
				Usage: &Usage{
					PromptTokens:     it.promptTokens,
					CompletionTokens: payload.Usage.OutputTokens,
					TotalTokens:      it.promptTokens + payload.Usage.OutputTokens,
				},
			}
			return chunk, nil
		case "message_stop":
			it.done = true
			return nil, io.EOF
		default:
			continue
		}
	}
}

func (it *anthropicStreamIterator) Close() error {
	if it.closer == nil {
		return nil
	}
	return it.closer.Close()
}

func (it *anthropicStreamIterator) readEvent() (anthropicStreamEvent, error) {
	var event anthropicStreamEvent
	var data bytes.Buffer

	for {
		line, err := it.reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				if event.Event != "" || data.Len() > 0 {
					event.Data = bytes.TrimSpace(data.Bytes())
					return event, nil
				}
			}
			return anthropicStreamEvent{}, err
		}

		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			if event.Event == "" && data.Len() == 0 {
				continue
			}
			event.Data = bytes.TrimSpace(data.Bytes())
			return event, nil
		}

		switch {
		case strings.HasPrefix(line, "event:"):
			event.Event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "data:"):
			if data.Len() > 0 {
				data.WriteByte('\n')
			}
			data.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
}

func (it *anthropicStreamIterator) startContentBlock(payload anthropicStreamContentBlockStartEvent) {
	if payload.ContentBlock.Type != "tool_use" {
		return
	}
	if it.toolBlocks == nil {
		it.toolBlocks = make(map[int]*anthropicStreamToolState)
	}
	it.toolBlocks[payload.Index] = &anthropicStreamToolState{
		ID:    payload.ContentBlock.ID,
		Name:  payload.ContentBlock.Name,
		Input: normalizeRawJSON(payload.ContentBlock.Input),
	}
}

func (it *anthropicStreamIterator) applyContentBlockDelta(payload anthropicStreamContentBlockDeltaEvent) *StreamChunk {
	switch payload.Delta.Type {
	case "text_delta":
		it.seenText = true
		it.backgroundWork = false
		return &StreamChunk{Content: payload.Delta.Text}
	case "input_json_delta":
		state, ok := it.toolBlocks[payload.Index]
		if !ok || state == nil {
			return nil
		}
		state.Partial.WriteString(payload.Delta.PartialJSON)
		if it.seenText && !it.backgroundWork {
			it.backgroundWork = true
			return &StreamChunk{BackgroundWork: true}
		}
	}

	return nil
}

func (it *anthropicStreamIterator) finishContentBlock(index int) (*StreamChunk, error) {
	state, ok := it.toolBlocks[index]
	if !ok || state == nil {
		return nil, nil
	}
	delete(it.toolBlocks, index)

	arguments := normalizeRawJSON(state.Input)
	if partial := state.Partial.String(); partial != "" {
		partialArgs := normalizeRawJSON(json.RawMessage(partial))
		if json.Valid(partialArgs) {
			arguments = partialArgs
		}
	}
	if !json.Valid(arguments) {
		return nil, fmt.Errorf("decode tool_use input for block %d: invalid json", index)
	}

	call := ToolCall{
		ID:   state.ID,
		Type: "function",
		Function: ToolCallFunc{
			Name:      state.Name,
			Arguments: arguments,
		},
	}
	it.completedCalls = append(it.completedCalls, call)

	return &StreamChunk{
		ToolCalls: it.snapshotCompletedCalls(),
	}, nil
}

func (it *anthropicStreamIterator) snapshotCompletedCalls() []ToolCall {
	if len(it.completedCalls) == 0 {
		return nil
	}

	snapshot := make([]ToolCall, len(it.completedCalls))
	copy(snapshot, it.completedCalls)
	return snapshot
}

func decodeAnthropicStreamError(data []byte) error {
	var payload struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(data, &payload); err == nil && payload.Error.Message != "" {
		return fmt.Errorf("stream error (%s): %s", payload.Error.Type, payload.Error.Message)
	}
	return fmt.Errorf("stream error: %s", strings.TrimSpace(string(data)))
}

func mapAnthropicStopReason(reason string) FinishReason {
	switch reason {
	case "end_turn", "stop_sequence":
		return FinishStop
	case "max_tokens":
		return FinishLength
	case "tool_use":
		return FinishToolCalls
	case "":
		return ""
	default:
		return FinishReason(reason)
	}
}

func normalizeRawJSON(raw json.RawMessage) json.RawMessage {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return json.RawMessage(`{}`)
	}
	return json.RawMessage(trimmed)
}
