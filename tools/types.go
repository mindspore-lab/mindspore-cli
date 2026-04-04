// Package tools provides executable tools for the agent.
package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
)

// Tool is the interface for executable tools.
type Tool interface {
	// Name returns the tool name (English, no spaces).
	Name() string

	// Description returns the tool description for LLM understanding.
	Description() string

	// Schema returns the JSON schema for tool parameters.
	Schema() llm.ToolSchema

	// Execute executes the tool with the given parameters.
	Execute(ctx context.Context, params json.RawMessage) (*Result, error)
}

// StreamEventType describes an incremental execution update from a streaming tool.
type StreamEventType string

const (
	StreamEventStarted StreamEventType = "started"
	StreamEventOutput  StreamEventType = "output"
)

// StreamEvent is an incremental execution update emitted while a tool runs.
type StreamEvent struct {
	Type    StreamEventType
	Message string
	Summary string
}

// StreamingTool is an optional extension for tools that can emit live updates.
type StreamingTool interface {
	ExecuteStream(ctx context.Context, params json.RawMessage, emit func(StreamEvent)) (*Result, error)
}

// Result is the result of a tool execution.
type Result struct {
	Content string // Main output content
	Summary string // Summary for UI display (e.g., "42 lines", "5 matches")
	Error   error  // Execution error
}

// StringResult creates a result with just content.
func StringResult(content string) *Result {
	return &Result{Content: content}
}

// StringResultWithSummary creates a result with content and summary.
func StringResultWithSummary(content, summary string) *Result {
	return &Result{Content: content, Summary: summary}
}

// ErrorResult creates an error result.
func ErrorResult(err error) *Result {
	return &Result{Error: err}
}

// ErrorResultf creates an error result with formatted message.
func ErrorResultf(format string, args ...any) *Result {
	return &Result{Error: fmt.Errorf(format, args...)}
}

// ParseParams parses the raw JSON parameters into a struct.
func ParseParams(data json.RawMessage, v any) error {
	if err := json.Unmarshal(data, v); err != nil {
		return fmt.Errorf("parse params: %w", err)
	}
	return nil
}
