package app

import (
	"fmt"
	"strings"
	"sync"

	"github.com/vigo999/ms-cli/ui/model"
)

type permissionDecision struct {
	granted  bool
	remember bool
}

// PermissionPromptUI bridges permission prompts into the TUI input/event flow.
type PermissionPromptUI struct {
	mu      sync.Mutex
	pending *pendingPermissionRequest
	eventCh chan<- model.Event
}

type pendingPermissionRequest struct {
	wait chan permissionDecision
	tool string
	path string
}

func NewPermissionPromptUI(eventCh chan<- model.Event) *PermissionPromptUI {
	return &PermissionPromptUI{
		eventCh: eventCh,
	}
}

// RequestPermission asks the user and blocks until a decision is provided.
func (p *PermissionPromptUI) RequestPermission(tool, action, path string) (bool, bool, error) {
	p.mu.Lock()
	if p.pending != nil {
		p.mu.Unlock()
		return false, false, fmt.Errorf("permission request already pending")
	}
	req := &pendingPermissionRequest{
		wait: make(chan permissionDecision, 1),
		tool: strings.TrimSpace(tool),
		path: strings.TrimSpace(path),
	}
	p.pending = req
	p.mu.Unlock()

	msg := p.promptMessage(tool, action, path)
	p.eventCh <- model.Event{
		Type:       model.PermissionPrompt,
		Message:    msg,
		Permission: p.promptData(tool, action, path),
	}

	decision := <-req.wait
	return decision.granted, decision.remember, nil
}

// HandleInput consumes pending permission replies from user input.
// Returns true if input was consumed by permission flow.
func (p *PermissionPromptUI) HandleInput(input string) bool {
	input = strings.ToLower(strings.TrimSpace(input))

	p.mu.Lock()
	req := p.pending
	p.mu.Unlock()
	if req == nil {
		return false
	}

	resolve := func(d permissionDecision) bool {
		p.mu.Lock()
		current := p.pending
		if current != nil {
			p.pending = nil
		}
		p.mu.Unlock()
		if current == nil {
			return true
		}
		current.wait <- d
		return true
	}

	switch input {
	case "1", "y", "yes":
		return resolve(permissionDecision{granted: true, remember: false})
	case "2", "a", "allow", "allow_session", "session":
		return resolve(permissionDecision{granted: true, remember: true})
	case "3", "n", "no", "esc", "escape":
		return resolve(permissionDecision{granted: false, remember: false})
	default:
		p.eventCh <- model.Event{
			Type:       model.PermissionPrompt,
			Message:    "Please choose 1, 2, or 3.",
			Permission: p.promptData(req.tool, "", req.path),
		}
		return true
	}
}

func (p *PermissionPromptUI) promptMessage(tool, action, path string) string {
	tool = strings.TrimSpace(tool)
	action = strings.TrimSpace(action)
	path = strings.TrimSpace(path)

	if isEditTool(tool) && path != "" {
		return fmt.Sprintf("Do you want to make this edit to %s?\n  1. Yes\n  2. Yes, allow all edits during this session (shift+tab)\n  3. No\n\nEsc to cancel", path)
	}

	msg := fmt.Sprintf("Do you want to allow tool `%s`?", tool)
	if action != "" {
		msg += fmt.Sprintf("\naction: %s", action)
	}
	if path != "" {
		msg += fmt.Sprintf("\npath: %s", path)
	}
	msg += "\n  1. Yes\n  2. Yes, don't ask again for this session\n  3. No\n\nEsc to cancel"
	return msg
}

func isEditTool(tool string) bool {
	switch strings.TrimSpace(strings.ToLower(tool)) {
	case "edit", "write":
		return true
	default:
		return false
	}
}

func (p *PermissionPromptUI) promptData(tool, action, path string) *model.PermissionPromptData {
	tool = strings.TrimSpace(tool)
	action = strings.TrimSpace(action)
	path = strings.TrimSpace(path)

	if isEditTool(tool) && path != "" {
		return &model.PermissionPromptData{
			Title:   "Confirm Edit",
			Message: fmt.Sprintf("Do you want to make this edit to %s?", path),
			Options: []model.PermissionOption{
				{Input: "1", Label: "1. Yes"},
				{Input: "2", Label: "2. Yes, allow all edits during this session"},
				{Input: "3", Label: "3. No"},
			},
			DefaultIndex: 0,
		}
	}

	msg := fmt.Sprintf("Do you want to allow tool `%s`?", tool)
	if action != "" {
		msg += fmt.Sprintf("\naction: %s", action)
	}
	if path != "" {
		msg += fmt.Sprintf("\npath: %s", path)
	}
	return &model.PermissionPromptData{
		Title:   "Permission required",
		Message: msg,
		Options: []model.PermissionOption{
			{Input: "1", Label: "1. Yes"},
			{Input: "2", Label: "2. Yes, don't ask again for this session"},
			{Input: "3", Label: "3. No"},
		},
		DefaultIndex: 0,
	}
}
