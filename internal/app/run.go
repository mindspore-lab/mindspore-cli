package app

import (
	"context"
	"flag"
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/vigo999/ms-cli/agent/loop"
	"github.com/vigo999/ms-cli/ui"
	"github.com/vigo999/ms-cli/ui/model"
)

const provideAPIKeyFirstMsg = "provide api key first"

// Run parses CLI args, wires dependencies, and starts the application.
func Run(args []string) error {
	fs := flag.NewFlagSet("ms-cli", flag.ContinueOnError)
	url := fs.String("url", "", "OpenAI-compatible base URL")
	modelFlag := fs.String("model", "", "Model name")
	apiKey := fs.String("api-key", "", "API key")

	if err := fs.Parse(args); err != nil {
		return err
	}

	app, err := Wire(BootstrapConfig{
		URL:   *url,
		Model: *modelFlag,
		Key:   *apiKey,
	})
	if err != nil {
		return err
	}

	return app.run()
}

// run starts the TUI.
func (a *Application) run() error {
	if closer, ok := a.traceWriter.(interface{ Close() error }); ok {
		defer closer.Close()
	}
	return a.runReal()
}

func (a *Application) runReal() error {
	userCh := make(chan string, 8)
	tui := ui.New(a.EventCh, userCh, Version, a.WorkDir, a.RepoURL, a.Config.Model.Model, a.Config.Context.MaxTokens)
	p := tea.NewProgram(tui, tea.WithAltScreen(), tea.WithMouseCellMotion())

	go a.inputLoop(userCh)

	_, err := p.Run()
	close(userCh)
	return err
}

func (a *Application) inputLoop(userCh <-chan string) {
	for input := range userCh {
		a.processInput(input)
	}
}

func (a *Application) processInput(input string) {
	trimmed := strings.TrimSpace(input)
	if trimmed == "" {
		return
	}

	if strings.HasPrefix(trimmed, "/") {
		a.handleCommand(trimmed)
		return
	}

	// Train mode intercepts non-slash input
	if a.isTrainMode() {
		a.handleTrainInput(trimmed)
		return
	}

	go a.runTask(trimmed)
}

func (a *Application) runTask(description string) {
	if !a.llmReady {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: provideAPIKeyFirstMsg,
		}
		return
	}

	task := loop.Task{
		ID:          generateTaskID(),
		Description: description,
	}

	err := a.Engine.RunWithContextStream(context.Background(), task, func(ev loop.Event) {
		uiEvent := convertLoopEvent(ev)
		if uiEvent != nil {
			a.EventCh <- *uiEvent
		}
	})
	if err != nil {
		errMsg := err.Error()
		if strings.Contains(errMsg, "timeout") || strings.Contains(errMsg, "deadline") {
			errMsg = fmt.Sprintf("%s\n\nTip: The request timed out. Try:\n  1. Run /compact to reduce context size\n  2. Start a new conversation with /clear\n  3. Increase timeout in config (model.timeout_sec)", errMsg)
		}
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "Engine",
			Message:  errMsg,
		}
		return
	}
}

// convertLoopEvent maps loop.Event -> UI model.Event.
func convertLoopEvent(ev loop.Event) *model.Event {
	typeMap := map[string]model.EventType{
		"ToolCallStart": model.ToolCallStart,
		"AgentReply":    model.AgentReply,
		"AgentThinking": model.AgentThinking,
		"ToolRead":      model.ToolRead,
		"ToolGrep":      model.ToolGrep,
		"ToolGlob":      model.ToolGlob,
		"ToolEdit":      model.ToolEdit,
		"ToolWrite":     model.ToolWrite,
		"ToolSkill":     model.ToolSkill,
		"ToolError":     model.ToolError,
		"CmdStarted":    model.CmdStarted,
		"AnalysisReady": model.AnalysisReady,
		"TokenUpdate":   model.TokenUpdate,
		"TaskFailed":    model.ToolError,
	}

	uiType, ok := typeMap[ev.Type]
	if !ok {
		if ev.Type == "TaskCompleted" {
			return nil
		}
		if ev.Message != "" {
			return &model.Event{Type: model.AgentReply, Message: ev.Message}
		}
		return nil
	}

	return &model.Event{
		Type:       uiType,
		Message:    ev.Message,
		ToolName:   ev.ToolName,
		Summary:    ev.Summary,
		CtxUsed:    ev.CtxUsed,
		CtxMax:     ev.CtxMax,
		TokensUsed: ev.TokensUsed,
	}
}

func generateTaskID() string {
	return time.Now().Format("20060102-150405-000")
}
