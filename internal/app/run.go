package app

import (
	"context"
	"flag"
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/vigo999/ms-cli/agent/orchestrator"
	"github.com/vigo999/ms-cli/agent/planner"
	"github.com/vigo999/ms-cli/ui"
	"github.com/vigo999/ms-cli/ui/model"
)

const provideAPIKeyFirstMsg = "provide api key first"

// Run parses CLI args, wires dependencies, and starts the application.
func Run(args []string) error {
	fs := flag.NewFlagSet("ms-cli", flag.ContinueOnError)
	demo := fs.Bool("demo", false, "Run in demo mode")
	configPath := fs.String("config", "", "Path to config file")
	url := fs.String("url", "", "OpenAI-compatible base URL")
	modelFlag := fs.String("model", "", "Model name")
	apiKey := fs.String("api-key", "", "API key")

	if err := fs.Parse(args); err != nil {
		return err
	}

	app, err := Wire(BootstrapConfig{
		Demo:       *demo,
		ConfigPath: *configPath,
		URL:        *url,
		Model:      *modelFlag,
		Key:        *apiKey,
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

	if a.Demo {
		return a.runDemo()
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
	if !a.llmReady && !a.Demo {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: provideAPIKeyFirstMsg,
		}
		return
	}

	a.EventCh <- model.Event{Type: model.AgentThinking}

	req := orchestrator.RunRequest{
		ID:          generateTaskID(),
		Description: description,
	}

	events, err := a.Orchestrator.Run(context.Background(), req)
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

	for _, ev := range events {
		if ev.DelayMs > 0 {
			time.Sleep(time.Duration(ev.DelayMs) * time.Millisecond)
		}
		uiEvent := convertRunEvent(ev)
		if uiEvent != nil {
			a.EventCh <- *uiEvent
		}
	}
}

// convertRunEvent maps orchestrator RunEvent → UI model.Event.
func convertRunEvent(ev orchestrator.RunEvent) *model.Event {
	// Map event type string to UI event type.
	// RunEvent types are a superset of loop event types since the engine
	// adapter passes them through.
	typeMap := map[string]model.EventType{
		"AgentReply":    model.AgentReply,
		"AgentThinking": model.AgentThinking,
		"ToolRead":      model.ToolRead,
		"ToolGrep":      model.ToolGrep,
		"ToolGlob":      model.ToolGlob,
		"ToolEdit":      model.ToolEdit,
		"ToolWrite":     model.ToolWrite,
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

func (a *Application) runDemo() error {
	userCh := make(chan string, 8)
	tui := ui.New(a.EventCh, userCh, Version, a.WorkDir, a.RepoURL, "demo-model", a.Config.Context.MaxTokens)
	p := tea.NewProgram(tui, tea.WithAltScreen(), tea.WithMouseCellMotion())

	go a.demoInputLoop(userCh)

	_, err := p.Run()
	close(userCh)
	return err
}

// demoInputLoop handles user input in demo mode. Slash commands are handled
// normally; free-text input is routed to the demo workflow executor which
// plays back a scenario through the real orchestrator dispatch path.
func (a *Application) demoInputLoop(userCh <-chan string) {
	for input := range userCh {
		trimmed := strings.TrimSpace(input)
		if trimmed == "" {
			continue
		}
		if strings.HasPrefix(trimmed, "/") {
			a.handleCommand(trimmed)
			continue
		}
		if a.isTrainMode() {
			a.handleTrainInput(trimmed)
			continue
		}
		go a.runDemoTask(trimmed)
	}
}

// runDemoTask runs a user request through the demo workflow executor.
// Since demo mode has no LLM provider, the orchestrator would fall back to
// agent mode (which has no provider either). Instead, we call the workflow
// executor directly with a synthetic plan, preserving the executor contract.
func (a *Application) runDemoTask(description string) {
	a.EventCh <- model.Event{Type: model.AgentThinking}

	req := orchestrator.RunRequest{
		ID:          generateTaskID(),
		Description: description,
	}

	// Build a synthetic plan pointing to the demo scenario.
	// In a future version with a real planner, this flows through
	// orchestrator.Run() → planner → dispatch → workflow executor.
	plan := planner.Plan{
		Mode:     planner.ModeWorkflow,
		Goal:     description,
		Workflow: "perf_opt",
	}

	events, err := a.Orchestrator.RunWorkflow(req, plan)
	if err != nil {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "DemoExecutor",
			Message:  err.Error(),
		}
		return
	}

	for _, ev := range events {
		if ev.DelayMs > 0 {
			time.Sleep(time.Duration(ev.DelayMs) * time.Millisecond)
		}
		uiEvent := convertRunEvent(ev)
		if uiEvent != nil {
			a.EventCh <- *uiEvent
		}
	}
}
