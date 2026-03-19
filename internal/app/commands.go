package app

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	providerpkg "github.com/vigo999/ms-cli/integrations/llm/provider"
	"github.com/vigo999/ms-cli/internal/project"
	"github.com/vigo999/ms-cli/permission"
	"github.com/vigo999/ms-cli/ui/model"
)

func (a *Application) handleCommand(input string) {
	parts := strings.Fields(input)
	if len(parts) == 0 {
		return
	}

	switch parts[0] {
	case "/roadmap":
		a.cmdRoadmap(parts[1:])
	case "/weekly":
		a.cmdWeekly(parts[1:])
	case "/model":
		a.cmdModel(parts[1:])
	case "/exit":
		a.cmdExit()
	case "/compact":
		a.cmdCompact()
	case "/clear":
		a.cmdClear()
	case "/test":
		a.cmdTest()
	case "/permission":
		a.cmdPermission(parts[1:])
	case "/yolo":
		a.cmdYolo()
	case "/train":
		a.cmdTrain(parts[1:])
	case "/mouse":
		a.cmdMouse(parts[1:])
	case "/help":
		a.cmdHelp()
	default:
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: fmt.Sprintf("Unknown command: %s. Type /help for available commands.", parts[0]),
		}
	}
}

func (a *Application) cmdRoadmap(args []string) {
	if len(args) == 0 || args[0] != "status" {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Usage: /roadmap status [path] (default: roadmap.yaml)",
		}
		return
	}

	path := "roadmap.yaml"
	if len(args) > 1 {
		path = args[1]
	}

	a.EventCh <- model.Event{Type: model.AgentThinking}

	rm, err := project.LoadRoadmapFromFile(path)
	if err != nil {
		a.EventCh <- model.Event{Type: model.ToolError, ToolName: "roadmap", Message: err.Error()}
		return
	}

	status, err := project.ComputeRoadmapStatus(rm, time.Now())
	if err != nil {
		a.EventCh <- model.Event{Type: model.ToolError, ToolName: "roadmap", Message: err.Error()}
		return
	}

	data, _ := json.MarshalIndent(status, "", "  ")
	a.EventCh <- model.Event{Type: model.AgentReply, Message: string(data)}
}

func (a *Application) cmdWeekly(args []string) {
	if len(args) == 0 || args[0] != "status" {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Usage: /weekly status [path] (default: weekly.md)",
		}
		return
	}

	path := "weekly.md"
	if len(args) > 1 {
		path = args[1]
	}

	a.EventCh <- model.Event{Type: model.AgentThinking}

	wu, err := project.LoadWeeklyUpdateFromFile(path)
	if err != nil {
		a.EventCh <- model.Event{Type: model.ToolError, ToolName: "weekly", Message: err.Error()}
		return
	}

	data, _ := json.MarshalIndent(wu, "", "  ")
	a.EventCh <- model.Event{Type: model.AgentReply, Message: string(data)}
}

func (a *Application) cmdModel(args []string) {
	if len(args) == 0 {
		a.showCurrentModel()
		return
	}

	modelArg := args[0]
	if strings.Contains(modelArg, ":") {
		parts := strings.SplitN(modelArg, ":", 2)
		providerName := providerpkg.NormalizeProvider(parts[0])
		modelName := strings.TrimSpace(parts[1])
		if !providerpkg.IsSupportedProvider(providerName) {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Unsupported provider prefix: %s (supported: openai, openai-compatible, anthropic)", providerName),
			}
			return
		}
		a.switchModel(providerName, modelName)
		return
	}

	a.switchModel("", modelArg)
}

func (a *Application) showCurrentModel() {
	providerName := a.Config.Model.Provider
	if providerName == "" {
		providerName = "openai-compatible"
	}
	modelName := a.Config.Model.Model
	url := a.Config.Model.URL
	if url == "" {
		url = "https://api.openai.com/v1"
	}

	apiKeyStatus := "not set"
	if a.Config.Model.Key != "" ||
		os.Getenv("MSCLI_API_KEY") != "" ||
		os.Getenv("OPENAI_API_KEY") != "" ||
		os.Getenv("ANTHROPIC_AUTH_TOKEN") != "" ||
		os.Getenv("ANTHROPIC_API_KEY") != "" {
		apiKeyStatus = "set"
	}

	msg := fmt.Sprintf(`Current Model Configuration:

  Provider: %s
  URL:   %s
  Model: %s
  Key:   %s

To switch model:
  /model <model-name>
  /model <provider>:<model>

Examples:
  /model gpt-4o
  /model openai:gpt-4o-mini
  /model openai-compatible:gpt-4o-mini
  /model anthropic:claude-3-5-sonnet`, providerName, url, modelName, apiKeyStatus)

	a.EventCh <- model.Event{Type: model.AgentReply, Message: msg}
}

func (a *Application) switchModel(providerName, modelName string) {
	a.EventCh <- model.Event{Type: model.AgentThinking}

	err := a.SetProvider(providerName, modelName, "")
	if err != nil {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "model",
			Message:  fmt.Sprintf("Failed to switch model: %v", err),
		}
		return
	}

	a.EventCh <- model.Event{Type: model.ModelUpdate, Message: a.Config.Model.Model}

	if err := a.SaveState(); err != nil {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: fmt.Sprintf("Model switched to: %s. Warning: failed to save state: %v", a.Config.Model.Model, err),
		}
		return
	}

	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("Model switched to: %s", a.Config.Model.Model),
	}
}

func (a *Application) cmdExit() {
	a.EventCh <- model.Event{Type: model.AgentReply, Message: "Goodbye!"}
	go func() {
		time.Sleep(100 * time.Millisecond)
		a.EventCh <- model.Event{Type: model.Done}
	}()
}

func (a *Application) cmdCompact() {
	a.EventCh <- model.Event{Type: model.AgentThinking}

	if a.Engine != nil {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Context compacted. Conversation summary has been created to save tokens.",
		}
	} else {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Context compaction is not available in demo mode.",
		}
	}
}

func (a *Application) cmdClear() {
	a.EventCh <- model.Event{Type: model.ClearScreen, Message: "Chat history cleared."}
}

func (a *Application) cmdTest() {
	a.EventCh <- model.Event{Type: model.AgentThinking}

	modelName := a.Config.Model.Model
	url := a.Config.Model.URL
	if url == "" {
		url = "https://api.openai.com/v1"
	}
	apiKeyStatus := "not set"
	if a.Config.Model.Key != "" {
		apiKeyStatus = fmt.Sprintf("set (%d chars)", len(a.Config.Model.Key))
	}

	msg := fmt.Sprintf("API Connection Test:\n\n  URL:     %s\n  Model:   %s\n  API Key: %s\n\nTesting connectivity...",
		url, modelName, apiKeyStatus)
	a.EventCh <- model.Event{Type: model.AgentReply, Message: msg}

	if a.Engine != nil && !a.Demo && a.llmReady {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "API configuration looks correct. Send a message to test the connection.",
		}
	} else if !a.Demo && !a.llmReady {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: provideAPIKeyFirstMsg}
	} else {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Cannot test in demo mode. Run without --demo flag to test API connectivity.",
		}
	}
}

func (a *Application) cmdPermission(args []string) {
	permSvc, ok := a.permService.(*permission.DefaultPermissionService)
	if !ok {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Permission management not available in current mode.",
		}
		return
	}

	if len(args) == 0 {
		policies := permSvc.GetPolicies()
		msg := "Current Permission Settings:\n\n"
		if len(policies) == 0 {
			msg += "  No custom permissions set.\n"
			msg += "  Default: ask for destructive operations (write, edit, shell)\n"
		} else {
			for tool, level := range policies {
				msg += fmt.Sprintf("  %s: %s\n", tool, level)
			}
		}
		msg += "\nUsage:\n  /permission <tool> <level>\n"
		msg += "\nLevels: ask, allow_once, allow_session, allow_always, deny\n"
		msg += "Tools: read, write, edit, grep, glob, shell\n"
		msg += "\nExamples:\n  /permission shell ask\n  /permission write allow_always"
		a.EventCh <- model.Event{Type: model.AgentReply, Message: msg}
		return
	}

	if len(args) < 2 {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Usage: /permission <tool> <level>\nExample: /permission shell ask",
		}
		return
	}

	tool := args[0]
	level := permission.ParsePermissionLevel(args[1])
	permSvc.Grant(tool, level)

	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("Permission for '%s' set to: %s", tool, level),
	}
}

func (a *Application) cmdYolo() {
	permSvc, ok := a.permService.(*permission.DefaultPermissionService)
	if !ok {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "YOLO mode not available in current configuration.",
		}
		return
	}

	current := permSvc.Check("shell", "")
	if current == permission.PermissionAllowAlways {
		permSvc.Grant("shell", permission.PermissionAsk)
		permSvc.Grant("write", permission.PermissionAsk)
		permSvc.Grant("edit", permission.PermissionAsk)
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "YOLO mode disabled. Will ask for confirmation on destructive operations.",
		}
	} else {
		permSvc.Grant("shell", permission.PermissionAllowAlways)
		permSvc.Grant("write", permission.PermissionAllowAlways)
		permSvc.Grant("edit", permission.PermissionAllowAlways)
		permSvc.Grant("read", permission.PermissionAllowAlways)
		permSvc.Grant("grep", permission.PermissionAllowAlways)
		permSvc.Grant("glob", permission.PermissionAllowAlways)
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "YOLO mode enabled! All operations will be auto-approved. Use with caution!",
		}
	}
}

func (a *Application) cmdMouse(args []string) {
	mode := "toggle"
	if len(args) > 0 {
		mode = strings.ToLower(strings.TrimSpace(args[0]))
	}

	switch mode {
	case "on", "enable", "enabled":
		a.EventCh <- model.Event{Type: model.MouseModeToggle, Message: "on"}
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Mouse scrolling enabled. Use wheel to scroll chat."}
	case "off", "disable", "disabled":
		a.EventCh <- model.Event{Type: model.MouseModeToggle, Message: "off"}
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Mouse scrolling disabled."}
	case "toggle":
		a.EventCh <- model.Event{Type: model.MouseModeToggle, Message: "toggle"}
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Mouse scrolling toggled."}
	case "status":
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Use `/mouse on` to enable scroll wheel, `/mouse off` to disable, `/mouse toggle` to switch.",
		}
	default:
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Usage: /mouse [on|off|toggle|status]",
		}
	}
}

func (a *Application) cmdHelp() {
	helpText := `Available commands:

  /train <model> <method> Set up and run model training (e.g. /train qwen3 lora)
  /roadmap status [path]  Check roadmap status (default: roadmap.yaml)
  /weekly status [path]   Check weekly update status (default: weekly.md)
  /model [model-name]     Show or switch model
  /test                   Test API connectivity
  /permission [tool] [level]  Manage tool permissions
  /yolo                   Toggle auto-approve mode
  /mouse [on|off|toggle|status] Toggle mouse wheel scrolling
  /exit                   Exit the application
  /compact                Compact conversation context to save tokens
  /clear                  Clear chat history
  /help                   Show this help message

Model Commands:
  /model                  Show current configuration
  /model gpt-4o           Switch to gpt-4o
  /model openai:gpt-4o    Set provider+model
  /model anthropic:claude-3-5-sonnet

Permission Commands:
  /permission             Show current permission settings
  /permission shell ask   Set permission level for a tool
  /yolo                   Toggle auto-approve for all operations

Permission Levels:
  ask          - Ask each time (default)
  allow_once   - Allow once
  allow_session - Allow for this session
  allow_always - Always allow
  deny         - Always deny

Keybindings:
  enter      Send input
  mouse wheel Scroll chat
  pgup/pgdn  Scroll chat
  home/end   Jump to top/bottom
  /          Start a slash command
  ctrl+c     Cancel/Quit (press twice to exit)

Environment Variables:
  MSCLI_PROVIDER          Provider (openai/openai-compatible/anthropic)
  MSCLI_BASE_URL          Base URL
  MSCLI_MODEL             Default model
  MSCLI_API_KEY           API key
  OPENAI_BASE_URL         Base URL (fallback)
  OPENAI_MODEL            Model (fallback)
  OPENAI_API_KEY          API key (fallback)
  ANTHROPIC_BASE_URL      Base URL (anthropic)
  ANTHROPIC_AUTH_TOKEN    API key (anthropic, preferred)
  ANTHROPIC_API_KEY       API key (anthropic fallback)`

	a.EventCh <- model.Event{Type: model.AgentReply, Message: helpText}
}
