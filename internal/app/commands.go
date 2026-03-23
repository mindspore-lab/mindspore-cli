package app

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/vigo999/ms-cli/integrations/llm"
	providerpkg "github.com/vigo999/ms-cli/integrations/llm/provider"
	"github.com/vigo999/ms-cli/integrations/skills"
	"github.com/vigo999/ms-cli/permission"
	"github.com/vigo999/ms-cli/ui/model"
)

func (a *Application) handleCommand(input string) {
	parts := strings.Fields(input)
	if len(parts) == 0 {
		return
	}

	switch parts[0] {
	case "/model":
		a.cmdModel(parts[1:])
	case "/exit", "/quit":
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
	case "/project":
		a.cmdProjectInput(strings.TrimSpace(strings.TrimPrefix(input, "/project")))
	case "/login":
		a.cmdLogin(parts[1:])
	case "/report":
		a.cmdReport(parts[1:])
	case "/bugs":
		a.cmdBugs(parts[1:])
	case "/claim":
		a.cmdClaim(parts[1:])
	case "/dock":
		a.cmdDock()
	case "/skill":
		a.cmdSkill(parts[1:])
	case "/help":
		a.cmdHelp()
	default:
		// Check if the command matches a skill name directly (e.g. /pdf → /skill pdf).
		skillName := strings.TrimPrefix(parts[0], "/")
		if a.skillLoader != nil {
			if _, err := a.skillLoader.Load(skillName); err == nil {
				a.cmdSkill(append([]string{skillName}, parts[1:]...))
				return
			}
		}
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: fmt.Sprintf("Unknown command: %s. Type /help for available commands.", parts[0]),
		}
	}
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
	if strings.TrimSpace(a.Config.Model.Key) != "" {
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
			Message: "Context compaction is not available.",
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

	if a.Engine != nil && a.llmReady {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "API configuration looks correct. Send a message to test the connection.",
		}
	} else {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: provideAPIKeyFirstMsg}
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

func (a *Application) cmdSkill(args []string) {
	if a.skillLoader == nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Skills not available."}
		return
	}
	if len(args) == 0 {
		summaries := a.skillLoader.List()
		if len(summaries) == 0 {
			a.EventCh <- model.Event{Type: model.AgentReply, Message: "No skills available."}
			return
		}
		msg := "Available skills:\n\n" + skills.FormatSummaries(summaries) + "\nUsage: /skill <name> [request...]"
		a.EventCh <- model.Event{Type: model.AgentReply, Message: msg}
		return
	}

	skillName := args[0]
	content, err := a.skillLoader.Load(skillName)
	if err != nil {
		a.EventCh <- model.Event{
			Type:    model.ToolError,
			Message: fmt.Sprintf("Failed to load skill %q: %v", skillName, err),
		}
		return
	}

	// Inject a synthetic assistant tool_call + tool result into context so the
	// model sees the skill as already loaded and won't call load_skill again.
	toolCallID := "slash_skill_" + skillName
	argBytes, _ := json.Marshal(map[string]string{"name": skillName})
	assistantMsg := llm.Message{
		Role: "assistant",
		ToolCalls: []llm.ToolCall{
			{
				ID:   toolCallID,
				Type: "function",
				Function: llm.ToolCallFunc{
					Name:      "load_skill",
					Arguments: json.RawMessage(argBytes),
				},
			},
		},
	}
	_ = a.ctxManager.AddMessage(assistantMsg)
	_ = a.ctxManager.AddMessage(llm.NewToolMessage(toolCallID, content))
	a.EventCh <- model.Event{
		Type:     model.ToolSkill,
		ToolName: "load_skill",
		Message:  skillName,
		Summary:  fmt.Sprintf("loaded skill: %s", skillName),
	}

	userRequest := ""
	if len(args) > 1 {
		userRequest = strings.Join(args[1:], " ")
	}
	if strings.TrimSpace(userRequest) == "" {
		return
	}
	go a.runTask(userRequest)
}

func (a *Application) cmdHelp() {
	helpText := `Available commands:

  /skill [name] [request] Load and run a skill (e.g. /skill pdf extract text from report.pdf)
  /train <model> <method> Start train workflow (e.g. /train qwen3 lora)
  /train <action>         Control active train HUD (start, stop, analyze, apply fix, retry, view diff, exit)
  /project [status]        Show project status snapshot (server + git status)
  /project add <section> "<title>" [--owner o] [--progress p]  Add a task
  /project update <section> <id> [--title t] [--owner o] [--progress p] [--status s]  Update a task
  /project rm <section> <id>  Remove a task
  /login <token>          Log in to the bug server
  /report <title>         Report a new bug
  /bugs [status]          List bugs (optional status filter: open, doing)
  /claim <id>             Claim a bug as your lead
  /dock                   Show bug dashboard (open count, ready, recent)
  /model [model-name]     Show or switch model
  /test                   Test API connectivity
  /permission [tool] [level]  Manage tool permissions
  /yolo                   Toggle auto-approve mode
  /exit                   Exit the application
  /quit                   Exit the application
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
  MSCLI_TEMPERATURE       Temperature
  MSCLI_MAX_TOKENS        Max completion tokens
  MSCLI_TIMEOUT           Request timeout seconds`

	a.EventCh <- model.Event{Type: model.AgentReply, Message: helpText}
}
