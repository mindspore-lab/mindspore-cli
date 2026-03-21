package app

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/vigo999/ms-cli/integrations/llm"
	providerpkg "github.com/vigo999/ms-cli/integrations/llm/provider"
	"github.com/vigo999/ms-cli/integrations/skills"
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
	case "/permissions":
		a.cmdPermissions(parts[1:])
	case "/yolo":
		a.cmdYolo()
	case "/train":
		a.cmdTrain(parts[1:])
	case "/project":
		a.cmdProject(parts[1:])
	case "/mouse":
		a.cmdMouse(parts[1:])
	case "/skill":
		a.cmdSkill(parts[1:])
	case "/help":
		a.cmdHelp()
	default:
		if parts[0] == "/permission" {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: "Command `/permission` has been removed. Use `/permissions`.",
			}
			return
		}
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

func (a *Application) cmdPermissions(args []string) {
	permSvc, ok := a.permService.(*permission.DefaultPermissionService)
	if !ok {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Permission management not available in current mode.",
		}
		return
	}

	if len(args) == 0 {
		mode := permSvc.Mode()

		data := &model.PermissionsViewData{
			Mode:        mode.String(),
			RuleSources: map[string]string{},
		}

		for _, rv := range permSvc.GetRuleViews() {
			entry := strings.TrimSpace(rv.Rule)
			if entry == "" {
				continue
			}
			switch rv.Level {
			case permission.PermissionAllowAlways, permission.PermissionAllowSession, permission.PermissionAllowOnce:
				data.Allow = append(data.Allow, entry)
			case permission.PermissionDeny:
				data.Deny = append(data.Deny, entry)
			default:
				data.Ask = append(data.Ask, entry)
			}
			if strings.TrimSpace(rv.Source) != "" {
				data.RuleSources[entry] = rv.Source
			}
		}
		for _, pp := range permSvc.GetPathPolicies() {
			if strings.HasSuffix(pp.Pattern, "/*") &&
				(pp.Level == permission.PermissionAllowAlways ||
					pp.Level == permission.PermissionAllowSession ||
					pp.Level == permission.PermissionAllowOnce) {
				dir := strings.TrimSuffix(pp.Pattern, "/*")
				if strings.TrimSpace(dir) != "" {
					data.Workspace = append(data.Workspace, dir)
				}
				continue
			}
			entry := fmt.Sprintf("edit(%s)", pp.Pattern)
			switch pp.Level {
			case permission.PermissionAllowAlways, permission.PermissionAllowSession, permission.PermissionAllowOnce:
				data.Allow = append(data.Allow, entry)
			case permission.PermissionDeny:
				data.Deny = append(data.Deny, entry)
			default:
				data.Ask = append(data.Ask, entry)
			}
			data.RuleSources[entry] = "runtime"
		}
		if wd := strings.TrimSpace(a.WorkDir); wd != "" {
			data.Workspace = append(data.Workspace, wd)
		}

		a.EventCh <- model.Event{
			Type:        model.PermissionsView,
			Permissions: data,
		}
		return
	}

	if len(args) >= 1 && strings.EqualFold(args[0], "add") {
		a.cmdPermissionsAdd(permSvc, args[1:])
		return
	}

	if len(args) >= 1 && strings.EqualFold(args[0], "remove") {
		a.cmdPermissionsRemove(permSvc, args[1:])
		return
	}

	if len(args) >= 1 && strings.EqualFold(args[0], "workspace") {
		a.cmdPermissionsWorkspace(permSvc, args[1:])
		return
	}

	if strings.EqualFold(args[0], "mode") {
		if len(args) == 1 {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Current permission mode: %s", permSvc.Mode()),
			}
			return
		}

		mode := permission.ParsePermissionMode(args[1])
		if !permission.IsValidPermissionMode(mode) {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: "Invalid mode. Use: default, acceptEdits, plan, dontAsk, bypassPermissions",
			}
			return
		}
		if err := permSvc.SetMode(mode); err != nil {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Failed to set permission mode: %v", err),
			}
			return
		}
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: fmt.Sprintf("Permission mode set to: %s", mode),
		}
		return
	}

	if len(args) < 2 {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Usage: /permissions <tool> <level>\nExample: /permissions shell ask",
		}
		return
	}

	tool := args[0]
	level := permission.ParsePermissionLevel(args[1])
	if err := permSvc.AddRule(tool, level); err != nil {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: fmt.Sprintf("Failed to set permission for '%s': %v", tool, err),
		}
		return
	}

	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("Permission for '%s' set to: %s", tool, level),
	}
}

func (a *Application) cmdPermissionsAdd(permSvc *permission.DefaultPermissionService, args []string) {
	if len(args) >= 2 {
		level := permission.ParsePermissionLevel(args[0])
		rule := strings.TrimSpace(strings.Join(args[1:], " "))
		if strings.Contains(rule, "(") || strings.HasPrefix(strings.ToLower(rule), "mcp__") {
			if err := permSvc.AddRule(rule, level); err != nil {
				a.EventCh <- model.Event{
					Type:    model.AgentReply,
					Message: fmt.Sprintf("Failed to add rule: %v", err),
				}
				return
			}
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Added rule: %s => %s", rule, level),
			}
			return
		}
	}

	if len(args) < 3 {
		a.EventCh <- model.Event{
			Type: model.AgentReply,
			Message: "Usage: /permissions add <tool|command|path> <target> <level>\n" +
				"Examples:\n" +
				"  /permissions add tool shell ask\n" +
				"  /permissions add command git allow_session\n" +
				"  /permissions add path \"*.md\" deny\n" +
				"  /permissions add allow Bash(npm test *)",
		}
		return
	}

	targetType := strings.ToLower(strings.TrimSpace(args[0]))
	target := strings.TrimSpace(args[1])
	level := permission.ParsePermissionLevel(args[2])

	switch targetType {
	case "tool":
		if err := permSvc.AddRule(permissionRuleForLegacyTarget("tool", target), level); err != nil {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Failed to add tool rule: %v", err),
			}
			return
		}
	case "command":
		if err := permSvc.AddRule(permissionRuleForLegacyTarget("command", target), level); err != nil {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Failed to add command rule: %v", err),
			}
			return
		}
	case "path":
		if err := permSvc.AddRule(permissionRuleForLegacyTarget("path", target), level); err != nil {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Failed to add path rule: %v", err),
			}
			return
		}
	default:
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Invalid rule type. Use: tool, command, path",
		}
		return
	}

	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("Added %s rule: %s => %s", targetType, target, level),
	}
}

func permissionRuleForLegacyTarget(targetType, target string) string {
	switch strings.ToLower(strings.TrimSpace(targetType)) {
	case "tool":
		return target
	case "command":
		cmd := strings.TrimSpace(target)
		if cmd == "" {
			return "Bash(*)"
		}
		return fmt.Sprintf("Bash(%s *)", cmd)
	case "path":
		p := strings.TrimSpace(target)
		if filepath.IsAbs(p) {
			p = "//" + strings.TrimPrefix(filepath.ToSlash(p), "/")
		}
		return fmt.Sprintf("Edit(%s)", p)
	default:
		return strings.TrimSpace(target)
	}
}

func (a *Application) cmdPermissionsRemove(permSvc *permission.DefaultPermissionService, args []string) {
	if len(args) >= 1 {
		rule := strings.TrimSpace(strings.Join(args, " "))
		if strings.Contains(rule, "(") || strings.HasPrefix(strings.ToLower(rule), "mcp__") {
			ok, err := permSvc.RemoveRule(rule)
			if err != nil {
				a.EventCh <- model.Event{
					Type:    model.AgentReply,
					Message: fmt.Sprintf("Failed to remove rule: %v", err),
				}
				return
			}
			if !ok {
				a.EventCh <- model.Event{
					Type:    model.AgentReply,
					Message: fmt.Sprintf("Rule not found: %s", rule),
				}
				return
			}
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Removed rule: %s", rule),
			}
			return
		}
	}

	if len(args) < 2 {
		a.EventCh <- model.Event{
			Type: model.AgentReply,
			Message: "Usage: /permissions remove <tool|command|path> <target>\n" +
				"Examples:\n" +
				"  /permissions remove tool shell\n" +
				"  /permissions remove command git\n" +
				"  /permissions remove path \"*.md\"\n" +
				"  /permissions remove Bash(npm test *)",
		}
		return
	}

	targetType := strings.ToLower(strings.TrimSpace(args[0]))
	target := strings.TrimSpace(args[1])

	switch targetType {
	case "tool":
		ok, err := permSvc.RemoveRule(permissionRuleForLegacyTarget("tool", target))
		if err != nil {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Failed to remove tool rule: %v", err),
			}
			return
		}
		if !ok {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Rule not found: %s", target),
			}
			return
		}
	case "command":
		ok, err := permSvc.RemoveRule(permissionRuleForLegacyTarget("command", target))
		if err != nil {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Failed to remove command rule: %v", err),
			}
			return
		}
		if !ok {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Rule not found: %s", target),
			}
			return
		}
	case "path":
		ok, err := permSvc.RemoveRule(permissionRuleForLegacyTarget("path", target))
		if err != nil {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Failed to remove path rule: %v", err),
			}
			return
		}
		if !ok {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Rule not found: %s", target),
			}
			return
		}
	default:
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Invalid rule type. Use: tool, command, path",
		}
		return
	}

	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("Removed %s rule: %s", targetType, target),
	}
}

func (a *Application) cmdPermissionsWorkspace(permSvc *permission.DefaultPermissionService, args []string) {
	if len(args) < 2 {
		a.EventCh <- model.Event{
			Type: model.AgentReply,
			Message: "Usage: /permissions workspace <add|remove> <directory>\n" +
				"Examples:\n" +
				"  /permissions workspace add /path/to/project\n" +
				"  /permissions workspace remove /path/to/project",
		}
		return
	}

	action := strings.ToLower(strings.TrimSpace(args[0]))
	dir := strings.TrimSpace(args[1])
	if dir == "" {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Directory cannot be empty.",
		}
		return
	}

	pattern := strings.TrimRight(dir, "/") + "/*"
	switch action {
	case "add":
		if err := permSvc.AddRule(permissionRuleForLegacyTarget("path", pattern), permission.PermissionAllowAlways); err != nil {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Failed to add workspace directory: %v", err),
			}
			return
		}
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: fmt.Sprintf("Workspace directory added: %s", dir),
		}
	case "remove":
		ok, err := permSvc.RemoveRule(permissionRuleForLegacyTarget("path", pattern))
		if err != nil {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Failed to remove workspace directory: %v", err),
			}
			return
		}
		if !ok {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Workspace directory rule not found: %s", dir),
			}
			return
		}
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: fmt.Sprintf("Workspace directory removed: %s", dir),
		}
	default:
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Invalid workspace action. Use: add, remove",
		}
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
  /project [status]        Show a formatted project status snapshot in the chat stream
  /roadmap status [path]  Check roadmap status (default: roadmap.yaml)
  /weekly status [path]   Check weekly update status (default: weekly.md)
  /model [model-name]     Show or switch model
  /test                   Test API connectivity
  /permissions [tool] [level] Manage tool permissions
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
  /permissions                  Show current permission settings
  /permissions mode plan        Set permission mode
  /permissions add tool shell ask
                               Add tool/command/path rule
  /permissions remove tool shell
                               Remove tool/command/path rule
  /permissions workspace add /path/to/dir
                               Add workspace directory rule
  /permissions shell ask        Set permission level for a tool
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
