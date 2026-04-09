package app

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	"github.com/mindspore-lab/mindspore-cli/agent/loop"
	"github.com/mindspore-lab/mindspore-cli/configs"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/internal/bugs"
	issuepkg "github.com/mindspore-lab/mindspore-cli/internal/issues"
	projectpkg "github.com/mindspore-lab/mindspore-cli/internal/project"
	"github.com/mindspore-lab/mindspore-cli/permission"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func (a *Application) handleCommand(input string) {
	cmd, ok := splitRawCommand(input)
	if !ok {
		return
	}
	args := strings.Fields(cmd.Remainder)

	switch cmd.Name {
	case "/model":
		a.cmdModel(args)
	case "/exit":
		a.cmdExit()
	case "/compact":
		a.cmdCompact()
	case "/clear":
		a.cmdClear()
	case "/permissions":
		a.cmdPermissions(nil)
	case "/yolo":
		a.cmdYolo()
	case "/train":
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Coming soon."}
		return
	case "/project":
		a.cmdProjectInput(cmd.Remainder)
	case "/login":
		a.cmdLogin(args)
	case "/feedback":
		expanded, err := a.expandReportInput(cmd.Remainder)
		if err != nil {
			a.emitInputExpansionError(err)
			return
		}
		a.cmdUnifiedReport(expanded)
	case "/issues":
		a.cmdIssues(args)
	case "/__issue_detail":
		a.cmdIssueDetail(args)
	case "/__issue_note":
		a.cmdIssueNoteInput(cmd.Remainder)
	case "/__issue_claim":
		a.cmdIssueClaim(args)
	case "/diagnose":
		expanded, err := a.expandIssueCommandInput(cmd.Remainder)
		if err != nil {
			a.emitInputExpansionError(err)
			return
		}
		a.cmdDiagnose(expanded)
	case "/fix":
		expanded, err := a.expandIssueCommandInput(cmd.Remainder)
		if err != nil {
			a.emitInputExpansionError(err)
			return
		}
		a.cmdFix(expanded)
	case "/migrate":
		expanded, err := a.expandIssueCommandInput(cmd.Remainder)
		if err != nil {
			a.emitInputExpansionError(err)
			return
		}
		a.cmdMigrate(expanded)
	case "/bugs":
		a.cmdBugs(args)
	case "/__bug_detail":
		a.cmdBugDetail(args)
	case "/claim":
		a.cmdClaim(args)
	case "/close":
		a.cmdClose(args)
	case "/now":
		a.cmdDock()
	case "/skill":
		if err := a.handleRawSkillCommand(cmd.Remainder); err != nil {
			a.emitInputExpansionError(err)
		}
	case "/skill-add":
		a.cmdSkillAddInput(cmd.Remainder)
	default:
		if cmd.Name == "/permission" {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: "Command `/permission` has been removed. Use `/permissions`.",
			}
			return
		}
		if handled, err := a.handleSkillAliasCommand(cmd.Name, cmd.Remainder); handled {
			if err != nil {
				a.emitInputExpansionError(err)
			}
			return
		}
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: fmt.Sprintf("Unknown command: %s. Type / to see available commands.", cmd.Name),
		}
	}
}

func (a *Application) handleRawSkillCommand(rawInput string) error {
	if strings.TrimSpace(rawInput) == "" {
		a.cmdSkill(nil)
		return nil
	}

	skillName, request := splitFirstToken(rawInput)
	if skillName == "" {
		a.cmdSkill(nil)
		return nil
	}

	if request != "" {
		expanded, err := a.expandInputText(request)
		if err != nil {
			return err
		}
		request = expanded
	}

	a.runLoadedSkillCommand(skillName, request)
	return nil
}

func (a *Application) handleSkillAliasCommand(commandName, rawRemainder string) (bool, error) {
	if a.skillLoader == nil {
		return false, nil
	}

	skillName := strings.TrimPrefix(strings.TrimSpace(commandName), "/")
	if skillName == "" {
		return false, nil
	}
	if _, err := a.skillLoader.Load(skillName); err != nil {
		return false, nil
	}

	request := strings.TrimSpace(rawRemainder)
	if request != "" {
		expanded, err := a.expandInputText(request)
		if err != nil {
			return true, err
		}
		request = expanded
	}

	a.runLoadedSkillCommand(skillName, request)
	return true, nil
}

func (a *Application) cmdModel(args []string) {
	if len(args) == 0 {
		a.emitModelBrowser()
		return
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: "/model no longer accepts arguments. Use /model and choose from the UI.",
	}
}

func (a *Application) cmdConnect(args []string) {
	if len(args) == 0 {
		a.emitModelBrowser()
		return
	}

	providerID := strings.TrimSpace(args[0])
	apiKey := ""
	if len(args) > 1 {
		apiKey = strings.TrimSpace(strings.Join(args[1:], " "))
	}

	a.EventCh <- model.Event{Type: model.AgentThinking}
	state, err := a.connectProvider(providerID, apiKey)
	if err != nil {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "connect",
			Message:  fmt.Sprintf("Failed to connect provider: %v", err),
		}
		return
	}
	a.emitModelBrowserWithState(state, parseConnectProviderSelection(providerID).ProviderID)
}

// applyPreset applies a preset with the given API key. It saves the current
// model config for later restoration, sets the provider, and updates the
// active preset ID. Returns an error if SetProvider fails.
func (a *Application) applyPreset(preset builtinModelPreset, apiKey string) error {
	if a.modelBeforePreset == nil {
		a.modelBeforePreset = copyModelConfig(a.Config.Model)
	}
	previous := a.Config.Model
	a.Config.Model.URL = preset.BaseURL
	if err := a.SetProvider(preset.Provider, preset.Model, apiKey); err != nil {
		a.Config.Model = previous
		return err
	}
	a.activeModelPresetID = preset.ID
	return nil
}

func (a *Application) switchToBuiltinModelPreset(preset builtinModelPreset) {
	a.EventCh <- model.Event{Type: model.AgentThinking}

	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()

	apiKey, err := a.resolveModelPresetAPIKey(ctx, preset)
	if err != nil {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "model",
			Message:  fmt.Sprintf("Failed to switch preset: %v", err),
		}
		return
	}

	if err := a.applyPreset(preset, apiKey); err != nil {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "model",
			Message:  fmt.Sprintf("Failed to switch preset: %v", err),
		}
		return
	}

	a.EventCh <- model.Event{
		Type:     model.ModelUpdate,
		Message:  a.Config.Model.Model,
		Provider: runtimeProviderDisplayLabel(a.Config.Model.Provider),
		CtxMax:   a.Config.Context.Window,
	}

	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("Model switched to preset: %s", preset.Label),
	}
}

func (a *Application) restoreModelConfigFromPreset() {
	if strings.TrimSpace(a.activeModelPresetID) == "" || a.modelBeforePreset == nil {
		return
	}
	a.Config.Model = *copyModelConfig(*a.modelBeforePreset)
	a.modelBeforePreset = nil
	a.activeModelPresetID = ""
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

	a.EventCh <- model.Event{
		Type:     model.ModelUpdate,
		Message:  a.Config.Model.Model,
		Provider: runtimeProviderDisplayLabel(a.Config.Model.Provider),
		CtxMax:   a.Config.Context.Window,
	}

	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("Model switched to: %s", a.Config.Model.Model),
	}
}

func (a *Application) cmdSelectModel(args []string) {
	if len(args) == 0 {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "model",
			Message:  "model selection requires provider:model",
		}
		return
	}

	parts := strings.SplitN(strings.TrimSpace(args[0]), ":", 2)
	if len(parts) != 2 || strings.TrimSpace(parts[0]) == "" || strings.TrimSpace(parts[1]) == "" {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "model",
			Message:  "model selection requires provider:model",
		}
		return
	}

	a.EventCh <- model.Event{Type: model.AgentThinking}
	result, err := a.activateLogicalModelSelection(parts[0], parts[1])
	if err != nil {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "model",
			Message:  fmt.Sprintf("Failed to switch model: %v", err),
		}
		return
	}

	a.EventCh <- model.Event{Type: model.ModelBrowserClose}
	a.EventCh <- model.Event{
		Type:     model.ModelUpdate,
		Message:  a.Config.Model.Model,
		Provider: result.ProviderLabel,
		CtxMax:   a.Config.Context.Window,
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("Model switched to: %s", a.Config.Model.Model),
	}
}

func (a *Application) cmdDeleteProvider(args []string) {
	if len(args) == 0 {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "model",
			Message:  "provider deletion requires provider id",
		}
		return
	}

	providerID := strings.TrimSpace(args[0])
	a.EventCh <- model.Event{Type: model.AgentThinking}
	result, err := a.deleteConnectedProvider(providerID)
	if err != nil {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "model",
			Message:  fmt.Sprintf("Failed to delete provider: %v", err),
		}
		return
	}

	if result.Fallback != nil {
		a.EventCh <- model.Event{
			Type:     model.ModelUpdate,
			Message:  a.Config.Model.Model,
			Provider: runtimeProviderDisplayLabel(result.Fallback.ProviderID),
			CtxMax:   a.Config.Context.Window,
		}
	} else if result.Cleared {
		a.EventCh <- model.Event{
			Type:     model.ModelUpdate,
			Message:  "No model (/model to configure)",
			Provider: "",
			CtxMax:   a.Config.Context.Window,
		}
	}

	if state, err := a.loadProviderWorkflowState(providerCatalogLoadCacheFirst); err == nil {
		a.emitModelBrowserWithState(state, "")
	}
}

func (a *Application) clearActiveLogicalModel() error {
	previousModel := a.Config.Model.Model
	a.restoreModelConfigFromPreset()
	a.Config.Model.Provider = ""
	a.Config.Model.Model = ""
	a.Config.Model.Key = ""
	a.Config.Model.URL = ""
	configs.RefreshModelTokenDefaults(a.Config, previousModel)
	a.provider = nil
	a.llmReady = false

	systemPrompt := ""
	if a.ctxManager != nil {
		if msg := a.ctxManager.GetSystemPrompt(); msg != nil {
			systemPrompt = msg.Content
		}
	}

	engineCfg := newEngineConfig(a.Config, systemPrompt)
	newEngine := loop.NewEngine(engineCfg, nil, a.toolRegistry)
	if a.ctxManager != nil {
		newEngine.SetContextManager(a.ctxManager)
	}
	newEngine.SetLLMDebugDumper(a.llmDebugDumper)
	newEngine.SetPermissionService(a.permService)
	newEngine.SetTrajectoryRecorder(newTrajectoryRecorder(a.session, a.ctxManager, a.noteLiveLLMActivity))
	a.Engine = newEngine
	return nil
}

func (a *Application) cmdModelSetup(args []string) {
	if len(args) < 2 {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "model",
			Message:  "model setup requires preset ID and token",
		}
		return
	}
	presetID := args[0]
	token := strings.TrimSpace(args[1])

	preset, ok := resolveBuiltinModelPreset(presetID)
	if !ok {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "model",
			Message:  fmt.Sprintf("unknown preset: %s", presetID),
		}
		return
	}

	serverURL := strings.TrimRight(a.Config.Server.URL, "/")
	if serverURL == "" {
		a.EventCh <- model.Event{
			Type:    model.ModelSetupTokenError,
			Message: "server URL not set. export MSCLI_SERVER_URL first.",
		}
		return
	}

	a.EventCh <- model.Event{Type: model.AgentThinking}

	// Step 1: Verify token and get user info (same as /login).
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()

	userName, userRole, err := a.verifyUserToken(ctx, serverURL, token)
	if err != nil {
		a.EventCh <- model.Event{
			Type:    model.ModelSetupTokenError,
			Message: fmt.Sprintf("Login failed: %v", err),
		}
		return
	}

	// Step 2: Save credentials (so issue/bug services work too).
	cred := &credentials{
		ServerURL: serverURL,
		Token:     token,
		User:      userName,
		Role:      userRole,
	}
	if err := saveCredentials(cred); err != nil {
		a.emitToolError("config", "login ok but failed to save credentials: %v", err)
	}
	a.bugService = bugs.NewService(bugs.NewRemoteStore(serverURL, token))
	a.issueService = issuepkg.NewService(issuepkg.NewRemoteStore(serverURL, token))
	a.projectService = projectpkg.NewService(projectpkg.NewRemoteStore(serverURL, token))
	a.issueUser = userName
	a.issueRole = userRole

	// Step 3: Fetch LLM API key from server for the preset.
	apiKey, err := a.resolveModelPresetAPIKey(ctx, preset)
	if err != nil {
		a.EventCh <- model.Event{
			Type:    model.ModelSetupTokenError,
			Message: fmt.Sprintf("Failed to fetch model credential: %v", err),
		}
		return
	}

	// Step 4: Apply preset with fetched API key.
	if err := a.applyPreset(preset, apiKey); err != nil {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "model",
			Message:  fmt.Sprintf("Failed to apply preset: %v", err),
		}
		a.EventCh <- model.Event{
			Type:    model.ModelSetupTokenError,
			Message: fmt.Sprintf("Failed: %v", err),
		}
		return
	}

	// Step 5: Save model mode to config.json (token is in credentials.json).
	if err := saveAppConfig(&appConfig{
		ModelMode:     modelModeMSCLIProvided,
		ModelPresetID: preset.ID,
		ModelToken:    token,
	}); err != nil {
		a.emitToolError("config", "model applied but failed to save config: %v", err)
	}

	// Step 6: Emit UI updates.
	a.EventCh <- model.Event{Type: model.IssueUserUpdate, Message: userName}
	a.EventCh <- model.Event{
		Type:     model.ModelUpdate,
		Message:  a.Config.Model.Model,
		Provider: runtimeProviderDisplayLabel(a.Config.Model.Provider),
		CtxMax:   a.Config.Context.Window,
	}
	a.EventCh <- model.Event{Type: model.ModelSetupClose}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("Logged in as %s. Model configured: %s", userName, preset.Label),
	}
}

// verifyUserToken verifies a user token against the mscli server.
func (a *Application) verifyUserToken(ctx context.Context, serverURL, token string) (user, role string, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, serverURL+"/me", nil)
	if err != nil {
		return "", "", fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+token)

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", "", fmt.Errorf("cannot reach server: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", "", fmt.Errorf("%s", body)
	}

	var me struct {
		User string `json:"user"`
		Role string `json:"role"`
	}
	if err := json.Unmarshal(body, &me); err != nil {
		return "", "", fmt.Errorf("invalid response: %w", err)
	}
	return me.User, me.Role, nil
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

	if a.ctxManager == nil {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Context compaction is not available.",
		}
		return
	}

	before := a.ctxManager.TokenUsage()
	if err := a.ctxManager.Compact(); err != nil {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "context",
			Message:  fmt.Sprintf("Failed to compact context: %v", err),
		}
		return
	}
	after := a.ctxManager.TokenUsage()
	if err := a.persistSessionSnapshot(); err != nil {
		a.emitToolError("session", "Failed to persist session snapshot: %v", err)
	}
	a.emitTokenUsageSnapshot()

	message := fmt.Sprintf("Context compacted: %d -> %d tokens.", before.Current, after.Current)
	if after.Current >= before.Current {
		message = "Context compaction had nothing to remove."
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: message,
	}
}

func (a *Application) cmdClear() {
	a.EventCh <- model.Event{Type: model.ClearScreen, Message: "Chat history cleared."}
}

func (a *Application) cmdPermissions(args []string) {
	_ = args // /permissions is single-entry: ignore all trailing arguments.
	permSvc, ok := a.permService.(*permission.DefaultPermissionService)
	if !ok {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Permission management not available in current mode.",
		}
		return
	}

	a.EventCh <- model.Event{
		Type:        model.PermissionsView,
		Permissions: a.buildPermissionsViewData(permSvc),
	}
}

func (a *Application) cmdPermissionsInternal(args []string) {
	permSvc, ok := a.permService.(*permission.DefaultPermissionService)
	if !ok {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Permission management not available in current mode.",
		}
		return
	}
	if len(args) == 0 {
		a.cmdPermissions(nil)
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
	if len(args) >= 2 {
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
		return
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: "internal permissions command requires action",
	}
}

func (a *Application) buildPermissionsViewData(permSvc *permission.DefaultPermissionService) *model.PermissionsViewData {
	data := &model.PermissionsViewData{
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
	return data
}

func (a *Application) cmdPermissionsAdd(permSvc *permission.DefaultPermissionService, args []string) {
	if len(args) >= 2 {
		level := permission.ParsePermissionLevel(args[0])
		scope, hasScope, rest := parsePermissionScopeArgs(args[1:])
		rule := strings.TrimSpace(strings.Join(rest, " "))
		if strings.Contains(rule, "(") || strings.HasPrefix(strings.ToLower(rule), "mcp__") {
			if err := permSvc.AddRule(rule, level); err != nil {
				a.EventCh <- model.Event{
					Type:    model.AgentReply,
					Message: fmt.Sprintf("Failed to add rule: %v", err),
				}
				return
			}
			if hasScope {
				path, err := a.savePermissionRuleToScope(rule, level, scope)
				if err != nil {
					a.EventCh <- model.Event{
						Type:    model.AgentReply,
						Message: fmt.Sprintf("Added rule for this session, but failed to save settings file: %v", err),
					}
					return
				}
				a.EventCh <- model.Event{
					Type:    model.AgentReply,
					Message: fmt.Sprintf("Added rule: %s => %s (saved to %s)", rule, level, path),
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
			Type:    model.AgentReply,
			Message: "invalid internal permissions add command",
		}
		return
	}

	targetType := strings.ToLower(strings.TrimSpace(args[0]))
	target := strings.TrimSpace(strings.Join(args[1:len(args)-1], " "))
	level := permission.ParsePermissionLevel(args[len(args)-1])

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
		if strings.HasSuffix(cmd, "*") {
			return fmt.Sprintf("Bash(%s)", cmd)
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
			Type:    model.AgentReply,
			Message: "invalid internal permissions remove command",
		}
		return
	}

	targetType := strings.ToLower(strings.TrimSpace(args[0]))
	target := strings.TrimSpace(strings.Join(args[1:], " "))

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
		permSvc.Grant("load_skill", permission.PermissionAsk)
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
		permSvc.Grant("load_skill", permission.PermissionAllowAlways)
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
		a.emitAvailableSkills(true)
		return
	}

	skillName := args[0]
	userRequest := strings.TrimSpace(strings.Join(args[1:], " "))
	a.runLoadedSkillCommand(skillName, userRequest)
}

func (a *Application) runLoadedSkillCommand(skillName, userRequest string) {
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
	if err := a.addContextMessages(assistantMsg, llm.NewToolMessage(toolCallID, content)); err != nil {
		a.emitToolError("load_skill", "Failed to activate skill %q: %v", skillName, err)
		return
	}
	if a.session != nil {
		if err := a.session.AppendSkillActivation(skillName); err != nil {
			a.emitToolError("session", "Failed to persist skill activation: %v", err)
		}
		if err := a.persistSessionSnapshot(); err != nil {
			a.emitToolError("session", "Failed to persist session snapshot: %v", err)
		}
	}
	a.EventCh <- model.Event{
		Type:     model.ToolSkill,
		ToolName: "load_skill",
		Message:  skillName,
		Summary:  fmt.Sprintf("loaded skill: %s", skillName),
	}

	if userRequest == "" {
		userRequest = defaultSkillRequest(skillName)
	}
	go a.runTask(userRequest)
}

func defaultSkillRequest(skillName string) string {
	return fmt.Sprintf(
		`The %q skill is already loaded. Start following that skill now using the current workspace and conversation context. Begin with the first concrete step immediately, keep gathering evidence with tools, and only stop to ask the user if the skill cannot proceed without missing information.`,
		skillName,
	)
}
