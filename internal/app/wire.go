package app

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	agentctx "github.com/mindspore-lab/mindspore-cli/agent/context"
	"github.com/mindspore-lab/mindspore-cli/agent/loop"
	"github.com/mindspore-lab/mindspore-cli/agent/session"
	"github.com/mindspore-lab/mindspore-cli/configs"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/integrations/skills"
	issuepkg "github.com/mindspore-lab/mindspore-cli/internal/issues"
	projectpkg "github.com/mindspore-lab/mindspore-cli/internal/project"
	itrain "github.com/mindspore-lab/mindspore-cli/internal/train"
	"github.com/mindspore-lab/mindspore-cli/internal/version"
	"github.com/mindspore-lab/mindspore-cli/permission"
	rshell "github.com/mindspore-lab/mindspore-cli/runtime/shell"
	"github.com/mindspore-lab/mindspore-cli/tools"
	"github.com/mindspore-lab/mindspore-cli/tools/fs"
	"github.com/mindspore-lab/mindspore-cli/tools/shell"
	skillstool "github.com/mindspore-lab/mindspore-cli/tools/skills"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
	wtrain "github.com/mindspore-lab/mindspore-cli/workflow/train"
)

var errAPIKeyNotFound = errors.New("api key not found")

var buildProvider = func(resolved llm.ResolvedConfig) (llm.Provider, error) {
	return llm.DefaultManager().Build(resolved)
}

var Version = "MindSpore CLI. " + version.Version

// Application is the top-level composition container.
type Application struct {
	Engine                  *loop.Engine
	EventCh                 chan model.Event
	llmReady                bool
	llmDebugDumper          *llm.DebugDumper
	WorkDir                 string
	RepoURL                 string
	Config                  *configs.Config
	provider                llm.Provider
	toolRegistry            *tools.Registry
	ctxManager              *agentctx.Manager
	permService             permission.PermissionService
	permissionUI            *PermissionPromptUI
	permissionSettingsIssue *permissionSettingsIssue
	session                 *session.Session
	replayBacklog           []model.Event
	replayTimeline          []session.ReplayFrame
	deferHistoryReplay      bool
	historyReplayStarted    atomic.Bool
	replayOnly              bool
	replaySpeed             float64
	sessionLLMActivity      atomic.Bool
	sessionStoreReady       atomic.Bool

	// Skills
	skillLoader   *skills.Loader
	skillsHomeDir string
	startupOnce   sync.Once

	// Issue tracking
	issueService *issuepkg.Service
	issueUser    string
	issueRole    string

	// Project tracking
	projectService *projectpkg.Service

	// Foreground chat task state
	taskRunID    uint64
	taskCancels  map[uint64]context.CancelFunc
	replayCancel context.CancelFunc
	taskMu       sync.Mutex

	// Model preset runtime override state.
	activeModelPresetID string
	modelBeforePreset   *configs.ModelConfig
	needsSetupPopup     bool
	savedModelToken     string

	// Train mode state
	trainMode       bool
	trainPhase      string // "setup","ready","running","failed","analyzing","fixing","evaluating","drift_detected","completed","stopped"
	trainReq        *itrain.Request
	trainReqs       map[string]itrain.Request
	trainBootstrap  map[string]*bootstrapRunState
	trainCurrentRun string
	trainCancel     context.CancelFunc
	trainIssueType  string // "runtime", "accuracy", or ""
	trainRunID      uint64
	trainTasks      map[uint64]struct{}
	trainController *wtrain.Controller
	pendingTrain    *pendingTrainStart
	trainMu         sync.RWMutex
}

// BootstrapConfig holds bootstrap configuration.
type BootstrapConfig struct {
	URL             string
	Model           string
	Key             string
	Debug           bool
	Resume          bool
	ResumeSessionID string
	Replay          bool
	ReplaySessionID string
	ReplaySpeed     float64
}

// Wire builds and returns the Application.
func Wire(cfg BootstrapConfig) (*Application, error) {
	workDir, err := os.Getwd()
	if err != nil {
		workDir = "."
	}
	workDir, _ = filepath.Abs(workDir)

	eventCh := make(chan model.Event, 64)

	config, err := configs.LoadWithEnv()
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	if cfg.URL != "" {
		config.Model.URL = cfg.URL
	}
	previousModel := config.Model.Model
	if cfg.Model != "" {
		config.Model.Model = cfg.Model
	}
	if cfg.Key != "" {
		config.Model.Key = cfg.Key
	}
	configs.RefreshModelTokenDefaults(config, previousModel)

	var provider llm.Provider
	llmReady := true
	resolveOpts := llm.ResolveOptions{
		PreferConfigAPIKey:  strings.TrimSpace(cfg.Key) != "",
		PreferConfigBaseURL: strings.TrimSpace(cfg.URL) != "",
	}
	provider, err = initProvider(config.Model, resolveOpts)
	if err != nil {
		if errors.Is(err, errAPIKeyNotFound) {
			llmReady = false
			provider = nil
		} else {
			return nil, fmt.Errorf("init provider: %w", err)
		}
	}

	// If LLM is not ready (missing API key), try detecting saved model config.
	var needsSetupPopup bool
	var activePresetID string
	var savedModelToken string
	if !llmReady {
		mode, appCfg := detectModelMode()
		switch mode {
		case modelModeMSCLIProvided:
			savedModelToken = appCfg.ModelToken
			if preset, ok := resolveBuiltinModelPreset(appCfg.ModelPresetID); ok {
				config.Model.URL = preset.BaseURL
				config.Model.Provider = preset.Provider
				config.Model.Model = preset.Model
				// Always re-fetch the API key from the server instead of
				// reusing the cached one — it may have been rotated.
				apiKey := appCfg.ModelToken
				if freshKey, fetchErr := fetchPresetAPIKey(preset); fetchErr == nil {
					apiKey = freshKey
					// Update the saved config with the fresh key.
					appCfg.ModelToken = freshKey
					_ = saveAppConfig(appCfg)
				}
				config.Model.Key = apiKey
				savedModelToken = apiKey
				configs.RefreshModelTokenDefaults(config, previousModel)
				provider, err = initProvider(config.Model, llm.ResolveOptions{PreferConfigAPIKey: true})
				if err == nil {
					llmReady = true
					activePresetID = preset.ID
				}
			}
		case modelModeOwnEnv:
			// Env vars were applied but init still failed for another reason.
		default:
			needsSetupPopup = true
		}
	}

	toolRegistry := initTools(config, workDir)

	// Skills: embedded skills are extracted next to the executable,
	// user-installed skills in ~/.mscli/skills/ override them,
	// project-local skills in .mscli/skills/ override both.
	homeDir, _ := os.UserHomeDir()
	execSkillsDir := ""
	if ep, err := os.Executable(); err == nil {
		execSkillsDir = filepath.Join(filepath.Dir(ep), ".mscli", "skills")
		_ = skills.ExtractBuiltin(execSkillsDir)
	}
	skillLoader := skills.NewLoader(
		execSkillsDir,
		filepath.Join(homeDir, ".mscli", "skills"),
		filepath.Join(workDir, ".mscli", "skills"),
	)
	toolRegistry.MustRegister(skillstool.NewLoadSkillTool(skillLoader))

	registerSkillCommands(skillLoader.List())

	managerCfg := agentctx.DefaultManagerConfig()
	managerCfg.ContextWindow = config.Context.Window
	managerCfg.ReserveTokens = config.Context.ReserveTokens
	managerCfg.CompactionThreshold = config.Context.CompactionThreshold
	ctxManager := agentctx.NewManager(managerCfg)

	// Build system prompt: base + skill summaries.
	systemPrompt := buildSystemPrompt(skillLoader.List())

	var (
		runtimeSession *session.Session
		replayBacklog  []model.Event
		replayTimeline []session.ReplayFrame
	)
	if cfg.Resume || cfg.Replay {
		sessionID := cfg.ResumeSessionID
		if cfg.Replay {
			sessionID = cfg.ReplaySessionID
		}
		if strings.TrimSpace(sessionID) != "" {
			targetLabel := "session"
			if cfg.Replay && looksLikeTrajectoryPath(sessionID) {
				targetLabel = "trajectory"
			}
			if cfg.Replay && looksLikeTrajectoryPath(sessionID) {
				runtimeSession, err = session.LoadReplayPath(sessionID)
			} else {
				runtimeSession, err = session.LoadByID(workDir, sessionID)
			}
			if err != nil {
				return nil, fmt.Errorf("load %s %s: %w", targetLabel, sessionID, err)
			}
		} else {
			runtimeSession, err = session.LoadLatest(workDir)
			if err != nil {
				return nil, fmt.Errorf("load latest session: %w", err)
			}
		}
		systemPrompt, restoredMessages := runtimeSession.RestoreContext()
		ctxManager.SetSystemPrompt(systemPrompt)
		ctxManager.SetNonSystemMessages(restoredMessages)
		if cfg.Replay {
			replayTimeline = runtimeSession.PlaybackTimeline()
			if metaWorkDir := strings.TrimSpace(runtimeSession.Meta().WorkDir); metaWorkDir != "" {
				workDir = metaWorkDir
			}
		} else {
			replayBacklog = runtimeSession.ReplayEvents()
		}
	} else {
		runtimeSession, err = session.Create(workDir, systemPrompt)
		if err != nil {
			return nil, fmt.Errorf("create session: %w", err)
		}
		ctxManager.SetSystemPrompt(systemPrompt)
	}

	var llmDebugDumper *llm.DebugDumper
	if cfg.Debug && runtimeSession != nil {
		llmDebugDumper = llm.NewDebugDumper(filepath.Dir(runtimeSession.Path()))
	}

	engineCfg := newEngineConfig(config, systemPrompt)
	engine := loop.NewEngine(engineCfg, provider, toolRegistry)
	engine.SetContextManager(ctxManager)
	engine.SetLLMDebugDumper(llmDebugDumper)
	permService := permission.NewDefaultPermissionService(config.Permissions)
	permissionUI := NewPermissionPromptUI(eventCh)
	permService.SetUI(permissionUI)
	var (
		permSettingsIssue *permissionSettingsIssue
		sessionStoreReady bool
	)
	if issue := preloadScopedPermissionRules(permService, workDir); issue != nil {
		permSettingsIssue = issue
	}
	if cfg.Resume {
		storeCfg := sessionPermissionStoreConfig(runtimeSession)
		if store, err := permission.NewPermissionStore(storeCfg); err == nil {
			permService.SetStore(store)
			sessionStoreReady = true
		} else {
			if permSettingsIssue == nil {
				storePath := storeCfg.Path
				permSettingsIssue = &permissionSettingsIssue{
					FilePath: normalizePermissionSettingsPath(storePath, workDir),
					Detail:   err.Error(),
				}
			}
		}
	}
	engine.SetPermissionService(permService)

	app := &Application{
		Engine:                  engine,
		EventCh:                 eventCh,
		WorkDir:                 workDir,
		RepoURL:                 "github.com/mindspore-lab/mindspore-cli",
		Config:                  config,
		llmDebugDumper:          llmDebugDumper,
		provider:                provider,
		toolRegistry:            toolRegistry,
		ctxManager:              ctxManager,
		permService:             permService,
		permissionUI:            permissionUI,
		permissionSettingsIssue: permSettingsIssue,
		session:                 runtimeSession,
		replayBacklog:           replayBacklog,
		replayTimeline:          replayTimeline,
		deferHistoryReplay:      cfg.Resume && !cfg.Replay,
		replayOnly:              cfg.Replay,
		replaySpeed:             replaySpeedOrDefault(cfg.ReplaySpeed),
		llmReady:                llmReady,
		skillLoader:             skillLoader,
		skillsHomeDir:           strings.TrimSpace(homeDir),
		activeModelPresetID:     activePresetID,
		needsSetupPopup:         needsSetupPopup,
		savedModelToken:         savedModelToken,
	}
	permissionUI.SetYOLOCallbacks(
		func() bool {
			if svc, ok := app.permService.(*permission.DefaultPermissionService); ok {
				return svc.Check("shell", "") == permission.PermissionAllowAlways
			}
			return false
		},
		func() {
			app.cmdYolo()
		},
	)

	// Auto-login from saved credentials.
	if cred, err := loadCredentials(); err == nil {
		app.issueService = issuepkg.NewService(issuepkg.NewRemoteStore(cred.ServerURL, cred.Token))
		app.projectService = projectpkg.NewService(projectpkg.NewRemoteStore(cred.ServerURL, cred.Token))
		app.issueUser = cred.User
		app.issueRole = cred.Role
	}
	if sessionStoreReady {
		app.sessionStoreReady.Store(true)
	}
	engine.SetTrajectoryRecorder(newTrajectoryRecorder(runtimeSession, ctxManager, app.noteLiveLLMActivity))

	return app, nil
}

func looksLikeTrajectoryPath(target string) bool {
	target = strings.TrimSpace(target)
	if target == "" {
		return false
	}
	if strings.HasSuffix(target, ".json") || strings.HasSuffix(target, ".jsonl") {
		return true
	}
	if strings.HasPrefix(target, ".") || strings.ContainsAny(target, `/\`) {
		return true
	}
	return false
}

func replaySpeedOrDefault(speed float64) float64 {
	if speed <= 0 {
		return 1
	}
	return speed
}

func sessionPermissionStoreConfig(runtimeSession *session.Session) permission.PermissionStoreConfig {
	cfg := permission.DefaultPermissionStoreConfig()
	if runtimeSession == nil {
		return cfg
	}
	sessionDir := filepath.Dir(runtimeSession.Path())
	if strings.TrimSpace(sessionDir) == "" {
		return cfg
	}
	cfg.Path = filepath.Join(sessionDir, "permissions.json")
	return cfg
}

func (a *Application) ensureSessionPermissionStore() {
	if a == nil || a.permService == nil || a.session == nil || a.replayOnly {
		return
	}
	if a.sessionStoreReady.Load() {
		return
	}
	storeSetter, ok := a.permService.(interface {
		SetStore(permission.PermissionStore)
	})
	if !ok {
		return
	}

	storeCfg := sessionPermissionStoreConfig(a.session)
	store, err := permission.NewPermissionStore(storeCfg)
	if err != nil {
		if a.permissionSettingsIssue == nil {
			a.permissionSettingsIssue = &permissionSettingsIssue{
				FilePath: normalizePermissionSettingsPath(storeCfg.Path, a.WorkDir),
				Detail:   err.Error(),
			}
		}
		return
	}

	storeSetter.SetStore(store)
	a.sessionStoreReady.Store(true)
}

// SetProvider updates model/key and reinitializes the engine.
func (a *Application) SetProvider(providerName, modelName, apiKey string) error {
	normalizedProvider := llm.NormalizeProvider(providerName)
	if normalizedProvider != "" && !llm.IsSupportedProvider(normalizedProvider) {
		return fmt.Errorf("unsupported provider: %s", providerName)
	}

	previousModel := a.Config.Model.Model

	if normalizedProvider != "" {
		a.Config.Model.Provider = normalizedProvider
	}

	if modelName != "" {
		a.Config.Model.Model = modelName
	}
	if apiKey != "" {
		a.Config.Model.Key = apiKey
	}
	configs.RefreshModelTokenDefaults(a.Config, previousModel)

	resolveOpts := llm.ResolveOptions{
		PreferConfigAPIKey: strings.TrimSpace(apiKey) != "",
	}
	provider, err := initProvider(a.Config.Model, resolveOpts)
	if err != nil {
		if err == errAPIKeyNotFound {
			a.llmReady = false
			provider = nil
		} else {
			return fmt.Errorf("init provider: %w", err)
		}
	} else {
		a.llmReady = true
	}

	systemPrompt := ""
	if a.ctxManager != nil {
		if msg := a.ctxManager.GetSystemPrompt(); msg != nil {
			systemPrompt = msg.Content
		}
	}

	engineCfg := newEngineConfig(a.Config, systemPrompt)
	newEngine := loop.NewEngine(engineCfg, provider, a.toolRegistry)
	if a.ctxManager != nil {
		if err := a.ctxManager.SetContextWindowLimits(a.Config.Context.Window, a.Config.Context.ReserveTokens); err != nil {
			return fmt.Errorf("update context limits: %w", err)
		}
	}
	newEngine.SetContextManager(a.ctxManager)
	newEngine.SetLLMDebugDumper(a.llmDebugDumper)
	newEngine.SetPermissionService(a.permService)
	newEngine.SetTrajectoryRecorder(newTrajectoryRecorder(a.session, a.ctxManager, a.noteLiveLLMActivity))

	a.Engine = newEngine
	a.provider = provider

	return nil
}

func initProvider(cfg configs.ModelConfig, opts llm.ResolveOptions) (llm.Provider, error) {
	resolved, err := llm.ResolveConfigWithOptions(cfg, opts)
	if err != nil {
		if errors.Is(err, llm.ErrMissingAPIKey) {
			return nil, errAPIKeyNotFound
		}
		return nil, fmt.Errorf("resolve provider config: %w", err)
	}

	client, err := buildProvider(resolved)
	if err != nil {
		return nil, fmt.Errorf("build provider: %w", err)
	}
	return client, nil
}

func newTrajectoryRecorder(s *session.Session, cm *agentctx.Manager, noteLiveLLMActivity func() error) *loop.TrajectoryRecorder {
	ensureSessionActive := func() error {
		if noteLiveLLMActivity == nil {
			return nil
		}
		return noteLiveLLMActivity()
	}

	return &loop.TrajectoryRecorder{
		RecordUserInput: func(content string) error {
			if s == nil {
				return nil
			}
			return s.AppendUserInput(content)
		},
		RecordAssistant: func(content string) error {
			if s == nil {
				return nil
			}
			if err := ensureSessionActive(); err != nil {
				return err
			}
			return s.AppendAssistant(content)
		},
		RecordToolCall: func(tc llm.ToolCall) error {
			if s == nil {
				return nil
			}
			if err := ensureSessionActive(); err != nil {
				return err
			}
			return s.AppendToolCall(tc)
		},
		RecordToolResult: func(tc llm.ToolCall, content string) error {
			if s == nil {
				return nil
			}
			if err := ensureSessionActive(); err != nil {
				return err
			}
			return s.AppendToolResult(tc.ID, tc.Function.Name, content)
		},
		RecordSkillActivate: func(skillName string) error {
			if s == nil {
				return nil
			}
			if err := ensureSessionActive(); err != nil {
				return err
			}
			return s.AppendSkillActivation(skillName)
		},
		PersistSnapshot: func() error {
			if s == nil || cm == nil {
				return nil
			}
			systemPrompt := ""
			if msg := cm.GetSystemPrompt(); msg != nil {
				systemPrompt = msg.Content
			}
			return s.SaveSnapshot(systemPrompt, cm.GetNonSystemMessages())
		},
	}
}

func requestMaxTokensPtr(v *int) *int {
	if v == nil {
		return nil
	}
	copy := *v
	return &copy
}

func requestTemperaturePtr(v *float64) *float32 {
	if v == nil {
		return nil
	}
	copy := float32(*v)
	return &copy
}

func requestMaxIterations(v *int) int {
	if v == nil {
		return configs.DefaultRequestMaxIterations
	}
	return *v
}

func newEngineConfig(cfg *configs.Config, systemPrompt string) loop.EngineConfig {
	return loop.EngineConfig{
		MaxIterations:  requestMaxIterations(cfg.Request.MaxIterations),
		ContextWindow:  cfg.Context.Window,
		MaxTokens:      requestMaxTokensPtr(cfg.Request.MaxTokens),
		Temperature:    requestTemperaturePtr(cfg.Request.Temperature),
		TimeoutPerTurn: time.Duration(cfg.Model.TimeoutSec) * time.Second,
		SystemPrompt:   systemPrompt,
	}
}

// detectModelMode checks whether model config is already available.
// Returns the mode string and the loaded appConfig (if any).
// Mode is modelModeOwnEnv if env vars are complete, modelModeMSCLIProvided
// if a saved token exists, or "" if neither is configured.
func detectModelMode() (string, *appConfig) {
	provider := strings.TrimSpace(os.Getenv("MSCLI_PROVIDER"))
	apiKey := strings.TrimSpace(os.Getenv("MSCLI_API_KEY"))
	modelName := strings.TrimSpace(os.Getenv("MSCLI_MODEL"))
	if provider != "" && apiKey != "" && modelName != "" {
		return modelModeOwnEnv, nil
	}

	cfg, err := loadAppConfig()
	if err != nil {
		return "", nil
	}
	if cfg.ModelMode == modelModeMSCLIProvided &&
		strings.TrimSpace(cfg.ModelPresetID) != "" &&
		strings.TrimSpace(cfg.ModelToken) != "" {
		return modelModeMSCLIProvided, cfg
	}
	return "", nil
}

func (a *Application) emitModelSetupPopup(canEscape bool) {
	presetOptions := []model.SelectionOption{}
	for _, preset := range listBuiltinModelPresets() {
		presetOptions = append(presetOptions, model.SelectionOption{
			ID:       preset.ID,
			Label:    preset.Label,
			Disabled: preset.ComingSoon,
		})
	}

	currentMode := ""
	currentPreset := ""
	if a.activeModelPresetID != "" {
		currentMode = modelModeMSCLIProvided
		currentPreset = a.activeModelPresetID
	} else if a.llmReady {
		currentMode = modelModeOwn
	}

	popup := &model.SetupPopup{
		Screen:        model.SetupScreenModeSelect,
		PresetOptions: presetOptions,
		CanEscape:     canEscape,
		CurrentMode:   currentMode,
		CurrentPreset: currentPreset,
		TokenValue:    a.savedModelToken,
	}

	a.EventCh <- model.Event{
		Type:       model.ModelSetupOpen,
		SetupPopup: popup,
	}
}

func initTools(cfg *configs.Config, workDir string) *tools.Registry {
	registry := tools.NewRegistry()

	registry.MustRegister(fs.NewReadTool(workDir))
	registry.MustRegister(fs.NewListDirTool(workDir))
	registry.MustRegister(fs.NewWriteTool(workDir))
	registry.MustRegister(fs.NewEditTool(workDir))
	registry.MustRegister(fs.NewGrepTool(workDir))
	registry.MustRegister(fs.NewGlobTool(workDir))

	shellRunner := rshell.NewRunner(rshell.Config{
		WorkDir:        workDir,
		Timeout:        time.Duration(cfg.Execution.TimeoutSec) * time.Second,
		AllowedCmds:    cfg.Permissions.AllowedTools,
		BlockedCmds:    cfg.Permissions.BlockedTools,
		RequireConfirm: []string{"rm", "mv", "cp"},
	})
	registry.MustRegister(shell.NewShellTool(shellRunner))

	return registry
}
