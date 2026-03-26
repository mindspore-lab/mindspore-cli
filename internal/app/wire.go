package app

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	agentctx "github.com/vigo999/ms-cli/agent/context"
	"github.com/vigo999/ms-cli/agent/loop"
	"github.com/vigo999/ms-cli/agent/session"
	"github.com/vigo999/ms-cli/configs"
	"github.com/vigo999/ms-cli/integrations/llm"
	"github.com/vigo999/ms-cli/integrations/skills"
	"github.com/vigo999/ms-cli/internal/bugs"
	issuepkg "github.com/vigo999/ms-cli/internal/issues"
	projectpkg "github.com/vigo999/ms-cli/internal/project"
	itrain "github.com/vigo999/ms-cli/internal/train"
	"github.com/vigo999/ms-cli/internal/version"
	"github.com/vigo999/ms-cli/permission"
	rshell "github.com/vigo999/ms-cli/runtime/shell"
	"github.com/vigo999/ms-cli/tools"
	"github.com/vigo999/ms-cli/tools/fs"
	"github.com/vigo999/ms-cli/tools/shell"
	skillstool "github.com/vigo999/ms-cli/tools/skills"
	"github.com/vigo999/ms-cli/ui/model"
	wtrain "github.com/vigo999/ms-cli/workflow/train"
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

	// Skills
	skillLoader   *skills.Loader
	skillsHomeDir string
	startupOnce   sync.Once

	// Bug tracking
	bugService   *bugs.Service
	issueService *issuepkg.Service
	issueUser    string
	issueRole    string

	// Project tracking
	projectService *projectpkg.Service

	// Foreground chat task state
	taskRunID   uint64
	taskCancels map[uint64]context.CancelFunc
	taskMu      sync.Mutex

	// Model preset runtime override state.
	activeModelPresetID string
	modelBeforePreset   *configs.ModelConfig

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
	Resume          bool
	ResumeSessionID string
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

	toolRegistry := initTools(config, workDir)

	// Skills: all skills live under ~/.ms-cli/skills/.
	// Shared repo skills are copied there after sync.
	homeDir, _ := os.UserHomeDir()
	execSkillsDir := ""
	if ep, err := os.Executable(); err == nil {
		execSkillsDir = filepath.Join(filepath.Dir(ep), ".ms-cli", "skills")
	}
	skillLoader := skills.NewLoader(
		execSkillsDir,
		filepath.Join(homeDir, ".ms-cli", "skills"),
		filepath.Join(workDir, ".ms-cli", "skills"),
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
	)
	if cfg.Resume {
		if strings.TrimSpace(cfg.ResumeSessionID) != "" {
			runtimeSession, err = session.LoadByID(workDir, cfg.ResumeSessionID)
			if err != nil {
				return nil, fmt.Errorf("load session %s: %w", cfg.ResumeSessionID, err)
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
		replayBacklog = runtimeSession.ReplayEvents()
	} else {
		runtimeSession, err = session.Create(workDir, systemPrompt)
		if err != nil {
			return nil, fmt.Errorf("create session: %w", err)
		}
		ctxManager.SetSystemPrompt(systemPrompt)
	}

	engineCfg := loop.EngineConfig{
		MaxIterations:  0,
		ContextWindow:  config.Context.Window,
		MaxTokens:      requestMaxTokensPtr(config.Request.MaxTokens),
		Temperature:    requestTemperaturePtr(config.Request.Temperature),
		TimeoutPerTurn: time.Duration(config.Model.TimeoutSec) * time.Second,
		SystemPrompt:   systemPrompt,
	}
	engine := loop.NewEngine(engineCfg, provider, toolRegistry)
	engine.SetContextManager(ctxManager)
	engine.SetTrajectoryRecorder(newTrajectoryRecorder(runtimeSession, ctxManager))

	permService := permission.NewDefaultPermissionService(config.Permissions)
	permissionUI := NewPermissionPromptUI(eventCh)
	permService.SetUI(permissionUI)
	var permSettingsIssue *permissionSettingsIssue
	if issue := preloadScopedPermissionRules(permService, workDir); issue != nil {
		permSettingsIssue = issue
	}
	storeCfg := sessionPermissionStoreConfig(runtimeSession)
	if store, err := permission.NewPermissionStore(storeCfg); err == nil {
		permService.SetStore(store)
	} else {
		if permSettingsIssue == nil {
			storePath := storeCfg.Path
			permSettingsIssue = &permissionSettingsIssue{
				FilePath: normalizePermissionSettingsPath(storePath, workDir),
				Detail:   err.Error(),
			}
		}
	}
	engine.SetPermissionService(permService)

	app := &Application{
		Engine:                  engine,
		EventCh:                 eventCh,
		WorkDir:                 workDir,
		RepoURL:                 "github.com/vigo999/ms-cli",
		Config:                  config,
		provider:                provider,
		toolRegistry:            toolRegistry,
		ctxManager:              ctxManager,
		permService:             permService,
		permissionUI:            permissionUI,
		permissionSettingsIssue: permSettingsIssue,
		session:                 runtimeSession,
		replayBacklog:           replayBacklog,
		llmReady:                llmReady,
		skillLoader:             skillLoader,
		skillsHomeDir:           strings.TrimSpace(homeDir),
	}

	// Auto-login from saved credentials.
	if cred, err := loadCredentials(); err == nil {
		app.bugService = bugs.NewService(bugs.NewRemoteStore(cred.ServerURL, cred.Token))
		app.issueService = issuepkg.NewService(issuepkg.NewRemoteStore(cred.ServerURL, cred.Token))
		app.projectService = projectpkg.NewService(projectpkg.NewRemoteStore(cred.ServerURL, cred.Token))
		app.issueUser = cred.User
		app.issueRole = cred.Role
	}

	return app, nil
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

	engineCfg := loop.EngineConfig{
		MaxIterations:  10,
		ContextWindow:  a.Config.Context.Window,
		MaxTokens:      requestMaxTokensPtr(a.Config.Request.MaxTokens),
		Temperature:    requestTemperaturePtr(a.Config.Request.Temperature),
		TimeoutPerTurn: time.Duration(a.Config.Model.TimeoutSec) * time.Second,
	}
	newEngine := loop.NewEngine(engineCfg, provider, a.toolRegistry)
	if a.ctxManager != nil {
		if err := a.ctxManager.SetContextWindowLimits(a.Config.Context.Window, a.Config.Context.ReserveTokens); err != nil {
			return fmt.Errorf("update context limits: %w", err)
		}
	}
	newEngine.SetContextManager(a.ctxManager)
	newEngine.SetPermissionService(a.permService)
	newEngine.SetTrajectoryRecorder(newTrajectoryRecorder(a.session, a.ctxManager))

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

func newTrajectoryRecorder(s *session.Session, cm *agentctx.Manager) *loop.TrajectoryRecorder {
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
			return s.AppendAssistant(content)
		},
		RecordToolCall: func(tc llm.ToolCall) error {
			if s == nil {
				return nil
			}
			return s.AppendToolCall(tc)
		},
		RecordToolResult: func(tc llm.ToolCall, content string) error {
			if s == nil {
				return nil
			}
			return s.AppendToolResult(tc.ID, tc.Function.Name, content)
		},
		RecordSkillActivate: func(skillName string) error {
			if s == nil {
				return nil
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

func initTools(cfg *configs.Config, workDir string) *tools.Registry {
	registry := tools.NewRegistry()

	registry.MustRegister(fs.NewReadTool(workDir))
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
