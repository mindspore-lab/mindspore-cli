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
	"github.com/vigo999/ms-cli/configs"
	"github.com/vigo999/ms-cli/integrations/llm"
	providerpkg "github.com/vigo999/ms-cli/integrations/llm/provider"
	"github.com/vigo999/ms-cli/integrations/skills"
	itrain "github.com/vigo999/ms-cli/internal/train"
	"github.com/vigo999/ms-cli/permission"
	rshell "github.com/vigo999/ms-cli/runtime/shell"
	"github.com/vigo999/ms-cli/tools"
	"github.com/vigo999/ms-cli/tools/fs"
	"github.com/vigo999/ms-cli/tools/shell"
	skillstool "github.com/vigo999/ms-cli/tools/skills"
	"github.com/vigo999/ms-cli/trace"
	"github.com/vigo999/ms-cli/ui/model"
	"github.com/vigo999/ms-cli/ui/slash"
	wtrain "github.com/vigo999/ms-cli/workflow/train"
)

var errAPIKeyNotFound = errors.New("api key not found")

const Version = "MindSpore AI Infra Agent CLI. v0.2.0"

// Application is the top-level composition container.
type Application struct {
	Engine       *loop.Engine
	EventCh      chan model.Event
	llmReady     bool
	WorkDir      string
	RepoURL      string
	Config       *configs.Config
	provider     llm.Provider
	toolRegistry *tools.Registry
	ctxManager   *agentctx.Manager
	permService  permission.PermissionService
	permissionUI *PermissionPromptUI
	traceWriter  trace.Writer

	// Skills
	skillLoader *skills.Loader

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
	URL   string
	Model string
	Key   string
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
	if cfg.Model != "" {
		config.Model.Model = cfg.Model
	}
	if cfg.Key != "" {
		config.Model.Key = cfg.Key
	}

	var provider llm.Provider
	llmReady := true
	resolveOpts := providerpkg.ResolveOptions{
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

	// Skills: discover from binary dir, home dir, and project dir.
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

	// Register each skill as a slash command in the UI registry.
	for _, s := range skillLoader.List() {
		name := s.Name
		desc := s.Description
		slash.Register(slash.Command{
			Name:        "/" + name,
			Description: desc,
			Usage:       "/" + name + " [request...]",
		})
	}

	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		MaxTokens:           config.Context.MaxTokens,
		ReserveTokens:       config.Context.ReserveTokens,
		CompactionThreshold: config.Context.CompactionThreshold,
		MaxHistoryRounds:    config.Context.MaxHistoryRounds,
	})

	traceWriter, err := trace.NewTimestampWriter(filepath.Join(workDir, ".cache"))
	if err != nil {
		return nil, fmt.Errorf("init trace writer: %w", err)
	}

	// Build system prompt: base + skill summaries.
	systemPrompt := loop.DefaultSystemPrompt()
	if summaries := skillLoader.List(); len(summaries) > 0 {
		systemPrompt += "\n\n## Available Skills\n\n" +
			"Use the load_skill tool to load a skill when the user's task matches one:\n\n" +
			skills.FormatSummaries(summaries)
	}

	engineCfg := loop.EngineConfig{
		MaxIterations:  0,
		MaxTokens:      config.Budget.MaxTokens,
		Temperature:    float32(config.Model.Temperature),
		TimeoutPerTurn: time.Duration(config.Model.TimeoutSec) * time.Second,
		SystemPrompt:   systemPrompt,
	}
	engine := loop.NewEngine(engineCfg, provider, toolRegistry)
	engine.SetContextManager(ctxManager)
	engine.SetTraceWriter(traceWriter)

	permService := permission.NewDefaultPermissionService(config.Permissions)
	permissionUI := NewPermissionPromptUI(eventCh)
	permService.SetUI(permissionUI)
	if store, err := permission.NewPermissionStore(permission.DefaultPermissionStoreConfig()); err == nil {
		permService.SetStore(store)
	}
	engine.SetPermissionService(permService)

	return &Application{
		Engine:       engine,
		EventCh:      eventCh,
		WorkDir:      workDir,
		RepoURL:      "github.com/vigo999/ms-cli",
		Config:       config,
		provider:     provider,
		toolRegistry: toolRegistry,
		ctxManager:   ctxManager,
		permService:  permService,
		permissionUI: permissionUI,
		traceWriter:  traceWriter,
		llmReady:     llmReady,
		skillLoader:  skillLoader,
	}, nil
}

// SetProvider updates model/key and reinitializes the engine.
func (a *Application) SetProvider(providerName, modelName, apiKey string) error {
	normalizedProvider := providerpkg.NormalizeProvider(providerName)
	if normalizedProvider != "" && !providerpkg.IsSupportedProvider(normalizedProvider) {
		return fmt.Errorf("unsupported provider: %s", providerName)
	}

	if normalizedProvider != "" {
		a.Config.Model.Provider = normalizedProvider
	}

	if modelName != "" {
		a.Config.Model.Model = modelName
	}
	if apiKey != "" {
		a.Config.Model.Key = apiKey
	}

	resolveOpts := providerpkg.ResolveOptions{
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
		MaxTokens:      a.Config.Budget.MaxTokens,
		Temperature:    float32(a.Config.Model.Temperature),
		TimeoutPerTurn: time.Duration(a.Config.Model.TimeoutSec) * time.Second,
	}
	newEngine := loop.NewEngine(engineCfg, provider, a.toolRegistry)
	newEngine.SetContextManager(a.ctxManager)
	newEngine.SetPermissionService(a.permService)
	newEngine.SetTraceWriter(a.traceWriter)

	a.Engine = newEngine
	a.provider = provider

	return nil
}

func initProvider(cfg configs.ModelConfig, opts providerpkg.ResolveOptions) (llm.Provider, error) {
	resolved, err := providerpkg.ResolveConfigWithOptions(cfg, opts)
	if err != nil {
		if errors.Is(err, providerpkg.ErrMissingAPIKey) {
			return nil, errAPIKeyNotFound
		}
		return nil, fmt.Errorf("resolve provider config: %w", err)
	}

	client, err := providerpkg.DefaultManager().Build(resolved)
	if err != nil {
		return nil, fmt.Errorf("build provider: %w", err)
	}
	return client, nil
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
