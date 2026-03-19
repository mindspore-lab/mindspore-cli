package app

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	agentctx "github.com/vigo999/ms-cli/agent/context"
	"github.com/vigo999/ms-cli/agent/loop"
	"github.com/vigo999/ms-cli/configs"
	"github.com/vigo999/ms-cli/integrations/llm"
	providerpkg "github.com/vigo999/ms-cli/integrations/llm/provider"
	itrain "github.com/vigo999/ms-cli/internal/train"
	"github.com/vigo999/ms-cli/permission"
	rshell "github.com/vigo999/ms-cli/runtime/shell"
	"github.com/vigo999/ms-cli/tools"
	"github.com/vigo999/ms-cli/tools/fs"
	"github.com/vigo999/ms-cli/tools/shell"
	"github.com/vigo999/ms-cli/trace"
	"github.com/vigo999/ms-cli/ui/model"
	wtrain "github.com/vigo999/ms-cli/workflow/train"
)

var errAPIKeyNotFound = errors.New("api key not found")

const Version = "MindSpore AI Infra Agent CLI. v0.2.0"

// Application is the top-level composition container.
type Application struct {
	Engine       *loop.Engine
	EventCh      chan model.Event
	Demo         bool
	llmReady     bool
	WorkDir      string
	RepoURL      string
	Config       *configs.Config
	provider     llm.Provider
	toolRegistry *tools.Registry
	ctxManager   *agentctx.Manager
	permService  permission.PermissionService
	stateManager *configs.StateManager
	traceWriter  trace.Writer

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
	trainMu         sync.RWMutex
}

// BootstrapConfig holds bootstrap configuration.
type BootstrapConfig struct {
	Demo       bool
	ConfigPath string
	URL        string
	Model      string
	Key        string
}

// Wire builds and returns the Application.
func Wire(cfg BootstrapConfig) (*Application, error) {
	configPath := cfg.ConfigPath
	if configPath == "" {
		configPath = configs.FindConfigFile()
	}

	config, err := configs.LoadWithEnv(configPath)
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	workDir, err := os.Getwd()
	if err != nil {
		workDir = "."
	}
	workDir, _ = filepath.Abs(workDir)

	stateManager := configs.NewStateManager(workDir)
	if err := stateManager.Load(); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: failed to load state: %v\n", err)
	}
	stateManager.ApplyToConfig(config)
	configs.ApplyEnvOverrides(config)

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
	if cfg.Demo {
		llmReady = false
	} else {
		provider, err = initProvider(config.Model)
		if err != nil {
			if errors.Is(err, errAPIKeyNotFound) {
				llmReady = false
				provider = nil
			} else {
				return nil, fmt.Errorf("init provider: %w", err)
			}
		}
	}

	toolRegistry := initTools(config, workDir)

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

	engineCfg := loop.EngineConfig{
		MaxIterations:  0,
		MaxTokens:      config.Budget.MaxTokens,
		Temperature:    float32(config.Model.Temperature),
		TimeoutPerTurn: time.Duration(config.Model.TimeoutSec) * time.Second,
	}
	engine := loop.NewEngine(engineCfg, provider, toolRegistry)
	engine.SetContextManager(ctxManager)
	engine.SetTraceWriter(traceWriter)

	permService := permission.NewDefaultPermissionService(config.Permissions)
	engine.SetPermissionService(permService)

	return &Application{
		Engine:       engine,
		EventCh:      make(chan model.Event, 64),
		Demo:         cfg.Demo,
		WorkDir:      workDir,
		RepoURL:      "github.com/vigo999/ms-cli",
		Config:       config,
		provider:     provider,
		toolRegistry: toolRegistry,
		ctxManager:   ctxManager,
		permService:  permService,
		stateManager: stateManager,
		traceWriter:  traceWriter,
		llmReady:     llmReady,
	}, nil
}

// SetProvider updates model/key and reinitializes the engine.
func (a *Application) SetProvider(providerName, modelName, apiKey string) error {
	if providerName != "" && providerName != "openai" {
		return fmt.Errorf("unsupported provider: %s (only openai-compatible is supported)", providerName)
	}

	if modelName != "" {
		a.Config.Model.Model = modelName
	}
	if apiKey != "" {
		a.Config.Model.Key = apiKey
	}

	provider, err := initProvider(a.Config.Model)
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

	if a.stateManager != nil {
		a.stateManager.SaveFromConfig(a.Config)
		if err := a.stateManager.Save(); err != nil {
			return fmt.Errorf("save state: %w", err)
		}
	}

	return nil
}

// SaveState saves current configuration to persistent state.
func (a *Application) SaveState() error {
	if a.stateManager == nil {
		return nil
	}
	a.stateManager.SaveFromConfig(a.Config)
	return a.stateManager.Save()
}

func initProvider(cfg configs.ModelConfig) (llm.Provider, error) {
	resolved, err := providerpkg.ResolveConfig(cfg)
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
