package app

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/mindspore-lab/mindspore-cli/agent/loop"
	"github.com/mindspore-lab/mindspore-cli/agent/session"
	"github.com/mindspore-lab/mindspore-cli/configs"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
)

func TestInitProviderAnthropic(t *testing.T) {
	provider, err := initProvider(configs.ModelConfig{
		Provider: "anthropic",
		Model:    "claude-3-5-sonnet",
		Key:      "anthropic-token",
	}, providerResolveNoOverrides())
	if err != nil {
		t.Fatalf("initProvider() error = %v", err)
	}
	if provider == nil {
		t.Fatal("initProvider() provider = nil, want provider")
	}
	if got, want := provider.Name(), "anthropic"; got != want {
		t.Fatalf("provider.Name() = %q, want %q", got, want)
	}
}

func TestInitProviderOpenAICompletionDefault(t *testing.T) {
	provider, err := initProvider(configs.ModelConfig{Model: "gpt-4o-mini", Key: "mscli-token"}, providerResolveNoOverrides())
	if err != nil {
		t.Fatalf("initProvider() error = %v", err)
	}
	if provider == nil {
		t.Fatal("initProvider() provider = nil, want provider")
	}
	if got, want := provider.Name(), "openai-completion"; got != want {
		t.Fatalf("provider.Name() = %q, want %q", got, want)
	}
}

func TestInitProviderMapsMissingKeyToAppSentinel(t *testing.T) {
	_, err := initProvider(configs.ModelConfig{Model: "gpt-4o-mini"}, providerResolveNoOverrides())
	if err == nil {
		t.Fatal("initProvider() error = nil, want missing api key error")
	}
	if !errors.Is(err, errAPIKeyNotFound) {
		t.Fatalf("initProvider() error = %v, want errAPIKeyNotFound", err)
	}
}

func TestWireBootstrapKeyAndURLOverrideEnvDuringProviderInit(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	t.Setenv("MSCLI_PROVIDER", "openai-completion")
	t.Setenv("MSCLI_API_KEY", "env-key")
	t.Setenv("MSCLI_BASE_URL", "http://127.0.0.1:1")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENAI_BASE_URL", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_BASE_URL", "")

	tempDir := t.TempDir()
	t.Chdir(tempDir)

	var gotAuth string
	var gotPath string
	origBuildProvider := buildProvider
	buildProvider = func(resolved llm.ResolvedConfig) (llm.Provider, error) {
		return newOpenAICompletionTestProvider(t, resolved, func(req *http.Request) {
			gotPath = req.URL.Path
			gotAuth = req.Header.Get("Authorization")
		}), nil
	}
	defer func() { buildProvider = origBuildProvider }()

	app, err := Wire(BootstrapConfig{
		URL:   "https://example.test/v1",
		Key:   "cli-key",
		Model: "gpt-4o-mini",
	})
	if err != nil {
		t.Fatalf("Wire() error = %v", err)
	}
	if got, want := app.Config.Model.URL, "https://example.test/v1"; got != want {
		t.Fatalf("config.Model.URL = %q, want %q", got, want)
	}
	if got, want := app.Config.Model.Key, "cli-key"; got != want {
		t.Fatalf("config.Model.Key = %q, want %q", got, want)
	}

	_, err = app.provider.Complete(context.Background(), &llm.CompletionRequest{
		Messages: []llm.Message{
			{Role: "user", Content: "ping"},
		},
	})
	if err != nil {
		t.Fatalf("provider.Complete() error = %v", err)
	}

	if got, want := gotPath, "/v1/chat/completions"; got != want {
		t.Fatalf("request path = %q, want %q", got, want)
	}
	if got, want := gotAuth, "Bearer cli-key"; got != want {
		t.Fatalf("Authorization header = %q, want %q", got, want)
	}
}

func TestWireFreshSessionDoesNotCreateSessionDirBeforeLiveLLMActivity(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("MSCLI_PROVIDER", "")
	t.Setenv("MSCLI_API_KEY", "")
	t.Setenv("MSCLI_BASE_URL", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENAI_BASE_URL", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_BASE_URL", "")

	tempDir := t.TempDir()
	t.Chdir(tempDir)

	app, err := Wire(BootstrapConfig{})
	if err != nil {
		t.Fatalf("Wire() error = %v", err)
	}
	if app.session == nil {
		t.Fatal("expected session to be allocated")
	}

	sessionDir := filepath.Dir(app.session.Path())
	if _, err := os.Stat(sessionDir); !os.IsNotExist(err) {
		t.Fatalf("expected fresh session dir to stay absent before live llm activity, got %v", err)
	}
}

func TestWireResumeAcceptsLegacyLoadSkillSessionPermissions(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("MSCLI_PROVIDER", "")
	t.Setenv("MSCLI_API_KEY", "")
	t.Setenv("MSCLI_BASE_URL", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENAI_BASE_URL", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_BASE_URL", "")

	workDir := t.TempDir()
	t.Chdir(workDir)

	runtimeSession, err := session.Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("session.Create() err = %v", err)
	}
	if err := runtimeSession.Activate(); err != nil {
		t.Fatalf("runtimeSession.Activate() err = %v", err)
	}
	if err := runtimeSession.Close(); err != nil {
		t.Fatalf("runtimeSession.Close() err = %v", err)
	}

	sessionDir := filepath.Dir(runtimeSession.Path())
	legacy := `{
  "permissions": {
    "allow": [
      "load_skill({\"name\":\"failure-agent\"})"
    ]
  }
}`
	if err := os.WriteFile(filepath.Join(sessionDir, "permissions.json"), []byte(legacy), 0644); err != nil {
		t.Fatalf("WriteFile(session permissions) err = %v", err)
	}

	app, err := Wire(BootstrapConfig{
		Resume:          true,
		ResumeSessionID: runtimeSession.ID(),
	})
	if err != nil {
		t.Fatalf("Wire() err = %v", err)
	}
	if app.permissionSettingsIssue != nil {
		t.Fatalf("permissionSettingsIssue = %#v, want nil", app.permissionSettingsIssue)
	}
	if !app.sessionStoreReady.Load() {
		t.Fatal("expected session permission store to be ready")
	}
}

func providerResolveNoOverrides() llm.ResolveOptions {
	return llm.ResolveOptions{}
}

type captureStreamProvider struct {
	lastReq *llm.CompletionRequest
}

func (p *captureStreamProvider) Name() string {
	return "capture"
}

func (p *captureStreamProvider) Complete(context.Context, *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	return nil, io.EOF
}

func (p *captureStreamProvider) CompleteStream(ctx context.Context, req *llm.CompletionRequest) (llm.StreamIterator, error) {
	copied := *req
	copied.Messages = append([]llm.Message(nil), req.Messages...)
	copied.Tools = append([]llm.Tool(nil), req.Tools...)
	p.lastReq = &copied

	return &captureTestStreamIterator{
		chunks: []llm.StreamChunk{{
			Content:      "ok",
			FinishReason: llm.FinishStop,
		}},
	}, nil
}

func (p *captureStreamProvider) SupportsTools() bool {
	return true
}

func (p *captureStreamProvider) AvailableModels() []llm.ModelInfo {
	return nil
}

type captureTestStreamIterator struct {
	chunks []llm.StreamChunk
	index  int
}

func (it *captureTestStreamIterator) Next() (*llm.StreamChunk, error) {
	if it.index >= len(it.chunks) {
		return nil, io.EOF
	}
	chunk := it.chunks[it.index]
	it.index++
	return &chunk, nil
}

func (it *captureTestStreamIterator) Close() error {
	return nil
}

type scriptedAppStreamProvider struct {
	responses []*llm.CompletionResponse
}

func (p *scriptedAppStreamProvider) Name() string {
	return "scripted"
}

func (p *scriptedAppStreamProvider) Complete(context.Context, *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	return nil, io.EOF
}

func (p *scriptedAppStreamProvider) CompleteStream(context.Context, *llm.CompletionRequest) (llm.StreamIterator, error) {
	if len(p.responses) == 0 {
		return &captureTestStreamIterator{}, nil
	}

	resp := p.responses[0]
	p.responses = p.responses[1:]

	return &captureTestStreamIterator{
		chunks: []llm.StreamChunk{{
			Content:      resp.Content,
			ToolCalls:    append([]llm.ToolCall(nil), resp.ToolCalls...),
			FinishReason: resp.FinishReason,
			Usage:        &resp.Usage,
		}},
	}, nil
}

func (p *scriptedAppStreamProvider) SupportsTools() bool {
	return true
}

func (p *scriptedAppStreamProvider) AvailableModels() []llm.ModelInfo {
	return nil
}

func TestWirePassesMSCLIMaxTokensToModelRequests(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("MSCLI_PROVIDER", "anthropic")
	t.Setenv("MSCLI_API_KEY", "anthropic-token")
	t.Setenv("MSCLI_MODEL", "claude-sonnet-4-5")
	t.Setenv("MSCLI_MAX_TOKENS", "2048")

	tempDir := t.TempDir()
	t.Chdir(tempDir)

	provider := &captureStreamProvider{}
	origBuildProvider := buildProvider
	buildProvider = func(resolved llm.ResolvedConfig) (llm.Provider, error) {
		return provider, nil
	}
	defer func() { buildProvider = origBuildProvider }()

	app, err := Wire(BootstrapConfig{})
	if err != nil {
		t.Fatalf("Wire() error = %v", err)
	}

	_, err = app.Engine.Run(loop.Task{
		ID:          "wire-max-tokens",
		Description: "ping",
	})
	if err != nil {
		t.Fatalf("Engine.Run() error = %v", err)
	}

	if provider.lastReq == nil {
		t.Fatal("expected provider to receive completion request")
	}
	if provider.lastReq.MaxTokens == nil {
		t.Fatal("provider.lastReq.MaxTokens = nil, want value")
	}
	if got, want := *provider.lastReq.MaxTokens, 2048; got != want {
		t.Fatalf("provider.lastReq.MaxTokens = %d, want %d", got, want)
	}
	if provider.lastReq.Temperature != nil {
		t.Fatalf("provider.lastReq.Temperature = %v, want nil", *provider.lastReq.Temperature)
	}
}

func TestWireOmitsRequestOverridesWhenEnvUnset(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("MSCLI_PROVIDER", "openai-completion")
	t.Setenv("MSCLI_API_KEY", "token")
	t.Setenv("MSCLI_MODEL", "gpt-4o-mini")

	tempDir := t.TempDir()
	t.Chdir(tempDir)

	provider := &captureStreamProvider{}
	origBuildProvider := buildProvider
	buildProvider = func(resolved llm.ResolvedConfig) (llm.Provider, error) {
		return provider, nil
	}
	defer func() { buildProvider = origBuildProvider }()

	app, err := Wire(BootstrapConfig{})
	if err != nil {
		t.Fatalf("Wire() error = %v", err)
	}

	_, err = app.Engine.Run(loop.Task{
		ID:          "wire-no-overrides",
		Description: "ping",
	})
	if err != nil {
		t.Fatalf("Engine.Run() error = %v", err)
	}

	if provider.lastReq == nil {
		t.Fatal("expected provider to receive completion request")
	}
	if provider.lastReq.MaxTokens != nil {
		t.Fatalf("provider.lastReq.MaxTokens = %d, want nil", *provider.lastReq.MaxTokens)
	}
	if provider.lastReq.Temperature != nil {
		t.Fatalf("provider.lastReq.Temperature = %v, want nil", *provider.lastReq.Temperature)
	}
}

func TestWirePassesMSCLITemperatureToModelRequests(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("MSCLI_PROVIDER", "openai-completion")
	t.Setenv("MSCLI_API_KEY", "token")
	t.Setenv("MSCLI_MODEL", "gpt-4o-mini")
	t.Setenv("MSCLI_TEMPERATURE", "0.25")

	tempDir := t.TempDir()
	t.Chdir(tempDir)

	provider := &captureStreamProvider{}
	origBuildProvider := buildProvider
	buildProvider = func(resolved llm.ResolvedConfig) (llm.Provider, error) {
		return provider, nil
	}
	defer func() { buildProvider = origBuildProvider }()

	app, err := Wire(BootstrapConfig{})
	if err != nil {
		t.Fatalf("Wire() error = %v", err)
	}

	_, err = app.Engine.Run(loop.Task{
		ID:          "wire-temperature",
		Description: "ping",
	})
	if err != nil {
		t.Fatalf("Engine.Run() error = %v", err)
	}

	if provider.lastReq == nil {
		t.Fatal("expected provider to receive completion request")
	}
	if provider.lastReq.Temperature == nil {
		t.Fatal("provider.lastReq.Temperature = nil, want value")
	}
	if got, want := *provider.lastReq.Temperature, float32(0.25); got != want {
		t.Fatalf("provider.lastReq.Temperature = %v, want %v", got, want)
	}
}

func TestWireAndSetProviderRespectMSCLIMaxIterations(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("MSCLI_PROVIDER", "openai-completion")
	t.Setenv("MSCLI_API_KEY", "token")
	t.Setenv("MSCLI_MODEL", "gpt-4o-mini")
	t.Setenv("MSCLI_MAX_ITERATIONS", "1")
	t.Setenv("MSCLI_PERMISSIONS_SKIP", "true")

	tempDir := t.TempDir()
	t.Chdir(tempDir)

	if err := os.WriteFile("README.md", []byte("hello"), 0600); err != nil {
		t.Fatalf("write README.md: %v", err)
	}

	args, err := json.Marshal(map[string]string{"path": "README.md"})
	if err != nil {
		t.Fatalf("marshal tool args: %v", err)
	}

	newProvider := func() llm.Provider {
		return &scriptedAppStreamProvider{
			responses: []*llm.CompletionResponse{
				{
					ToolCalls: []llm.ToolCall{{
						ID:   "call-read-1",
						Type: "function",
						Function: llm.ToolCallFunc{
							Name:      "read",
							Arguments: args,
						},
					}},
					FinishReason: llm.FinishToolCalls,
				},
				{
					Content:      "done",
					FinishReason: llm.FinishStop,
				},
			},
		}
	}

	origBuildProvider := buildProvider
	buildProvider = func(resolved llm.ResolvedConfig) (llm.Provider, error) {
		return newProvider(), nil
	}
	defer func() { buildProvider = origBuildProvider }()

	app, err := Wire(BootstrapConfig{})
	if err != nil {
		t.Fatalf("Wire() error = %v", err)
	}

	assertMaxIterationsFailure := func(taskID string) {
		t.Helper()

		events, err := app.Engine.Run(loop.Task{
			ID:          taskID,
			Description: "read the file",
		})
		if err != nil {
			t.Fatalf("Engine.Run() error = %v", err)
		}
		if len(events) == 0 {
			t.Fatal("expected events, got none")
		}

		last := events[len(events)-1]
		if got, want := last.Type, loop.EventTaskFailed; got != want {
			t.Fatalf("last event type = %q, want %q", got, want)
		}
		if got, want := last.Message, "Task exceeded maximum iterations."; got != want {
			t.Fatalf("last event message = %q, want %q", got, want)
		}
	}

	assertMaxIterationsFailure("wire-max-iterations")

	if err := app.SetProvider("", "gpt-4o-mini", ""); err != nil {
		t.Fatalf("SetProvider() error = %v", err)
	}

	assertMaxIterationsFailure("set-provider-max-iterations")
}
