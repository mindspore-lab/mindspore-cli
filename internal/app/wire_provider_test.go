package app

import (
	"context"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/vigo999/ms-cli/configs"
	"github.com/vigo999/ms-cli/integrations/llm"
	"github.com/vigo999/ms-cli/permission"
	"github.com/vigo999/ms-cli/ui/model"
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

func providerResolveNoOverrides() llm.ResolveOptions {
	return llm.ResolveOptions{}
}

func TestWire_ConfiguresPermissionPromptUI(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("MSCLI_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")
	t.Setenv("ANTHROPIC_API_KEY", "")

	tempDir := t.TempDir()
	t.Chdir(tempDir)

	app, err := Wire(BootstrapConfig{})
	if err != nil {
		t.Fatalf("Wire() error = %v", err)
	}

	if app.permissionUI == nil {
		t.Fatal("permissionUI = nil, want initialized prompt UI")
	}

	permSvc, ok := app.permService.(*permission.DefaultPermissionService)
	if !ok {
		t.Fatalf("permService type = %T, want *permission.DefaultPermissionService", app.permService)
	}

	done := make(chan struct {
		granted bool
		err     error
	}, 1)
	go func() {
		granted, err := permSvc.Request(context.Background(), "write", "", "tmp.txt")
		done <- struct {
			granted bool
			err     error
		}{granted: granted, err: err}
	}()

	select {
	case ev := <-app.EventCh:
		if ev.Type != model.PermissionPrompt {
			t.Fatalf("event type = %s, want %s", ev.Type, model.PermissionPrompt)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for permission prompt event")
	}

	app.processInput("yes")

	select {
	case out := <-done:
		if out.err != nil {
			t.Fatalf("Request() err = %v", out.err)
		}
		if !out.granted {
			t.Fatal("Request() granted = false, want true")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for permission request completion")
	}
}

func TestWire_LegacyPermissionSettingsErrorDeferredToTUI(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("MSCLI_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")
	t.Setenv("ANTHROPIC_API_KEY", "")

	tempDir := t.TempDir()
	t.Chdir(tempDir)

	legacyPath := filepath.Join(tempDir, ".ms-cli", "permissions.json")
	if err := os.MkdirAll(filepath.Dir(legacyPath), 0755); err != nil {
		t.Fatalf("mkdir permissions dir: %v", err)
	}
	if err := os.WriteFile(legacyPath, []byte(`[]`), 0644); err != nil {
		t.Fatalf("write legacy permissions: %v", err)
	}

	app, err := Wire(BootstrapConfig{})
	if err != nil {
		t.Fatalf("Wire() error = %v, want nil and deferred prompt", err)
	}
	if app.permissionSettingsIssue == nil {
		t.Fatal("permissionSettingsIssue = nil, want deferred issue")
	}
	if !strings.Contains(app.permissionSettingsIssue.Detail, "legacy array") {
		t.Fatalf("detail = %q, want legacy array error", app.permissionSettingsIssue.Detail)
	}
}

func TestWire_LoadsScopedPermissionSettingsFiles(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("MSCLI_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")
	t.Setenv("ANTHROPIC_API_KEY", "")

	tempDir := t.TempDir()
	t.Chdir(tempDir)

	localPath := filepath.Join(tempDir, ".ms-cli", "permissions.json")
	if err := os.MkdirAll(filepath.Dir(localPath), 0755); err != nil {
		t.Fatalf("mkdir settings dir: %v", err)
	}
	if err := os.WriteFile(localPath, []byte(`{"permissions":{"allow":["Bash(ls -la)"]}}`), 0644); err != nil {
		t.Fatalf("write permissions.json: %v", err)
	}

	app, err := Wire(BootstrapConfig{})
	if err != nil {
		t.Fatalf("Wire() error = %v", err)
	}

	permSvc, ok := app.permService.(*permission.DefaultPermissionService)
	if !ok {
		t.Fatalf("permService type = %T, want *permission.DefaultPermissionService", app.permService)
	}
	if got := permSvc.Check("shell", "ls -la"); got != permission.PermissionAllowAlways {
		t.Fatalf("Check(shell ls -la) = %s, want %s", got, permission.PermissionAllowAlways)
	}
}

func TestWire_SessionPermissionStorePathAndPersist(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("MSCLI_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")
	t.Setenv("ANTHROPIC_API_KEY", "")

	workDir := t.TempDir()
	t.Chdir(workDir)

	app, err := Wire(BootstrapConfig{})
	if err != nil {
		t.Fatalf("Wire() error = %v", err)
	}
	t.Cleanup(func() { _ = app.session.Close() })

	permSvc, ok := app.permService.(*permission.DefaultPermissionService)
	if !ok {
		t.Fatalf("permService type = %T, want *permission.DefaultPermissionService", app.permService)
	}

	rememberPermissionViaUI(t, app, permSvc, "shell", "ls -la", "")

	sessionPermPath := filepath.Join(filepath.Dir(app.session.Path()), "permissions.json")
	raw, err := os.ReadFile(sessionPermPath)
	if err != nil {
		t.Fatalf("ReadFile(%s) err = %v", sessionPermPath, err)
	}
	if !strings.Contains(string(raw), "Bash(ls -la)") {
		t.Fatalf("session permissions file = %s, want Bash(ls -la)", string(raw))
	}

	globalPath := filepath.Join(workDir, ".ms-cli", "permissions.state.json")
	if _, err := os.Stat(globalPath); !os.IsNotExist(err) {
		t.Fatalf("global store file should not exist, path=%s err=%v", globalPath, err)
	}
}

func TestWire_ResumeLoadsSessionScopedPermissionsOnly(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("MSCLI_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")
	t.Setenv("ANTHROPIC_API_KEY", "")

	workDir := t.TempDir()
	t.Chdir(workDir)

	first, err := Wire(BootstrapConfig{})
	if err != nil {
		t.Fatalf("first Wire() error = %v", err)
	}
	t.Cleanup(func() { _ = first.session.Close() })
	firstPermSvc, ok := first.permService.(*permission.DefaultPermissionService)
	if !ok {
		t.Fatalf("first permService type = %T, want *permission.DefaultPermissionService", first.permService)
	}
	rememberPermissionViaUI(t, first, firstPermSvc, "shell", "ls -la", "")
	sessionID := first.session.ID()

	resumed, err := Wire(BootstrapConfig{Resume: true, ResumeSessionID: sessionID})
	if err != nil {
		t.Fatalf("resume Wire() error = %v", err)
	}
	t.Cleanup(func() { _ = resumed.session.Close() })
	resumedPermSvc, ok := resumed.permService.(*permission.DefaultPermissionService)
	if !ok {
		t.Fatalf("resumed permService type = %T, want *permission.DefaultPermissionService", resumed.permService)
	}
	if got := resumedPermSvc.Check("shell", "ls -la"); got != permission.PermissionAllowSession {
		t.Fatalf("resumed Check(shell ls -la) = %s, want %s", got, permission.PermissionAllowSession)
	}

	fresh, err := Wire(BootstrapConfig{})
	if err != nil {
		t.Fatalf("fresh Wire() error = %v", err)
	}
	t.Cleanup(func() { _ = fresh.session.Close() })
	freshPermSvc, ok := fresh.permService.(*permission.DefaultPermissionService)
	if !ok {
		t.Fatalf("fresh permService type = %T, want *permission.DefaultPermissionService", fresh.permService)
	}
	if got := freshPermSvc.Check("shell", "ls -la"); got == permission.PermissionAllowSession {
		t.Fatalf("fresh Check(shell ls -la) = %s, want not %s", got, permission.PermissionAllowSession)
	}
}

func rememberPermissionViaUI(t *testing.T, app *Application, permSvc *permission.DefaultPermissionService, tool, action, path string) {
	t.Helper()
	if err := app.activateSessionPersistence(); err != nil {
		t.Fatalf("activateSessionPersistence() err = %v", err)
	}

	done := make(chan struct {
		granted bool
		err     error
	}, 1)
	go func() {
		granted, err := permSvc.Request(context.Background(), tool, action, path)
		done <- struct {
			granted bool
			err     error
		}{granted: granted, err: err}
	}()

	select {
	case ev := <-app.EventCh:
		if ev.Type != model.PermissionPrompt {
			t.Fatalf("event type = %s, want %s", ev.Type, model.PermissionPrompt)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for permission prompt event")
	}

	app.processInput("2")

	select {
	case out := <-done:
		if out.err != nil {
			t.Fatalf("Request() err = %v", out.err)
		}
		if !out.granted {
			t.Fatal("Request() granted = false, want true")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for permission request completion")
	}
}
