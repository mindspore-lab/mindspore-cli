package app

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	issuepkg "github.com/mindspore-lab/mindspore-cli/internal/issues"
	"github.com/mindspore-lab/mindspore-cli/configs"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestCmdLogoutClearsState(t *testing.T) {
	tmp := t.TempDir()
	credentialsPathOverride = filepath.Join(tmp, "credentials.json")
	appConfigPathOverride = filepath.Join(tmp, "config.json")
	defer func() { credentialsPathOverride = ""; appConfigPathOverride = "" }()

	_ = os.WriteFile(credentialsPathOverride,
		[]byte(`{"server_url":"http://x","token":"tok","user":"alice","role":"admin"}`), 0o600)
	_ = saveAppConfig(&appConfig{ModelMode: modelModeMSCLIProvided, ModelPresetID: "kimi-k2.5-free", ModelToken: "tok"})

	app := &Application{
		EventCh:      make(chan model.Event, 8),
		issueService: issuepkg.NewService(&fakeAppIssueStore{}),
		issueUser:    "alice",
		issueRole:    "admin",
	}
	app.cmdLogout()

	if app.issueService != nil {
		t.Error("issueService should be nil")
	}
	if app.issueUser != "" || app.issueRole != "" {
		t.Errorf("user/role should be empty, got %q/%q", app.issueUser, app.issueRole)
	}
	ev := drainUntilEventType(t, app, model.IssueUserUpdate)
	if ev.Message != "" {
		t.Errorf("IssueUserUpdate.Message should be empty, got %q", ev.Message)
	}
	if _, err := os.Stat(credentialsPathOverride); !os.IsNotExist(err) {
		t.Error("credentials file should be deleted")
	}
	// config.json should have ModelToken and ModelMode cleared
	data, _ := os.ReadFile(appConfigPathOverride)
	var cfg appConfig
	_ = json.Unmarshal(data, &cfg)
	if cfg.ModelToken != "" {
		t.Errorf("config.json ModelToken should be cleared, got %q", cfg.ModelToken)
	}
	if cfg.ModelMode != "" {
		t.Errorf("config.json ModelMode should be cleared, got %q", cfg.ModelMode)
	}
}

func TestCmdLogoutOwnModelKeepsConfig(t *testing.T) {
	tmp := t.TempDir()
	credentialsPathOverride = filepath.Join(tmp, "credentials.json")
	appConfigPathOverride = filepath.Join(tmp, "config.json")
	defer func() { credentialsPathOverride = ""; appConfigPathOverride = "" }()

	_ = os.WriteFile(credentialsPathOverride,
		[]byte(`{"server_url":"http://x","token":"tok","user":"bob","role":"user"}`), 0o600)
	_ = saveAppConfig(&appConfig{ModelMode: modelModeOwn})

	app := &Application{
		EventCh:      make(chan model.Event, 8),
		issueService: issuepkg.NewService(&fakeAppIssueStore{}),
		issueUser:    "bob",
		issueRole:    "user",
	}
	app.cmdLogout()

	// config.json should be untouched (own model, no server token in it)
	data, _ := os.ReadFile(appConfigPathOverride)
	var cfg appConfig
	_ = json.Unmarshal(data, &cfg)
	if cfg.ModelMode != modelModeOwn {
		t.Errorf("config.json ModelMode should remain %q, got %q", modelModeOwn, cfg.ModelMode)
	}
}

func TestCmdLogoutStartupPresetClearsModel(t *testing.T) {
	tmp := t.TempDir()
	credentialsPathOverride = filepath.Join(tmp, "credentials.json")
	appConfigPathOverride = filepath.Join(tmp, "config.json")
	defer func() { credentialsPathOverride = ""; appConfigPathOverride = "" }()

	_ = os.WriteFile(credentialsPathOverride,
		[]byte(`{"server_url":"http://x","token":"tok","user":"alice","role":"admin"}`), 0o600)
	_ = saveAppConfig(&appConfig{ModelMode: modelModeMSCLIProvided, ModelPresetID: "kimi-k2.5-free", ModelToken: "tok"})

	cfg := configs.DefaultConfig()
	cfg.Model.Model = "kimi-k2.5"
	cfg.Model.URL = "https://preset.api/"
	cfg.Model.Key = "sk-preset"

	// Simulate startup-restored preset: activeModelPresetID is set but modelBeforePreset is nil.
	app := &Application{
		EventCh:             make(chan model.Event, 16),
		issueService:        issuepkg.NewService(&fakeAppIssueStore{}),
		issueUser:           "alice",
		issueRole:           "admin",
		Config:              cfg,
		activeModelPresetID: "kimi-k2.5-free",
		modelBeforePreset:   nil,
	}
	app.cmdLogout()

	if app.Config.Model.Model != "" {
		t.Errorf("Config.Model.Model should be cleared, got %q", app.Config.Model.Model)
	}
	if app.Config.Model.Key != "" {
		t.Errorf("Config.Model.Key should be cleared, got %q", app.Config.Model.Key)
	}
	if app.Config.Model.URL != "" {
		t.Errorf("Config.Model.URL should be cleared, got %q", app.Config.Model.URL)
	}
	// ModelUpdate should carry empty model name.
	ev := drainUntilEventType(t, app, model.ModelUpdate)
	if ev.Message != "" {
		t.Errorf("ModelUpdate.Message should be empty, got %q", ev.Message)
	}
}

func TestCmdLogoutWhenNotLoggedIn(t *testing.T) {
	tmp := t.TempDir()
	credentialsPathOverride = filepath.Join(tmp, "credentials.json")
	defer func() { credentialsPathOverride = "" }()
	// No credentials file, no in-memory service.
	app := &Application{EventCh: make(chan model.Event, 4)}
	app.cmdLogout()

	ev := drainUntilEventType(t, app, model.AgentReply)
	if ev.Message != "not logged in." {
		t.Errorf("unexpected message: %q", ev.Message)
	}
}
