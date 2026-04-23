package app

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mindspore-lab/mindspore-cli/configs"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestCmdModelSetup_VerifiesUserAndConfiguresModel(t *testing.T) {
	dir := t.TempDir()
	origPath := appConfigPathOverride
	appConfigPathOverride = dir + "/config.json"
	t.Cleanup(func() { appConfigPathOverride = origPath })
	if err := saveAppConfig(&appConfig{SessionRetentionDays: 21}); err != nil {
		t.Fatalf("saveAppConfig: %v", err)
	}

	// Mock server: /me returns user info, /model-presets/... returns API key.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/me" {
			w.Write([]byte(`{"user":"alice","role":"dev"}`))
			return
		}
		if strings.HasPrefix(r.URL.Path, "/model-presets/") {
			w.Write([]byte(`{"api_key":"sk-llm-key-123"}`))
			return
		}
		w.WriteHeader(404)
	}))
	defer srv.Close()

	eventCh := make(chan model.Event, 64)
	cfg := configs.DefaultConfig()
	cfg.Server.URL = srv.URL
	app := &Application{
		EventCh: eventCh,
		Config:  cfg,
	}

	app.cmdModelSetup([]string{"kimi-k2.5-free", "user-token-abc"})

	var gotUserUpdate, gotModelUpdate, gotSetupClose bool
	var replyMsg string
	for len(eventCh) > 0 {
		ev := <-eventCh
		switch ev.Type {
		case model.IssueUserUpdate:
			gotUserUpdate = true
			if ev.Message != "alice" {
				t.Errorf("IssueUserUpdate = %q, want 'alice'", ev.Message)
			}
		case model.ModelUpdate:
			gotModelUpdate = true
		case model.ModelSetupClose:
			gotSetupClose = true
		case model.AgentReply:
			replyMsg = ev.Message
		}
	}
	if !gotUserUpdate {
		t.Error("expected IssueUserUpdate event")
	}
	if !gotModelUpdate {
		t.Error("expected ModelUpdate event")
	}
	if !gotSetupClose {
		t.Error("expected ModelSetupClose event")
	}
	if !strings.Contains(replyMsg, "alice") || !strings.Contains(replyMsg, "kimi-k2.5") {
		t.Errorf("reply = %q, want user name and preset label", replyMsg)
	}

	// Verify config.json saved
	acfg, err := loadAppConfig()
	if err != nil {
		t.Fatalf("loadAppConfig: %v", err)
	}
	if acfg.ModelMode != modelModeMSCLIProvided {
		t.Errorf("ModelMode = %q, want %q", acfg.ModelMode, modelModeMSCLIProvided)
	}
	if acfg.ModelPresetID != "kimi-k2.5-free" {
		t.Errorf("ModelPresetID = %q, want 'kimi-k2.5-free'", acfg.ModelPresetID)
	}
	if acfg.SessionRetentionDays != 21 {
		t.Errorf("SessionRetentionDays = %d, want %d", acfg.SessionRetentionDays, 21)
	}

	// Verify user state
	if app.issueUser != "alice" {
		t.Errorf("issueUser = %q, want 'alice'", app.issueUser)
	}
}

func TestCmdModelSetup_InvalidPreset(t *testing.T) {
	eventCh := make(chan model.Event, 64)
	app := &Application{
		EventCh: eventCh,
		Config:  configs.DefaultConfig(),
	}

	app.cmdModelSetup([]string{"nonexistent-preset", "sk-token"})

	var gotError bool
	for len(eventCh) > 0 {
		ev := <-eventCh
		if ev.Type == model.ToolError && strings.Contains(ev.Message, "unknown preset") {
			gotError = true
		}
	}
	if !gotError {
		t.Error("expected error event for unknown preset")
	}
}

func TestCmdModelSetup_NoServerURL(t *testing.T) {
	eventCh := make(chan model.Event, 64)
	cfg := configs.DefaultConfig()
	cfg.Server.URL = ""
	app := &Application{
		EventCh: eventCh,
		Config:  cfg,
	}

	app.cmdModelSetup([]string{"kimi-k2.5-free", "sk-token"})

	var gotTokenError bool
	for len(eventCh) > 0 {
		ev := <-eventCh
		if ev.Type == model.ModelSetupTokenError && strings.Contains(ev.Message, "server URL") {
			gotTokenError = true
		}
	}
	if !gotTokenError {
		t.Error("expected token error about missing server URL")
	}
}
