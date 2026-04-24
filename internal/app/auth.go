package app

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	issuepkg "github.com/mindspore-lab/mindspore-cli/internal/issues"
	projectpkg "github.com/mindspore-lab/mindspore-cli/internal/project"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

type credentials struct {
	ServerURL string `json:"server_url"`
	Token     string `json:"token"`
	User      string `json:"user"`
	Role      string `json:"role"`
}

// credentialsPathOverride allows tests to redirect the credentials path.
var credentialsPathOverride string

func credentialsPath() string {
	if credentialsPathOverride != "" {
		return credentialsPathOverride
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".mscli", "credentials.json")
}

func loadCredentials() (*credentials, error) {
	data, err := os.ReadFile(credentialsPath())
	if err != nil {
		return nil, err
	}
	var cred credentials
	if err := json.Unmarshal(data, &cred); err != nil {
		return nil, err
	}
	return &cred, nil
}

func saveCredentials(cred *credentials) error {
	dir := filepath.Dir(credentialsPath())
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(cred, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(credentialsPath(), data, 0o600)
}

func (a *Application) cmdLogin(args []string) {
	if len(args) == 0 {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Usage: /login <token>",
		}
		return
	}
	serverURL := strings.TrimRight(a.Config.Server.URL, "/")
	if serverURL == "" {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "server URL not set. Run: export MSCLI_SERVER_URL=http://<host>:9473",
		}
		return
	}
	token := args[0]

	client := &http.Client{Timeout: 5 * time.Second}
	req, err := http.NewRequest("GET", serverURL+"/me", nil)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("login failed: %v", err)}
		return
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := client.Do(req)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("login failed: cannot reach server: %v", err)}
		return
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("login failed: %s", body)}
		return
	}

	var me struct {
		User string `json:"user"`
		Role string `json:"role"`
	}
	if err := json.Unmarshal(body, &me); err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("login failed: invalid response: %v", err)}
		return
	}

	cred := &credentials{
		ServerURL: serverURL,
		Token:     token,
		User:      me.User,
		Role:      me.Role,
	}
	if err := saveCredentials(cred); err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("login ok but failed to save credentials: %v", err)}
		return
	}

	a.issueService = issuepkg.NewService(issuepkg.NewRemoteStore(serverURL, token))
	a.projectService = projectpkg.NewService(projectpkg.NewRemoteStore(serverURL, token))
	a.issueUser = me.User
	a.issueRole = me.Role

	a.EventCh <- model.Event{Type: model.IssueUserUpdate, Message: me.User}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("logged in as %s (%s)", me.User, me.Role),
	}
}

func (a *Application) ensureIssueService() bool {
	if a.issueService != nil {
		return true
	}
	cred, err := loadCredentials()
	if err != nil {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "not logged in. Run /login <token> first.",
		}
		return false
	}
	a.issueService = issuepkg.NewService(issuepkg.NewRemoteStore(cred.ServerURL, cred.Token))
	a.issueUser = cred.User
	a.issueRole = cred.Role
	a.EventCh <- model.Event{Type: model.IssueUserUpdate, Message: cred.User}
	return true
}

func (a *Application) ensureProjectService() bool {
	if a.projectService != nil {
		return true
	}
	cred, err := loadCredentials()
	if err != nil {
		return false
	}
	a.projectService = projectpkg.NewService(projectpkg.NewRemoteStore(cred.ServerURL, cred.Token))
	if a.issueUser == "" {
		a.issueUser = cred.User
	}
	return true
}

func (a *Application) cmdLogout() {
	if a.issueService == nil {
		if _, err := loadCredentials(); err != nil {
			a.EventCh <- model.Event{Type: model.AgentReply, Message: "not logged in."}
			return
		}
	}
	if err := os.Remove(credentialsPath()); err != nil && !os.IsNotExist(err) {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: fmt.Sprintf("logout failed: %v", err),
		}
		return
	}
	// If using mscli-provided model, clear config.json and in-session preset
	// state — ModelToken is the same server token; leaving it would cause
	// auto-relogin on next startup, and the preset would remain active in UI.
	if cfg, err := loadAppConfig(); err == nil && cfg.ModelMode == modelModeMSCLIProvided {
		if err := saveAppConfig(&appConfig{}); err != nil {
			a.emitToolError("config", "logout: failed to clear model config: %v", err)
		}
		// Always reset in-session preset state regardless of disk write outcome.
		startupPreset := a.activeModelPresetID != "" && a.modelBeforePreset == nil
		a.restoreModelConfigFromPreset() // restores Config.Model if switched mid-session
		if startupPreset && a.Config != nil {
			// Startup-restored preset: modelBeforePreset was never captured so
			// restoreModelConfigFromPreset() was a no-op and Config.Model still
			// holds preset values. Clear them so the session reflects no model.
			a.Config.Model.Key = ""
			a.Config.Model.URL = ""
			a.Config.Model.Model = ""
		}
		a.activeModelPresetID = ""
		a.modelBeforePreset = nil
		a.savedModelToken = ""
		a.llmReady = false
		// Notify UI so the model bar reflects the updated (cleared) state.
		if a.Config != nil {
			a.EventCh <- model.Event{
				Type:    model.ModelUpdate,
				Message: a.Config.Model.Model,
				CtxMax:  a.Config.Context.Window,
			}
		}
	}
	a.issueService = nil
	a.projectService = nil
	a.issueUser = ""
	a.issueRole = ""
	a.EventCh <- model.Event{Type: model.IssueUserUpdate, Message: ""}
	a.EventCh <- model.Event{Type: model.AgentReply, Message: "logged out. Run /login <token> to log in again."}
}

func (a *Application) ensureAdmin() bool {
	if a.issueRole == "" {
		if !a.ensureIssueService() {
			return false
		}
	}
	if a.issueRole != "admin" {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "permission denied: admin role required",
		}
		return false
	}
	return true
}
