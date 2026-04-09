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

	"github.com/mindspore-lab/mindspore-cli/internal/bugs"
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

func credentialsPath() string {
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

	a.bugService = bugs.NewService(bugs.NewRemoteStore(serverURL, token))
	a.issueService = issuepkg.NewService(issuepkg.NewRemoteStore(serverURL, token))
	a.projectService = projectpkg.NewService(projectpkg.NewRemoteStore(serverURL, token))
	a.issueUser = me.User
	a.issueRole = me.Role

	a.EventCh <- model.Event{Type: model.IssueUserUpdate, Message: me.User}
	if !a.llmReady {
		if result, err := a.activateLogicalModelSelection(mindsporeCLIFreeProviderID, "kimi-k2.5"); err == nil {
			a.EventCh <- model.Event{
				Type:     model.ModelUpdate,
				Message:  a.Config.Model.Model,
				Provider: result.ProviderLabel,
				CtxMax:   a.Config.Context.Window,
			}
		}
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("logged in as %s (%s)", me.User, me.Role),
	}
}

func (a *Application) ensureBugService() bool {
	if a.bugService != nil {
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
	a.bugService = bugs.NewService(bugs.NewRemoteStore(cred.ServerURL, cred.Token))
	a.issueUser = cred.User
	a.issueRole = cred.Role
	a.EventCh <- model.Event{Type: model.IssueUserUpdate, Message: cred.User}
	return true
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

func (a *Application) ensureAdmin() bool {
	if a.issueRole == "" {
		if !a.ensureBugService() {
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
