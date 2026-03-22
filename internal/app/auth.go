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

	"github.com/vigo999/ms-cli/internal/issues"
	"github.com/vigo999/ms-cli/ui/model"
)

type credentials struct {
	ServerURL string `json:"server_url"`
	Token     string `json:"token"`
	User      string `json:"user"`
	Role      string `json:"role"`
}

func credentialsPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".ms-cli", "credentials.json")
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
	if len(args) < 2 {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Usage: /login <server-url> <token>",
		}
		return
	}
	serverURL := strings.TrimRight(args[0], "/")
	token := args[1]

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

	a.issueService = issues.NewService(issues.NewRemoteStore(serverURL, token))
	a.issueUser = me.User

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
			Message: "not logged in. Run /login <server-url> <token> first.",
		}
		return false
	}
	a.issueService = issues.NewService(issues.NewRemoteStore(cred.ServerURL, cred.Token))
	a.issueUser = cred.User
	return true
}
