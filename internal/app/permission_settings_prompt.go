package app

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/vigo999/ms-cli/ui/model"
)

type permissionSettingsIssue struct {
	FilePath string
	Detail   string
}

func (a *Application) emitPermissionSettingsPrompt(extraHint string) {
	if a == nil || a.EventCh == nil || a.permissionSettingsIssue == nil {
		return
	}
	issue := a.permissionSettingsIssue
	msg := fmt.Sprintf("Settings Error\n\n%s\n\nFiles with errors are skipped entirely, not just the invalid settings.", permissionSettingsIssueBody(issue))
	if strings.TrimSpace(extraHint) != "" {
		msg += "\n\n" + strings.TrimSpace(extraHint)
	}
	a.EventCh <- model.Event{
		Type:    model.PermissionPrompt,
		Message: msg,
		Permission: &model.PermissionPromptData{
			Title:   "Settings Error",
			Message: msg,
			Options: []model.PermissionOption{
				{Input: "1", Label: "1. Exit and fix manually"},
				{Input: "2", Label: "2. Continue without these settings"},
			},
			DefaultIndex: 0,
		},
	}
}

func permissionSettingsIssueBody(issue *permissionSettingsIssue) string {
	if issue == nil {
		return ""
	}
	path := strings.TrimSpace(issue.FilePath)
	detail := strings.TrimSpace(issue.Detail)
	if path == "" {
		path = ".ms-cli/permissions.json"
	}
	if detail == "" {
		detail = "invalid settings"
	}
	return fmt.Sprintf("%s\n  └ %s", path, detail)
}

func normalizePermissionSettingsPath(path, workDir string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	if filepath.IsAbs(path) {
		return filepath.Clean(path)
	}
	if strings.TrimSpace(workDir) == "" {
		if abs, err := filepath.Abs(path); err == nil {
			return abs
		}
		return filepath.Clean(path)
	}
	return filepath.Clean(filepath.Join(workDir, path))
}
