package app

import (
	"strings"
	"testing"

	"github.com/vigo999/ms-cli/ui/model"
)

func TestProcessInput_PermissionSettingsErrorContinue(t *testing.T) {
	eventCh := make(chan model.Event, 8)
	app := &Application{
		EventCh: eventCh,
		permissionSettingsIssue: &permissionSettingsIssue{
			FilePath: "/tmp/.ms-cli/permissions.json",
			Detail:   "invalid settings",
		},
	}

	app.processInput("2")

	if app.permissionSettingsIssue != nil {
		t.Fatal("permissionSettingsIssue should be cleared after continue")
	}

	foundAck := false
	for len(eventCh) > 0 {
		ev := <-eventCh
		if ev.Type == model.AgentReply && strings.Contains(ev.Message, "Continuing without") {
			foundAck = true
		}
	}
	if !foundAck {
		t.Fatal("expected continue acknowledgement event")
	}
}

func TestProcessInput_PermissionSettingsErrorExit(t *testing.T) {
	eventCh := make(chan model.Event, 8)
	app := &Application{
		EventCh: eventCh,
		permissionSettingsIssue: &permissionSettingsIssue{
			FilePath: "/tmp/.ms-cli/permissions.json",
			Detail:   "invalid settings",
		},
	}

	app.processInput("1")

	if app.permissionSettingsIssue != nil {
		t.Fatal("permissionSettingsIssue should be cleared after exit selection")
	}

	foundDone := false
	for len(eventCh) > 0 {
		ev := <-eventCh
		if ev.Type == model.Done {
			foundDone = true
		}
	}
	if !foundDone {
		t.Fatal("expected Done event after selecting exit")
	}
}

func TestProcessInput_PermissionSettingsErrorInvalidInputRePrompts(t *testing.T) {
	eventCh := make(chan model.Event, 8)
	app := &Application{
		EventCh: eventCh,
		permissionSettingsIssue: &permissionSettingsIssue{
			FilePath: "/tmp/.ms-cli/permissions.json",
			Detail:   "invalid settings",
		},
	}

	app.processInput("abc")

	if app.permissionSettingsIssue == nil {
		t.Fatal("permissionSettingsIssue should remain for invalid input")
	}

	foundPrompt := false
	for len(eventCh) > 0 {
		ev := <-eventCh
		if ev.Type == model.PermissionPrompt && strings.Contains(ev.Message, "Settings Error") {
			foundPrompt = true
		}
	}
	if !foundPrompt {
		t.Fatal("expected permission settings prompt to be re-emitted")
	}
}
