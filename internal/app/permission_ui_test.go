package app

import (
	"strings"
	"testing"
	"time"

	"github.com/vigo999/mindspore-cli/ui/model"
)

func TestPermissionPromptUI_RequestPermissionAndApproveSession(t *testing.T) {
	eventCh := make(chan model.Event, 4)
	ui := NewPermissionPromptUI(eventCh)

	resultCh := make(chan struct {
		granted  bool
		remember bool
		err      error
	}, 1)

	go func() {
		granted, remember, err := ui.RequestPermission("write", "", "blank.md")
		resultCh <- struct {
			granted  bool
			remember bool
			err      error
		}{granted: granted, remember: remember, err: err}
	}()

	select {
	case ev := <-eventCh:
		if ev.Type != model.PermissionPrompt {
			t.Fatalf("event type = %s, want %s", ev.Type, model.PermissionPrompt)
		}
		if want := "Do you want to make this edit to blank.md?"; !strings.Contains(ev.Message, want) {
			t.Fatalf("prompt message = %q, want contains %q", ev.Message, want)
		}
		if !strings.Contains(ev.Message, "1. Yes") || !strings.Contains(ev.Message, "2. Yes, allow all edits during this session") || !strings.Contains(ev.Message, "3. No") {
			t.Fatalf("prompt options not in expected Claude-style format: %q", ev.Message)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for permission prompt")
	}

	if handled := ui.HandleInput("2"); !handled {
		t.Fatal("HandleInput() = false, want true for pending prompt option 2")
	}

	select {
	case out := <-resultCh:
		if out.err != nil {
			t.Fatalf("RequestPermission() err = %v", out.err)
		}
		if !out.granted {
			t.Fatal("RequestPermission() granted = false, want true")
		}
		if !out.remember {
			t.Fatal("RequestPermission() remember = false, want true")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for permission decision")
	}
}

func TestPermissionPromptUI_RequestPermissionOffersYOLOWhenDisabled(t *testing.T) {
	eventCh := make(chan model.Event, 4)
	ui := NewPermissionPromptUI(eventCh)
	ui.SetYOLOCallbacks(func() bool { return false }, func() {})

	resultCh := make(chan struct {
		granted  bool
		remember bool
		err      error
	}, 1)

	go func() {
		granted, remember, err := ui.RequestPermission("shell", "go test ./...", "")
		resultCh <- struct {
			granted  bool
			remember bool
			err      error
		}{granted: granted, remember: remember, err: err}
	}()

	select {
	case ev := <-eventCh:
		if ev.Type != model.PermissionPrompt {
			t.Fatalf("event type = %s, want %s", ev.Type, model.PermissionPrompt)
		}
		if !strings.Contains(ev.Message, "4. Enable YOLO mode and allow all operations") {
			t.Fatalf("prompt message = %q, want yolo option", ev.Message)
		}
		if ev.Permission == nil || len(ev.Permission.Options) != 4 {
			t.Fatalf("permission options = %#v, want 4 options", ev.Permission)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for permission prompt")
	}

	if handled := ui.HandleInput("3"); !handled {
		t.Fatal("HandleInput() = false, want true for pending prompt option 3")
	}

	select {
	case out := <-resultCh:
		if out.err != nil {
			t.Fatalf("RequestPermission() err = %v", out.err)
		}
		if out.granted {
			t.Fatal("RequestPermission() granted = true, want false")
		}
		if out.remember {
			t.Fatal("RequestPermission() remember = true, want false")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for permission decision")
	}
}

func TestPermissionPromptUI_SelectYOLOEnablesAndGrants(t *testing.T) {
	eventCh := make(chan model.Event, 4)
	ui := NewPermissionPromptUI(eventCh)

	yoloEnabled := false
	ui.SetYOLOCallbacks(
		func() bool { return yoloEnabled },
		func() { yoloEnabled = true },
	)

	resultCh := make(chan struct {
		granted  bool
		remember bool
		err      error
	}, 1)
	go func() {
		granted, remember, err := ui.RequestPermission("load_skill", "", "")
		resultCh <- struct {
			granted  bool
			remember bool
			err      error
		}{granted: granted, remember: remember, err: err}
	}()

	<-eventCh
	if handled := ui.HandleInput("4"); !handled {
		t.Fatal("HandleInput() = false, want true for pending prompt option 4")
	}

	select {
	case out := <-resultCh:
		if out.err != nil {
			t.Fatalf("RequestPermission() err = %v", out.err)
		}
		if !out.granted {
			t.Fatal("RequestPermission() granted = false, want true")
		}
		if out.remember {
			t.Fatal("RequestPermission() remember = true, want false")
		}
		if !yoloEnabled {
			t.Fatal("expected YOLO callback to enable yolo mode")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for permission decision")
	}
}

func TestPermissionPromptUI_HandleInputWithoutPending(t *testing.T) {
	ui := NewPermissionPromptUI(make(chan model.Event, 1))
	if handled := ui.HandleInput("yes"); handled {
		t.Fatal("HandleInput() = true, want false when no request is pending")
	}
}

func TestPermissionPromptUI_RejectWithEsc(t *testing.T) {
	eventCh := make(chan model.Event, 4)
	ui := NewPermissionPromptUI(eventCh)

	resultCh := make(chan struct {
		granted bool
		err     error
	}, 1)
	go func() {
		granted, _, err := ui.RequestPermission("edit", "", "main.go")
		resultCh <- struct {
			granted bool
			err     error
		}{granted: granted, err: err}
	}()

	<-eventCh
	ui.HandleInput("esc")

	select {
	case out := <-resultCh:
		if out.err != nil {
			t.Fatalf("RequestPermission() err = %v", out.err)
		}
		if out.granted {
			t.Fatal("RequestPermission() granted = true, want false")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for permission decision")
	}
}

func TestApplicationProcessInput_PermissionReplyPriority(t *testing.T) {
	app := &Application{
		EventCh: make(chan model.Event, 4),
	}
	app.permissionUI = NewPermissionPromptUI(app.EventCh)

	resultCh := make(chan bool, 1)
	go func() {
		granted, _, _ := app.permissionUI.RequestPermission("write", "", "a.txt")
		resultCh <- granted
	}()

	select {
	case <-app.EventCh:
		// prompt received
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for permission prompt")
	}

	app.processInput("yes")

	select {
	case granted := <-resultCh:
		if !granted {
			t.Fatal("permission response denied, want granted")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for processInput to resolve permission request")
	}
}
