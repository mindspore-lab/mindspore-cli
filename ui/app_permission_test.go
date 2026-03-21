package ui

import (
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/vigo999/ms-cli/ui/model"
)

func TestPermissionPrompt_ArrowSelectAndEnter(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.PermissionPrompt,
		Permission: &model.PermissionPromptData{
			Title:   "Confirm Edit",
			Message: "Do you want to make this edit to blank.md?",
			Options: []model.PermissionOption{
				{Input: "1", Label: "1. Yes"},
				{Input: "2", Label: "2. Yes, allow all edits during this session"},
				{Input: "3", Label: "3. No"},
			},
			DefaultIndex: 0,
		},
	})
	app = next.(App)

	nextModel, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	app = nextModel.(App)
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = nextModel.(App)

	select {
	case got := <-userCh:
		if got != "2" {
			t.Fatalf("submitted input = %q, want %q", got, "2")
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for permission selection result")
	}

	if app.permissionPrompt != nil {
		t.Fatal("permissionPrompt should be cleared after enter")
	}
}

func TestPermissionPrompt_EscCancels(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.PermissionPrompt,
		Permission: &model.PermissionPromptData{
			Title:        "Permission required",
			Message:      "allow shell?",
			Options:      []model.PermissionOption{{Input: "1", Label: "1. Yes"}, {Input: "3", Label: "3. No"}},
			DefaultIndex: 0,
		},
	})
	app = next.(App)

	nextModel, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	app = nextModel.(App)

	select {
	case got := <-userCh:
		if got != "esc" {
			t.Fatalf("submitted input = %q, want %q", got, "esc")
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for esc result")
	}

	if app.permissionPrompt != nil {
		t.Fatal("permissionPrompt should be cleared after esc")
	}
}

func TestPermissionsView_EnterAddRuleOpensDialogAndSubmits(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.PermissionsView,
		Permissions: &model.PermissionsViewData{
			Mode:      "default",
			Allow:     []string{"Tool(read)"},
			Ask:       []string{"Tool(write)"},
			Deny:      []string{"Tool(delete)"},
			Workspace: []string{"mode: default"},
		},
	})
	app = next.(App)

	nextModel, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = nextModel.(App)

	if app.permissionsView == nil || app.permissionsView.dialogMode != permissionsDialogAddRule {
		t.Fatal("expected add-rule dialog to be open")
	}
	for _, r := range []rune("edit(*.md)") {
		nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
		app = nextModel.(App)
	}
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = nextModel.(App)

	select {
	case got := <-userCh:
		if got != "/permissions add path *.md allow_always" {
			t.Fatalf("submitted command = %q, want %q", got, "/permissions add path *.md allow_always")
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for add-rule submit")
	}
}

func TestPermissionsView_TabSearchAndSelect(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.PermissionsView,
		Permissions: &model.PermissionsViewData{
			Mode:      "default",
			Allow:     []string{"Tool(read)"},
			Ask:       []string{"Tool(write)"},
			Deny:      []string{"Tool(delete)"},
			Workspace: []string{"workdir: /tmp/work"},
		},
	})
	app = next.(App)

	nextModel, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	app = nextModel.(App)
	if app.permissionsView == nil || app.permissionsView.tab != 1 {
		t.Fatalf("tab = %d, want 1", app.permissionsView.tab)
	}

	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("write")})
	app = nextModel.(App)
	items := permissionsFilteredItems(app.permissionsView)
	if len(items) != 1 || items[0] != "Tool(write)" {
		t.Fatalf("filtered items = %v, want [Tool(write)]", items)
	}
}

func TestPermissionsView_SelectExistingToolOpensDeleteConfirm(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.PermissionsView,
		Permissions: &model.PermissionsViewData{
			Mode:      "default",
			Allow:     []string{"Tool(shell)"},
			Ask:       []string{},
			Deny:      []string{},
			Workspace: []string{"mode: default"},
		},
	})
	app = next.(App)

	// index 0 is "Add a new rule…", index 1 is "Tool(shell)"
	nextModel, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	app = nextModel.(App)
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = nextModel.(App)

	if app.permissionsView == nil || app.permissionsView.dialogMode != permissionsDialogDeleteRule {
		t.Fatal("expected delete-confirm dialog to be open after selecting existing rule")
	}

	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = nextModel.(App)

	select {
	case got := <-userCh:
		if got != "/permissions remove tool Tool(shell)" {
			t.Fatalf("submitted command = %q, want %q", got, "/permissions remove tool Tool(shell)")
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for action command")
	}
}

func TestPermissionsView_WorkspaceTabItems(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.PermissionsView,
		Permissions: &model.PermissionsViewData{
			Mode:      "default",
			Allow:     []string{"shell"},
			Ask:       []string{},
			Deny:      []string{},
			Workspace: []string{"/Users/demo/project"},
		},
	})
	app = next.(App)

	// Allow -> Ask -> Deny -> Workspace
	nextModel, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	app = nextModel.(App)
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	app = nextModel.(App)
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	app = nextModel.(App)

	items := permissionsFilteredItems(app.permissionsView)
	if len(items) != 2 {
		t.Fatalf("workspace items length = %d, want 2 (dir + add directory)", len(items))
	}
	if items[0] != "/Users/demo/project" {
		t.Fatalf("workspace first item = %q, want %q", items[0], "/Users/demo/project")
	}
	if items[1] != "Add directory…" {
		t.Fatalf("workspace second item = %q, want %q", items[1], "Add directory…")
	}
}

func TestPermissionsView_WorkspaceAddDirectoryDialogSubmit(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.PermissionsView,
		Permissions: &model.PermissionsViewData{
			Mode:      "default",
			Allow:     []string{"shell"},
			Ask:       []string{},
			Deny:      []string{},
			Workspace: []string{"/Users/demo/project"},
		},
	})
	app = next.(App)

	// Go to workspace tab.
	nextModel, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	app = nextModel.(App)
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	app = nextModel.(App)
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	app = nextModel.(App)

	// Select "Add directory…"
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	app = nextModel.(App)
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = nextModel.(App)

	if app.permissionsView == nil || app.permissionsView.dialogMode != permissionsDialogAddWorkspace {
		t.Fatal("expected add-workspace dialog to be open")
	}

	for _, r := range []rune("/tmp/extra") {
		nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
		app = nextModel.(App)
	}
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = nextModel.(App)

	select {
	case got := <-userCh:
		if got != "/permissions workspace add /tmp/extra" {
			t.Fatalf("submitted command = %q, want %q", got, "/permissions workspace add /tmp/extra")
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for workspace add submit")
	}
}

func TestPermissionsView_DoubleCtrlCQuitsWithoutEsc(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.PermissionsView,
		Permissions: &model.PermissionsViewData{
			Mode:      "default",
			Allow:     []string{"edit(*.md)"},
			Ask:       []string{},
			Deny:      []string{},
			Workspace: []string{"/Users/demo/project"},
		},
	})
	app = next.(App)

	nextModel, cmd := app.handleKey(tea.KeyMsg{Type: tea.KeyCtrlC})
	app = nextModel.(App)
	if cmd != nil {
		t.Fatal("first ctrl+c should not quit immediately")
	}
	if app.permissionsView == nil {
		t.Fatal("permissions view should still be open after first ctrl+c")
	}

	nextModel, cmd = app.handleKey(tea.KeyMsg{Type: tea.KeyCtrlC})
	app = nextModel.(App)
	if cmd == nil {
		t.Fatal("second ctrl+c should quit")
	}
	if _, ok := cmd().(tea.QuitMsg); !ok {
		t.Fatal("second ctrl+c should return tea.Quit command")
	}
}

func TestViewportRenderState_IncludesPermissionsViewAsAgentMessage(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.PermissionsView,
		Permissions: &model.PermissionsViewData{
			Mode:      "default",
			Allow:     []string{"edit(*.md)"},
			Ask:       []string{},
			Deny:      []string{},
			Workspace: []string{"/Users/demo/project"},
		},
	})
	app = next.(App)

	rs := app.viewportRenderState()
	if len(rs.Messages) == 0 {
		t.Fatal("render state messages should not be empty")
	}
	last := rs.Messages[len(rs.Messages)-1]
	if last.Kind != model.MsgAgent {
		t.Fatalf("last message kind = %v, want %v", last.Kind, model.MsgAgent)
	}
	if !strings.Contains(last.Content, "Permissions:") {
		t.Fatalf("last message content should include permissions view header, got %q", last.Content)
	}
}

func TestPermissionsView_EscAddsDismissedMessage(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.PermissionsView,
		Permissions: &model.PermissionsViewData{
			Mode:      "default",
			Allow:     []string{"edit(*.md)"},
			Ask:       []string{},
			Deny:      []string{},
			Workspace: []string{"/Users/demo/project"},
		},
	})
	app = next.(App)

	nextModel, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	app = nextModel.(App)
	if app.permissionsView != nil {
		t.Fatal("permissions view should be closed after esc")
	}
	if len(app.state.Messages) == 0 {
		t.Fatal("messages should not be empty after dismissal")
	}
	last := app.state.Messages[len(app.state.Messages)-1]
	if last.Kind != model.MsgAgent || !strings.Contains(last.Content, "  ⎿  Permissions dialog dismissed") {
		t.Fatalf("last message = %#v, want dismissed agent message", last)
	}
}

func TestRenderPermissionsViewPopup_DialogOnlyWhenSecondaryOpen(t *testing.T) {
	v := &permissionsViewState{
		tab:         0,
		allow:       []string{"edit(*.md)"},
		workspace:   []string{"/Users/demo/project"},
		dialogMode:  permissionsDialogAddRule,
		dialogInput: "",
	}
	out := renderPermissionsViewPopup(v)
	if !strings.Contains(out, "Add allow permission rule") {
		t.Fatalf("dialog output missing add-rule title: %q", out)
	}
	if strings.Contains(out, "Permissions:") {
		t.Fatalf("dialog output should not include primary permissions list header: %q", out)
	}
}
