package ui

import (
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
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

func TestPermissionPrompt_YOLOOptionArrowSelectAndEnter(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.PermissionPrompt,
		Permission: &model.PermissionPromptData{
			Title:   "Permission required",
			Message: "allow shell?",
			Options: []model.PermissionOption{
				{Input: "1", Label: "1. Yes"},
				{Input: "2", Label: "2. Yes, don't ask again for this session"},
				{Input: "3", Label: "3. No"},
				{Input: "4", Label: "4. Enable YOLO mode and allow all operations"},
			},
			DefaultIndex: 0,
		},
	})
	app = next.(App)

	for i := 0; i < 3; i++ {
		nextModel, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyDown})
		app = nextModel.(App)
	}
	nextModel, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = nextModel.(App)

	select {
	case got := <-userCh:
		if got != "4" {
			t.Fatalf("submitted input = %q, want %q", got, "4")
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for yolo selection result")
	}

	if app.permissionPrompt != nil {
		t.Fatal("permissionPrompt should be cleared after enter")
	}
}

func TestPermissionsView_EnterAddRuleOpensDialogAndSubmits(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.PermissionsView,
		Permissions: &model.PermissionsViewData{
			Allow: []string{"Tool(read)"},
			Ask:   []string{"Tool(write)"},
			Deny:  []string{"Tool(delete)"},
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
	if app.permissionsView == nil || app.permissionsView.dialogMode != permissionsDialogChooseRuleScope {
		t.Fatal("expected choose-scope dialog after entering rule")
	}
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = nextModel.(App)

	select {
	case got := <-userCh:
		want := internalPermissionsActionPrefix + "add allow_always edit(*.md) --scope project"
		if got != want {
			t.Fatalf("submitted command = %q, want %q", got, want)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for add-rule submit")
	}
}

func TestPermissionsView_AddRuleDialogCursorLeftInsert(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.PermissionsView,
		Permissions: &model.PermissionsViewData{
			Allow: []string{},
		},
	})
	app = next.(App)

	nextModel, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = nextModel.(App)

	for _, r := range []rune("Bash(ls )") {
		nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
		app = nextModel.(App)
	}
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyLeft})
	app = nextModel.(App)
	for _, r := range []rune("-la") {
		nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
		app = nextModel.(App)
	}
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = nextModel.(App)
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = nextModel.(App)

	select {
	case got := <-userCh:
		want := internalPermissionsActionPrefix + "add allow_always Bash(ls -la) --scope project"
		if got != want {
			t.Fatalf("submitted command = %q, want %q", got, want)
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
			Allow: []string{"Tool(read)"},
			Ask:   []string{"Tool(write)"},
			Deny:  []string{"Tool(delete)"},
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

func TestPermissionsView_SearchAcceptsSpaceKeyAndKeepsIt(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.PermissionsView,
		Permissions: &model.PermissionsViewData{
			Allow: []string{"Bash(ls -la)"},
		},
	})
	app = next.(App)

	nextModel, _ := app.handleKey(tea.KeyMsg{Type: tea.KeySpace})
	app = nextModel.(App)
	if app.permissionsView == nil {
		t.Fatal("permissions view should be active")
	}
	if got := app.permissionsView.search; got != " " {
		t.Fatalf("search = %q, want single space", got)
	}
	out := renderPermissionsViewPopup(app.permissionsView)
	if !strings.Contains(out, "\u00a0") {
		t.Fatalf("popup should show visible space, got: %q", out)
	}
}

func TestPermissionsView_SearchCursorMoveAndInsert(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.PermissionsView,
		Permissions: &model.PermissionsViewData{
			Allow: []string{"Bash(ls -la)"},
		},
	})
	app = next.(App)

	nextModel, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("ab")})
	app = nextModel.(App)
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyLeft})
	app = nextModel.(App)
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeySpace})
	app = nextModel.(App)
	if app.permissionsView == nil {
		t.Fatal("permissions view should be active")
	}
	if got := app.permissionsView.search; got != "a b" {
		t.Fatalf("search = %q, want %q", got, "a b")
	}
	if got := app.permissionsView.searchCursor; got != 2 {
		t.Fatalf("searchCursor = %d, want 2", got)
	}
}

func TestPermissionsView_SelectExistingToolOpensDeleteConfirm(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.PermissionsView,
		Permissions: &model.PermissionsViewData{
			Allow: []string{"Tool(shell)"},
			Ask:   []string{},
			Deny:  []string{},
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
		want := internalPermissionsActionPrefix + "remove tool Tool(shell)"
		if got != want {
			t.Fatalf("submitted command = %q, want %q", got, want)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for action command")
	}
}

func TestPermissionsView_DoubleCtrlCQuitsWithoutEsc(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.PermissionsView,
		Permissions: &model.PermissionsViewData{
			Allow: []string{"edit(*.md)"},
			Ask:   []string{},
			Deny:  []string{},
		},
	})
	app = next.(App)

	nextModel, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyCtrlC})
	app = nextModel.(App)
	if app.permissionsView == nil {
		t.Fatal("permissions view should still be open after first ctrl+c")
	}

	nextModel, cmd := app.handleKey(tea.KeyMsg{Type: tea.KeyCtrlC})
	app = nextModel.(App)
	if cmd == nil {
		t.Fatal("second ctrl+c should quit")
	}
	if _, ok := cmd().(tea.QuitMsg); !ok {
		t.Fatal("second ctrl+c should return tea.Quit command")
	}
}

func TestPermissionsView_RendersInLiveArea(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	app.width = 80
	app.height = 24

	next, _ := app.handleEvent(model.Event{
		Type: model.PermissionsView,
		Permissions: &model.PermissionsViewData{
			Allow: []string{"edit(*.md)"},
			Ask:   []string{},
			Deny:  []string{},
		},
	})
	app = next.(App)

	if app.permissionsView == nil {
		t.Fatal("permissions view should be set after PermissionsView event")
	}
}

func TestPermissionsView_EscClosesWithoutDismissedMessage(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	app.state.Messages = []model.Message{{
		Kind:    model.MsgAgent,
		Content: "existing",
	}}

	next, _ := app.handleEvent(model.Event{
		Type: model.PermissionsView,
		Permissions: &model.PermissionsViewData{
			Allow: []string{"edit(*.md)"},
			Ask:   []string{},
			Deny:  []string{},
		},
	})
	app = next.(App)

	nextModel, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	app = nextModel.(App)
	if app.permissionsView != nil {
		t.Fatal("permissions view should be closed after esc")
	}
	if len(app.state.Messages) != 1 {
		t.Fatalf("messages length = %d, want 1", len(app.state.Messages))
	}
	if got := app.state.Messages[0].Content; got != "existing" {
		t.Fatalf("existing message changed to %q", got)
	}
}

func TestRenderPermissionsViewPopup_DialogOnlyWhenSecondaryOpen(t *testing.T) {
	v := &permissionsViewState{
		tab:         0,
		allow:       []string{"edit(*.md)"},
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

func TestPermissionsRemoveCommandForItem_CaseInsensitiveEdit(t *testing.T) {
	got, ok := permissionsRemoveCommandForItem(0, "Edit(*.md)")
	if !ok {
		t.Fatal("ok = false, want true")
	}
	want := internalPermissionsActionPrefix + "remove path *.md"
	if got != want {
		t.Fatalf("command = %q, want %q", got, want)
	}
}

func TestRenderDialogInputWithCursor_UsesReverseBlockAndVisibleSpaces(t *testing.T) {
	got := renderDialogInputWithCursor("a b", 1, "placeholder")
	if strings.Contains(got, "|") {
		t.Fatalf("got = %q, want no pipe cursor", got)
	}
	if !strings.Contains(got, "\u00a0") {
		t.Fatalf("got = %q, want visible space via NBSP", got)
	}
}

func TestRenderDialogInputWithCursor_CursorAtEndAddsTrailingBlock(t *testing.T) {
	got := renderDialogInputWithCursor("ab", 2, "placeholder")
	if !strings.HasSuffix(got, "\u00a0") {
		t.Fatalf("got = %q, want trailing NBSP cursor block", got)
	}
}

func TestPermissionsDialog_AddRuleAcceptsSpaceKey(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.PermissionsView,
		Permissions: &model.PermissionsViewData{
			Allow: []string{},
		},
	})
	app = next.(App)
	nextModel, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = nextModel.(App)
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("Bash(ls")})
	app = nextModel.(App)
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeySpace})
	app = nextModel.(App)
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("-la)")})
	app = nextModel.(App)
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = nextModel.(App)
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = nextModel.(App)

	select {
	case got := <-userCh:
		want := internalPermissionsActionPrefix + "add allow_always Bash(ls -la) --scope project"
		if got != want {
			t.Fatalf("submitted command = %q, want %q", got, want)
		}
	default:
		t.Fatal("expected command submit with preserved space")
	}
}
