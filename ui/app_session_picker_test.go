package ui

import (
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestSessionPickerOpenAndSelectResume(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	app = next.(App)

	next, cmd := app.handleEvent(model.Event{
		Type: model.SessionPickerOpen,
		SessionPicker: &model.SessionPicker{
			Mode: model.SessionPickerResume,
			Items: []model.SessionPickerItem{
				{
					ID:             "sess_1",
					CreatedAt:      time.Date(2026, time.April, 8, 9, 0, 0, 0, time.UTC),
					UpdatedAt:      time.Date(2026, time.April, 8, 9, 30, 0, 0, time.UTC),
					FirstUserInput: "first session",
				},
				{
					ID:             "sess_2",
					CreatedAt:      time.Date(2026, time.April, 8, 10, 0, 0, 0, time.UTC),
					UpdatedAt:      time.Date(2026, time.April, 8, 10, 30, 0, 0, time.UTC),
					FirstUserInput: "second session",
				},
			},
		},
	})
	app = next.(App)

	if cmd == nil {
		t.Fatal("expected session picker to request alt-screen")
	}
	if !app.modalAltScreen {
		t.Fatal("expected session picker to enable alt-screen")
	}
	if view := app.View(); !strings.Contains(view, "Resume Session") || !strings.Contains(view, "first session") {
		t.Fatalf("expected session picker view, got:\n%s", view)
	}

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	app = next.(App)
	next, cmd = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	if cmd == nil {
		t.Fatal("expected session picker close to update alt-screen")
	}
	if app.sessionPicker != nil {
		t.Fatal("expected session picker to close after selection")
	}

	select {
	case got := <-userCh:
		if got != "/resume sess_2" {
			t.Fatalf("selection command = %q, want %q", got, "/resume sess_2")
		}
	default:
		t.Fatal("expected session picker to submit resume command")
	}
}

func TestSessionPickerReplaySelectionIncludesSpeed(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type: model.SessionPickerOpen,
		SessionPicker: &model.SessionPicker{
			Mode:        model.SessionPickerReplay,
			ReplaySpeed: 1.5,
			Items: []model.SessionPickerItem{{
				ID:             "sess_replay",
				CreatedAt:      time.Date(2026, time.April, 8, 11, 0, 0, 0, time.UTC),
				UpdatedAt:      time.Date(2026, time.April, 8, 11, 30, 0, 0, time.UTC),
				FirstUserInput: "replay session",
			}},
		},
	})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	select {
	case got := <-userCh:
		if got != "/replay sess_replay 1.5x" {
			t.Fatalf("selection command = %q, want %q", got, "/replay sess_replay 1.5x")
		}
	default:
		t.Fatal("expected session picker to submit replay command")
	}
}

func TestClearScreenSummaryShowsNoticeMessage(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	app.state = app.state.WithMessage(model.Message{Kind: model.MsgUser, Content: "hello"})

	next, _ := app.handleEvent(model.Event{
		Type:    model.ClearScreen,
		Summary: "Resume the previous conversation with: /resume sess_123",
	})
	app = next.(App)

	if got, want := len(app.state.Messages), 1; got != want {
		t.Fatalf("message count after clear with summary = %d, want %d", got, want)
	}
	if got := app.state.Messages[0].Display; got != model.DisplayNotice {
		t.Fatalf("message display after clear = %v, want notice", got)
	}
	if got := app.state.Messages[0].Content; got != "Resume the previous conversation with: /resume sess_123" {
		t.Fatalf("message content after clear = %q", got)
	}
}

func TestClearScreenMarksBannerPrinted(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	app.bannerPrinted = false

	next, _ := app.handleEvent(model.Event{Type: model.ClearScreen})
	app = next.(App)

	if !app.bannerPrinted {
		t.Fatal("expected clear screen to count as banner already printed")
	}
	if cmd := app.maybePrintBanner(); cmd != nil {
		t.Fatal("expected no extra banner after clear screen")
	}
}

func TestStartupBannerSuppressedBlocksEarlyBannerBeforeSessionPicker(t *testing.T) {
	app := NewReplay(nil, nil, "test", ".", "", "demo-model", 4096).WithStartupBannerSuppressed()

	next, _ := app.handleEvent(model.Event{Type: model.IssueUserUpdate, Message: "alice"})
	app = next.(App)

	if app.bannerPrinted {
		t.Fatal("expected startup banner to stay suppressed before session picker resolves")
	}
}

func TestSessionPickerEnterReleasesStartupBannerSuppression(t *testing.T) {
	userCh := make(chan string, 1)
	app := NewReplay(nil, userCh, "test", ".", "", "demo-model", 4096).WithStartupBannerSuppressed()
	app.bootActive = false
	app.sessionPicker = &model.SessionPicker{
		Mode: model.SessionPickerResume,
		Items: []model.SessionPickerItem{{
			ID:             "sess_1",
			CreatedAt:      time.Date(2026, time.April, 8, 9, 0, 0, 0, time.UTC),
			UpdatedAt:      time.Date(2026, time.April, 8, 9, 30, 0, 0, time.UTC),
			FirstUserInput: "first session",
		}},
	}
	app.modalAltScreen = true

	next, cmd := app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	if cmd == nil {
		t.Fatal("expected session picker enter to exit alt-screen and print banner")
	}
	if app.startupBannerSuppressed {
		t.Fatal("expected startup banner suppression to be released on picker confirm")
	}
	if !app.bannerPrinted {
		t.Fatal("expected banner to be marked printed on picker confirm")
	}
	if app.sessionPicker != nil {
		t.Fatal("expected session picker to close after confirm")
	}

	select {
	case got := <-userCh:
		if got != "/resume sess_1" {
			t.Fatalf("selection command = %q, want %q", got, "/resume sess_1")
		}
	default:
		t.Fatal("expected session picker to submit resume command")
	}
}

func TestViewWithZeroHeightSessionPickerDoesNotPanic(t *testing.T) {
	app := NewReplay(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	app.modalAltScreen = true
	app.sessionPicker = &model.SessionPicker{
		Mode: model.SessionPickerReplay,
		Items: []model.SessionPickerItem{{
			ID:             "sess_replay",
			CreatedAt:      time.Date(2026, time.April, 8, 11, 0, 0, 0, time.UTC),
			UpdatedAt:      time.Date(2026, time.April, 8, 11, 30, 0, 0, time.UTC),
			FirstUserInput: "replay session",
		}},
	}

	_ = app.View()
}
