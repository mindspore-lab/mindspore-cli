package ui

import (
	"testing"

	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestInitStartsDeferredChecksDuringBoot(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	if !app.bootActive {
		t.Fatal("expected boot splash to be enabled by default")
	}

	_ = app.Init()

	select {
	case got := <-userCh:
		if got != bootReadyToken {
			t.Fatalf("boot token = %q, want %q", got, bootReadyToken)
		}
	default:
		t.Fatal("expected Init to send bootReadyToken")
	}
}

func TestNewReplaySkipsBootSplash(t *testing.T) {
	app := NewReplay(nil, nil, "test", ".", "", "demo-model", 4096)
	if app.bootActive {
		t.Fatal("expected replay app to skip boot splash")
	}
}

func TestBootSplashActiveOnStart(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	if !app.bootActive {
		t.Fatal("expected boot splash to be active on start")
	}
}

func TestBannerPrintsAtBootDoneWhenNoPopup(t *testing.T) {
	// Returning user: no setup popup → banner prints immediately at boot done.
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)

	next, cmd := app.Update(bootDoneMsg{})
	app = next.(App)

	if app.bootActive {
		t.Fatal("expected boot splash to stop after bootDoneMsg")
	}
	if !app.bannerPrinted {
		t.Fatal("expected banner to print at boot done when no popup is active")
	}
	if cmd == nil {
		t.Fatal("expected banner print command")
	}
}

func TestBannerDeferredWhenSetupPopupActive(t *testing.T) {
	// First boot: setup popup opens during boot → banner deferred until popup closes.
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)

	// Simulate setup popup arriving during boot.
	next, _ := app.handleEvent(model.Event{
		Type: model.ModelSetupOpen,
		SetupPopup: &model.SetupPopup{
			Screen:    model.SetupScreenModeSelect,
			CanEscape: false,
		},
	})
	app = next.(App)

	// Boot finishes — banner should NOT print because setup popup is open.
	next, cmd := app.Update(bootDoneMsg{})
	app = next.(App)
	if app.bannerPrinted {
		t.Fatal("expected banner to be deferred while setup popup is active")
	}

	// Setup closes → banner prints.
	next, cmd = app.handleEvent(model.Event{Type: model.ModelSetupClose})
	app = next.(App)
	if !app.bannerPrinted {
		t.Fatal("expected banner to print after setup popup closes")
	}
	if cmd == nil {
		t.Fatal("expected banner print command after setup close")
	}
}
