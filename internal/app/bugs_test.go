package app

import (
	"strings"
	"testing"
	"time"

	bugpkg "github.com/vigo999/mindspore-cli/internal/bugs"
	"github.com/vigo999/mindspore-cli/ui/model"
)

type fakeBugStore struct {
	dock *bugpkg.DockData
}

func (f *fakeBugStore) CreateBug(title, reporter string, tags []string) (*bugpkg.Bug, error) {
	return nil, nil
}

func (f *fakeBugStore) ListBugs(status string) ([]bugpkg.Bug, error) {
	return nil, nil
}

func (f *fakeBugStore) GetBug(id int) (*bugpkg.Bug, error) {
	return nil, nil
}

func (f *fakeBugStore) ClaimBug(id int, lead string) error {
	return nil
}

func (f *fakeBugStore) CloseBug(id int) error {
	return nil
}

func (f *fakeBugStore) AddNote(bugID int, author, content string) (*bugpkg.Note, error) {
	return nil, nil
}

func (f *fakeBugStore) ListActivity(bugID int) ([]bugpkg.Activity, error) {
	return nil, nil
}

func (f *fakeBugStore) DockSummary() (*bugpkg.DockData, error) {
	return f.dock, nil
}

func TestCmdDockStreamsRawANSI(t *testing.T) {
	store := &fakeBugStore{
		dock: &bugpkg.DockData{
			OpenCount:   21,
			OnlineCount: 5,
			ReadyBugs: []bugpkg.Bug{
				{ID: 43, Title: "abnormal display for contents of /project", Status: "open"},
			},
			RecentFeed: []bugpkg.Activity{
				{
					Actor:     "xinwen",
					Text:      "reported bug: abnormal display for contents of /dock",
					CreatedAt: time.Date(2026, 4, 3, 3, 22, 0, 0, time.FixedZone("CST", 8*3600)),
				},
			},
		},
	}

	app := &Application{
		EventCh:    make(chan model.Event, 4),
		bugService: bugpkg.NewService(store),
	}

	app.cmdDock()

	ev := drainUntilEventType(t, app, model.AgentReply)
	if !ev.RawANSI {
		t.Fatal("expected /dock output to be marked RawANSI")
	}
	for _, want := range []string{
		"DOCK",
		"open bugs",
		"online (24h)",
		"abnormal display for contents of /project",
		"reported bug: abnormal display for contents of /dock",
	} {
		if !strings.Contains(ev.Message, want) {
			t.Fatalf("expected /dock output to contain %q, got:\n%s", want, ev.Message)
		}
	}
}
