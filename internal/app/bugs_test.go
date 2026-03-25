package app

import (
	"reflect"
	"testing"
	"time"

	"github.com/vigo999/ms-cli/internal/bugs"
	"github.com/vigo999/ms-cli/ui/model"
)

func TestCmdBugsDefaultsToAllAndOpensBugIndexView(t *testing.T) {
	store := &fakeIssueStore{
		bugs: []bugs.Bug{
			{ID: 1042, Title: "loss spike after dataloader refactor", Status: "open", Reporter: "travis", UpdatedAt: time.Now()},
		},
	}
	app := &Application{
		EventCh:    make(chan model.Event, 4),
		bugService: bugs.NewService(store),
	}

	app.cmdBugs(nil)

	ev := <-app.EventCh
	if ev.Type != model.BugIndexOpen {
		t.Fatalf("event type = %s, want %s", ev.Type, model.BugIndexOpen)
	}
	if store.lastListStatus != "" {
		t.Fatalf("list status = %q, want empty for all", store.lastListStatus)
	}
	if ev.BugView == nil || ev.BugView.Filter != "all" {
		t.Fatalf("bug view filter = %#v, want all", ev.BugView)
	}
	if got := len(ev.BugView.Items); got != 1 {
		t.Fatalf("bug count = %d, want 1", got)
	}
}

func TestCmdBugDetailOpensDetailViewEvent(t *testing.T) {
	store := &fakeIssueStore{
		bug: &bugs.Bug{ID: 1042, Title: "loss spike after dataloader refactor", Tags: []string{"train"}, Status: "open", Reporter: "travis", UpdatedAt: time.Now()},
		activity: []bugs.Activity{
			{BugID: 1042, Actor: "travis", Text: "created bug", CreatedAt: time.Now()},
		},
	}
	app := &Application{
		EventCh:    make(chan model.Event, 4),
		bugService: bugs.NewService(store),
	}

	app.cmdBugDetail([]string{"1042"})

	ev := <-app.EventCh
	if ev.Type != model.BugDetailOpen {
		t.Fatalf("event type = %s, want %s", ev.Type, model.BugDetailOpen)
	}
	if ev.BugView == nil || ev.BugView.Bug == nil {
		t.Fatalf("missing bug detail payload: %#v", ev.BugView)
	}
	if ev.BugView.Bug.ID != 1042 {
		t.Fatalf("bug id = %d, want 1042", ev.BugView.Bug.ID)
	}
	if got := len(ev.BugView.Activity); got != 1 {
		t.Fatalf("activity count = %d, want 1", got)
	}
}

func TestHandleCommandReportParsesOptionalTags(t *testing.T) {
	store := &fakeIssueStore{}
	app := &Application{
		EventCh:    make(chan model.Event, 4),
		bugService: bugs.NewService(store),
		issueUser:  "travis",
	}

	app.handleCommand("/report [ui, train,ui] prompt overlaps bug detail")

	ev := <-app.EventCh
	if ev.Type != model.AgentReply {
		t.Fatalf("event type = %s, want %s", ev.Type, model.AgentReply)
	}
	if got, want := store.lastCreateTitle, "prompt overlaps bug detail"; got != want {
		t.Fatalf("create title = %q, want %q", got, want)
	}
	if got, want := store.lastCreateReporter, "travis"; got != want {
		t.Fatalf("create reporter = %q, want %q", got, want)
	}
	if got, want := store.lastCreateTags, []string{"ui", "train"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("create tags = %#v, want %#v", got, want)
	}
	if got, want := ev.Message, "created bug #1042 [ui,train]: prompt overlaps bug detail"; got != want {
		t.Fatalf("reply = %q, want %q", got, want)
	}
}

type fakeIssueStore struct {
	lastListStatus     string
	lastCreateTitle    string
	lastCreateReporter string
	lastCreateTags     []string
	bugs               []bugs.Bug
	bug                *bugs.Bug
	activity           []bugs.Activity
}

func (f *fakeIssueStore) CreateBug(title, reporter string, tags []string) (*bugs.Bug, error) {
	f.lastCreateTitle = title
	f.lastCreateReporter = reporter
	f.lastCreateTags = append([]string(nil), tags...)
	return &bugs.Bug{ID: 1042, Title: title, Tags: append([]string(nil), tags...)}, nil
}

func (f *fakeIssueStore) ListBugs(status string) ([]bugs.Bug, error) {
	f.lastListStatus = status
	return f.bugs, nil
}

func (f *fakeIssueStore) GetBug(id int) (*bugs.Bug, error) {
	return f.bug, nil
}

func (f *fakeIssueStore) ClaimBug(id int, lead string) error {
	return nil
}

func (f *fakeIssueStore) CloseBug(id int) error {
	return nil
}

func (f *fakeIssueStore) AddNote(bugID int, author, content string) (*bugs.Note, error) {
	return nil, nil
}

func (f *fakeIssueStore) ListActivity(bugID int) ([]bugs.Activity, error) {
	return f.activity, nil
}

func (f *fakeIssueStore) DockSummary() (*bugs.DockData, error) {
	return nil, nil
}
