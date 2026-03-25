package app

import (
	"testing"
	"time"

	issuepkg "github.com/vigo999/ms-cli/internal/issues"
	"github.com/vigo999/ms-cli/ui/model"
)

func TestCmdIssuesDefaultsToAllAndOpensIssueIndexView(t *testing.T) {
	store := &fakeAppIssueStore{
		issues: []issuepkg.Issue{
			{ID: 42, Key: "ISSUE-42", Title: "acc failure in migrate", Kind: issuepkg.KindAccuracy, Status: "ready", Reporter: "alice", UpdatedAt: time.Now()},
		},
	}
	app := &Application{
		EventCh:      make(chan model.Event, 4),
		issueService: issuepkg.NewService(store),
	}

	app.cmdIssues(nil)

	ev := <-app.EventCh
	if ev.Type != model.IssueIndexOpen {
		t.Fatalf("event type = %s, want %s", ev.Type, model.IssueIndexOpen)
	}
	if ev.IssueView == nil || ev.IssueView.Filter != "all" {
		t.Fatalf("issue view filter = %#v, want all", ev.IssueView)
	}
	if got := len(ev.IssueView.Items); got != 1 {
		t.Fatalf("issue count = %d, want 1", got)
	}
}

func TestCmdIssueReportInputCreatesIssue(t *testing.T) {
	store := &fakeAppIssueStore{}
	app := &Application{
		EventCh:      make(chan model.Event, 4),
		issueService: issuepkg.NewService(store),
		issueUser:    "alice",
	}

	app.cmdIssueReportInput("accuracy acc failure in migrate")

	ev := <-app.EventCh
	if ev.Type != model.AgentReply {
		t.Fatalf("event type = %s, want %s", ev.Type, model.AgentReply)
	}
	if got, want := store.lastCreateKind, issuepkg.KindAccuracy; got != want {
		t.Fatalf("kind = %q, want %q", got, want)
	}
	if got, want := store.lastCreateTitle, "acc failure in migrate"; got != want {
		t.Fatalf("title = %q, want %q", got, want)
	}
}

func TestParseIssueCommandTargetTreatsFreeTextAsPrompt(t *testing.T) {
	target, err := parseIssueCommandTarget("training loss too big", "/diagnose")
	if err != nil {
		t.Fatalf("parseIssueCommandTarget returned error: %v", err)
	}
	if target.HasIssue {
		t.Fatalf("HasIssue = true, want false")
	}
	if got, want := target.Prompt, "training loss too big"; got != want {
		t.Fatalf("Prompt = %q, want %q", got, want)
	}
}

func TestParseIssueCommandTargetAcceptsIssueKeyWithExtraContext(t *testing.T) {
	target, err := parseIssueCommandTarget("ISSUE-42 training loss too big", "/diagnose")
	if err != nil {
		t.Fatalf("parseIssueCommandTarget returned error: %v", err)
	}
	if !target.HasIssue {
		t.Fatalf("HasIssue = false, want true")
	}
	if got, want := target.IssueID, 42; got != want {
		t.Fatalf("IssueID = %d, want %d", got, want)
	}
	if got, want := target.Prompt, "training loss too big"; got != want {
		t.Fatalf("Prompt = %q, want %q", got, want)
	}
}

func TestParseIssueCommandTargetRejectsMalformedIssueKey(t *testing.T) {
	_, err := parseIssueCommandTarget("ISSUE-abc", "/fix")
	if err == nil {
		t.Fatal("parseIssueCommandTarget error = nil, want error")
	}
	if got, want := err.Error(), "invalid issue id"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
}

func TestCmdDiagnoseAcceptsFreeTextInput(t *testing.T) {
	app := &Application{
		EventCh: make(chan model.Event, 4),
	}

	app.cmdDiagnose("training loss too big")

	ev := <-app.EventCh
	if ev.Type != model.AgentReply {
		t.Fatalf("event type = %s, want %s", ev.Type, model.AgentReply)
	}
	if got, want := ev.Message, `diagnose flow for "training loss too big" is not wired yet`; got != want {
		t.Fatalf("message = %q, want %q", got, want)
	}
}

func TestCmdFixAcceptsIssueTarget(t *testing.T) {
	app := &Application{
		EventCh:      make(chan model.Event, 4),
		issueService: issuepkg.NewService(&fakeAppIssueStore{}),
	}

	app.cmdFix("ISSUE-42")

	ev := <-app.EventCh
	if ev.Type != model.AgentReply {
		t.Fatalf("event type = %s, want %s", ev.Type, model.AgentReply)
	}
	if got, want := ev.Message, "fix flow for ISSUE-42 is not wired yet"; got != want {
		t.Fatalf("message = %q, want %q", got, want)
	}
}

type fakeAppIssueStore struct {
	lastCreateTitle string
	lastCreateKind  issuepkg.Kind
	issues          []issuepkg.Issue
	issue           *issuepkg.Issue
	notes           []issuepkg.Note
	activity        []issuepkg.Activity
}

func (f *fakeAppIssueStore) CreateIssue(title string, kind issuepkg.Kind, reporter string) (*issuepkg.Issue, error) {
	f.lastCreateTitle = title
	f.lastCreateKind = kind
	return &issuepkg.Issue{ID: 42, Key: "ISSUE-42", Title: title, Kind: kind, Status: "ready", Reporter: reporter}, nil
}

func (f *fakeAppIssueStore) ListIssues(status string) ([]issuepkg.Issue, error) {
	return f.issues, nil
}

func (f *fakeAppIssueStore) GetIssue(id int) (*issuepkg.Issue, error) {
	return f.issue, nil
}

func (f *fakeAppIssueStore) AddNote(issueID int, author, content string) (*issuepkg.Note, error) {
	return &issuepkg.Note{ID: 1, IssueID: issueID, Author: author, Content: content}, nil
}

func (f *fakeAppIssueStore) ListNotes(issueID int) ([]issuepkg.Note, error) {
	return f.notes, nil
}

func (f *fakeAppIssueStore) ListActivity(issueID int) ([]issuepkg.Activity, error) {
	return f.activity, nil
}

func (f *fakeAppIssueStore) ClaimIssue(id int, lead string) (*issuepkg.Issue, error) {
	return f.issue, nil
}

func (f *fakeAppIssueStore) UpdateStatus(id int, status string, actor string) (*issuepkg.Issue, error) {
	return f.issue, nil
}
