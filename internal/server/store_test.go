package server

import (
	"database/sql"
	"reflect"
	"testing"

	_ "github.com/mattn/go-sqlite3"
	issuepkg "github.com/vigo999/mindspore-cli/internal/issues"
)

func TestStoreCreateBugPersistsTags(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite db: %v", err)
	}
	defer db.Close()

	store, err := NewStore(db)
	if err != nil {
		t.Fatalf("new store: %v", err)
	}

	bug, err := store.CreateBug("prompt overlaps bug detail", "travis", []string{"ui", " train ", "ui"})
	if err != nil {
		t.Fatalf("create bug: %v", err)
	}
	if got, want := bug.Tags, []string{"ui", "train"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("create bug tags = %#v, want %#v", got, want)
	}

	listed, err := store.ListBugs("")
	if err != nil {
		t.Fatalf("list bugs: %v", err)
	}
	if got, want := len(listed), 1; got != want {
		t.Fatalf("bug count = %d, want %d", got, want)
	}
	if got, want := listed[0].Tags, []string{"ui", "train"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("listed bug tags = %#v, want %#v", got, want)
	}
}

func TestStoreCreateIssueAndTrackNotesAndStatus(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite db: %v", err)
	}
	defer db.Close()

	store, err := NewStore(db)
	if err != nil {
		t.Fatalf("new store: %v", err)
	}

	issue, err := store.CreateIssue("acc failure in migrate", issuepkg.KindAccuracy, "alice")
	if err != nil {
		t.Fatalf("create issue: %v", err)
	}
	if got, want := issue.Key, "ISSUE-1"; got != want {
		t.Fatalf("issue key = %q, want %q", got, want)
	}
	if got, want := issue.Status, "ready"; got != want {
		t.Fatalf("issue status = %q, want %q", got, want)
	}

	if _, err := store.AddIssueNote(issue.ID, "alice", "maybe dtype mismatch"); err != nil {
		t.Fatalf("add issue note: %v", err)
	}
	notes, err := store.ListIssueNotes(issue.ID)
	if err != nil {
		t.Fatalf("list issue notes: %v", err)
	}
	if got, want := len(notes), 1; got != want {
		t.Fatalf("note count = %d, want %d", got, want)
	}
	if got, want := notes[0].Content, "maybe dtype mismatch"; got != want {
		t.Fatalf("note content = %q, want %q", got, want)
	}

	claimed, err := store.ClaimIssue(issue.ID, "bob")
	if err != nil {
		t.Fatalf("claim issue: %v", err)
	}
	if got, want := claimed.Status, "doing"; got != want {
		t.Fatalf("claimed status = %q, want %q", got, want)
	}
	if got, want := claimed.Lead, "bob"; got != want {
		t.Fatalf("claimed lead = %q, want %q", got, want)
	}

	closed, err := store.UpdateIssueStatus(issue.ID, "closed", "bob")
	if err != nil {
		t.Fatalf("close issue: %v", err)
	}
	if got, want := closed.Status, "closed"; got != want {
		t.Fatalf("closed status = %q, want %q", got, want)
	}

	activity, err := store.ListIssueActivity(issue.ID)
	if err != nil {
		t.Fatalf("list issue activity: %v", err)
	}
	if got, want := len(activity), 4; got != want {
		t.Fatalf("activity count = %d, want %d", got, want)
	}
	if got, want := activity[0].Type, "report"; got != want {
		t.Fatalf("first activity type = %q, want %q", got, want)
	}
	if got, want := activity[len(activity)-1].Type, "status"; got != want {
		t.Fatalf("last activity type = %q, want %q", got, want)
	}
}
