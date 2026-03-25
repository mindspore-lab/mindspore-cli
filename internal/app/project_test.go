package app

import (
	"strings"
	"testing"

	"github.com/vigo999/ms-cli/internal/project"
	"github.com/vigo999/ms-cli/ui/model"
)

// mockProjectStore implements project.Store for testing.
type mockProjectStore struct {
	snapshot *project.Snapshot
	tasks    []project.Task
	nextID   int
}

func newMockProjectStore() *mockProjectStore {
	return &mockProjectStore{
		snapshot: &project.Snapshot{
			Overview: project.Overview{},
			Tasks:    []project.Task{},
		},
		nextID: 1,
	}
}

func (m *mockProjectStore) GetSnapshot() (*project.Snapshot, error) {
	snap := &project.Snapshot{
		Overview: m.snapshot.Overview,
		Tasks:    make([]project.Task, len(m.tasks)),
	}
	copy(snap.Tasks, m.tasks)
	return snap, nil
}

func (m *mockProjectStore) CreateTask(section, title, owner, createdBy, due, tags string, progress *int) (*project.Task, error) {
	prog := 0
	if progress != nil {
		prog = *progress
	}
	t := project.Task{
		ID:        m.nextID,
		Section:   section,
		Title:     title,
		Status:    "todo",
		Progress:  prog,
		Owner:     owner,
		Due:       due,
		CreatedBy: createdBy,
	}
	m.nextID++
	m.tasks = append(m.tasks, t)
	return &t, nil
}

func (m *mockProjectStore) UpdateTask(id int, title, owner, status, due, tags *string, progress *int) (*project.Task, error) {
	for i := range m.tasks {
		if m.tasks[i].ID == id {
			if title != nil {
				m.tasks[i].Title = *title
			}
			if owner != nil {
				m.tasks[i].Owner = *owner
			}
			if status != nil {
				m.tasks[i].Status = *status
			}
			if progress != nil {
				m.tasks[i].Progress = *progress
			}
			t := m.tasks[i]
			return &t, nil
		}
	}
	return nil, nil
}

func (m *mockProjectStore) DeleteTask(id int) error {
	for i := range m.tasks {
		if m.tasks[i].ID == id {
			m.tasks = append(m.tasks[:i], m.tasks[i+1:]...)
			return nil
		}
	}
	return nil
}

func (m *mockProjectStore) UpdateOverview(phase, owner, repo, branch string) (*project.Overview, error) {
	if phase != "" {
		m.snapshot.Overview.Phase = phase
	}
	if owner != "" {
		m.snapshot.Overview.Owner = owner
	}
	if repo != "" {
		m.snapshot.Overview.Repo = repo
	}
	if branch != "" {
		m.snapshot.Overview.Branch = branch
	}
	ov := m.snapshot.Overview
	return &ov, nil
}

func TestCmdProjectStreamsFormattedSnapshot(t *testing.T) {
	orig := runProjectGit
	defer func() { runProjectGit = orig }()

	root := t.TempDir()

	runProjectGit = func(workDir string, args ...string) (string, error) {
		switch strings.Join(args, " ") {
		case "rev-parse --show-toplevel":
			return root, nil
		case "symbolic-ref --short HEAD":
			return "refactor-arch-3", nil
		case "status --short":
			return " M ui/app.go\nA  ui/model/project_test.go\n?? docs/project.yaml", nil
		case "rev-list --left-right --count @{upstream}...HEAD":
			return "2 5", nil
		default:
			t.Fatalf("unexpected git args: %v", args)
			return "", nil
		}
	}

	store := newMockProjectStore()
	store.snapshot.Overview = project.Overview{Phase: "refactor", Owner: "travis", Repo: "github.com/vigo999/ms-cli", Branch: "refactor-arch-4.2"}
	store.tasks = []project.Task{
		{ID: 1, Section: "tasks", Title: "project status command", Status: "done", Progress: 100, Owner: "travis"},
		{ID: 2, Section: "tasks", Title: "status schema draft", Status: "doing", Progress: 60, Owner: "alice"},
		{ID: 3, Section: "tomorrow", Title: "define schema", Status: "todo", Progress: 0, Owner: "travis"},
	}

	app := &Application{
		WorkDir:        root,
		EventCh:        make(chan model.Event, 8),
		projectService: project.NewService(store),
	}

	app.cmdProject(nil)

	ev := drainUntilEventType(t, app, model.AgentReply)
	for _, want := range []string{
		"phase: refactor",
		"owner: travis",
		"repo: github.com/vigo999/ms-cli",
		"branch: refactor-arch-4.2",
		"project status command",
		"status schema draft",
		"define schema",
		"[ OVERVIEW ]",
		"[ TASKS ]",
	} {
		if !strings.Contains(ev.Message, want) {
			t.Fatalf("expected project snapshot to contain %q, got:\n%s", want, ev.Message)
		}
	}
}

func TestCmdProjectCloseExplainsStreamMode(t *testing.T) {
	app := &Application{EventCh: make(chan model.Event, 4)}

	app.cmdProject([]string{"close"})

	ev := drainUntilEventType(t, app, model.AgentReply)
	if !strings.Contains(ev.Message, "stream-only") {
		t.Fatalf("expected stream-only explanation, got %q", ev.Message)
	}
}

func TestCmdProjectFallbackWhenNotLoggedIn(t *testing.T) {
	orig := runProjectGit
	defer func() { runProjectGit = orig }()

	root := t.TempDir()

	runProjectGit = func(workDir string, args ...string) (string, error) {
		switch strings.Join(args, " ") {
		case "rev-parse --show-toplevel":
			return root, nil
		case "symbolic-ref --short HEAD":
			return "main", nil
		case "status --short":
			return "", nil
		case "rev-list --left-right --count @{upstream}...HEAD":
			return "0 0", nil
		default:
			t.Fatalf("unexpected git args: %v", args)
			return "", nil
		}
	}

	app := &Application{
		WorkDir: root,
		EventCh: make(chan model.Event, 8),
	}

	app.cmdProject(nil)

	ev := drainUntilEventType(t, app, model.AgentReply)
	if !strings.Contains(ev.Message, "working tree is clean") {
		t.Fatalf("expected fallback git status, got:\n%s", ev.Message)
	}
}

func TestHandleCommandProjectAddCreatesTask(t *testing.T) {
	orig := runProjectGit
	defer func() { runProjectGit = orig }()

	root := t.TempDir()

	runProjectGit = func(workDir string, args ...string) (string, error) {
		switch strings.Join(args, " ") {
		case "rev-parse --show-toplevel":
			return root, nil
		case "symbolic-ref --short HEAD":
			return "refactor-arch-4", nil
		case "status --short":
			return "", nil
		case "rev-list --left-right --count @{upstream}...HEAD":
			return "0 0", nil
		default:
			t.Fatalf("unexpected git args: %v", args)
			return "", nil
		}
	}

	store := newMockProjectStore()
	app := &Application{
		WorkDir:        root,
		EventCh:        make(chan model.Event, 8),
		projectService: project.NewService(store),
		issueUser:      "alice",
		issueRole:      "admin",
	}

	app.handleCommand(`/project add tasks "new task title" --owner bob --progress 30`)

	ev := drainUntilEventType(t, app, model.AgentReply)
	if !strings.Contains(ev.Message, "created task #1") || !strings.Contains(ev.Message, "new task title") {
		t.Fatalf("expected task creation confirmation, got:\n%s", ev.Message)
	}
	if len(store.tasks) != 1 {
		t.Fatalf("expected 1 task in store, got %d", len(store.tasks))
	}
	if store.tasks[0].Title != "new task title" || store.tasks[0].Owner != "bob" || store.tasks[0].Progress != 30 {
		t.Fatalf("unexpected task: %+v", store.tasks[0])
	}
}

func TestHandleCommandProjectUpdateModifiesTask(t *testing.T) {
	orig := runProjectGit
	defer func() { runProjectGit = orig }()

	root := t.TempDir()

	runProjectGit = func(workDir string, args ...string) (string, error) {
		switch strings.Join(args, " ") {
		case "rev-parse --show-toplevel":
			return root, nil
		case "symbolic-ref --short HEAD":
			return "refactor-arch-4", nil
		case "status --short":
			return "", nil
		case "rev-list --left-right --count @{upstream}...HEAD":
			return "0 0", nil
		default:
			t.Fatalf("unexpected git args: %v", args)
			return "", nil
		}
	}

	store := newMockProjectStore()
	store.tasks = []project.Task{
		{ID: 1, Section: "tasks", Title: "existing task", Status: "todo", Progress: 20, Owner: "alice"},
	}

	app := &Application{
		WorkDir:        root,
		EventCh:        make(chan model.Event, 8),
		projectService: project.NewService(store),
		issueUser:      "alice",
		issueRole:      "admin",
	}

	app.handleCommand(`/project update 1 --title "updated task" --owner carol --progress 80`)

	ev := drainUntilEventType(t, app, model.AgentReply)
	if !strings.Contains(ev.Message, "updated:") {
		t.Fatalf("expected update confirmation, got:\n%s", ev.Message)
	}
	if store.tasks[0].Title != "updated task" || store.tasks[0].Owner != "carol" || store.tasks[0].Progress != 80 {
		t.Fatalf("unexpected task: %+v", store.tasks[0])
	}
}

func TestHandleCommandProjectRemoveDeletesTask(t *testing.T) {
	orig := runProjectGit
	defer func() { runProjectGit = orig }()

	root := t.TempDir()

	runProjectGit = func(workDir string, args ...string) (string, error) {
		switch strings.Join(args, " ") {
		case "rev-parse --show-toplevel":
			return root, nil
		case "symbolic-ref --short HEAD":
			return "refactor-arch-4", nil
		case "status --short":
			return "", nil
		case "rev-list --left-right --count @{upstream}...HEAD":
			return "0 0", nil
		default:
			t.Fatalf("unexpected git args: %v", args)
			return "", nil
		}
	}

	store := newMockProjectStore()
	store.tasks = []project.Task{
		{ID: 1, Section: "tasks", Title: "existing task", Status: "todo", Progress: 20, Owner: "alice"},
		{ID: 2, Section: "tasks", Title: "keep task", Status: "todo", Progress: 40, Owner: "bob"},
	}

	app := &Application{
		WorkDir:        root,
		EventCh:        make(chan model.Event, 8),
		projectService: project.NewService(store),
		issueUser:      "alice",
		issueRole:      "admin",
	}

	app.handleCommand(`/project rm 1`)

	ev := drainUntilEventType(t, app, model.AgentReply)
	if !strings.Contains(ev.Message, "removed:") {
		t.Fatalf("expected removal confirmation, got:\n%s", ev.Message)
	}
	if len(store.tasks) != 1 {
		t.Fatalf("expected 1 task remaining, got %d", len(store.tasks))
	}
	if store.tasks[0].Title != "keep task" {
		t.Fatalf("expected keep task to remain, got: %+v", store.tasks[0])
	}
}
