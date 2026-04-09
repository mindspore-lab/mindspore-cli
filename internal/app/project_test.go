package app

import (
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/mindspore-lab/mindspore-cli/internal/project"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

var projectANSIPattern = regexp.MustCompile(`\x1b\[[0-9;]*m`)

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
	store.snapshot.Overview = project.Overview{Phase: "refactor", Owner: "travis", Repo: "github.com/mindspore-lab/mindspore-cli", Branch: "refactor-arch-4.2"}
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
	if !ev.RawANSI {
		t.Fatal("expected /project snapshot to be marked RawANSI")
	}
	for _, want := range []string{
		"phase: refactor",
		"owner: travis",
		"repo: github.com/mindspore-lab/mindspore-cli",
		"branch: refactor-arch-4.2",
		"1 staged",
		"ahead 5",
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
	if !projectANSIPattern.MatchString(ev.Message) {
		t.Fatalf("expected /project snapshot to contain ANSI styling, got:\n%q", ev.Message)
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
	t.Setenv("HOME", t.TempDir())

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
	if !ev.RawANSI {
		t.Fatal("expected fallback /project snapshot to be marked RawANSI")
	}
	for _, want := range []string{
		"[ OVERVIEW ]",
		"repo: " + filepath.Base(root),
		"root: " + root,
	} {
		if !strings.Contains(ev.Message, want) {
			t.Fatalf("expected fallback overview to contain %q, got:\n%s", want, ev.Message)
		}
	}
}

func TestRenderTaskLinesWrapsLongDescription(t *testing.T) {
	lines := renderTaskLines([]projectTask{{
		ID:       "29",
		Title:    "decouple agent session with background training process and keep description fully visible without ellipsis",
		Status:   "todo",
		Progress: 0,
		Owner:    "vigo",
		Tags:     "mscli,session",
	}})

	got := strings.Join(lines, "\n")
	if strings.Contains(got, "...") {
		t.Fatalf("expected wrapped task description without ellipsis, got:\n%s", got)
	}
	if !strings.Contains(got, "decouple agent session") {
		t.Fatalf("expected first part of description, got:\n%s", got)
	}
	if !strings.Contains(got, "process and keep description fully visible without") {
		t.Fatalf("expected wrapped continuation content, got:\n%s", got)
	}
	if !strings.Contains(got, "\n") || !strings.Contains(got, "ellipsis") {
		t.Fatalf("expected wrapped continuation line, got:\n%s", got)
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

func TestRenderTaskLines_TagsColumn(t *testing.T) {
	tasks := []projectTask{
		{ID: "1", Title: "add startup detection", Status: "done", Progress: 100, Owner: "alice", Tags: "startup,config"},
		{ID: "2", Title: "build setup popup", Status: "doing", Progress: 60, Owner: "bob", Tags: "ui,model"},
	}
	lines := renderTaskLines(tasks)
	if len(lines) != 2 {
		t.Fatalf("expected 2 lines, got %d", len(lines))
	}
	for i, task := range tasks {
		if !strings.Contains(lines[i], task.Tags) {
			t.Errorf("line %d: expected tags %q, got:\n%s", i, task.Tags, lines[i])
		}
		if !strings.Contains(lines[i], task.Title) {
			t.Errorf("line %d: expected title %q, got:\n%s", i, task.Title, lines[i])
		}
	}
}

func TestRenderTaskLines_EmptyTags(t *testing.T) {
	tasks := []projectTask{
		{ID: "1", Title: "task with no tags", Status: "todo", Progress: 0, Owner: "alice"},
		{ID: "2", Title: "another task", Status: "doing", Progress: 50, Owner: "bob"},
	}
	lines := renderTaskLines(tasks)
	if len(lines) != 2 {
		t.Fatalf("expected 2 lines, got %d", len(lines))
	}
	for i, task := range tasks {
		if !strings.Contains(lines[i], task.Title) {
			t.Errorf("line %d: expected title %q, got:\n%s", i, task.Title, lines[i])
		}
		if !strings.Contains(lines[i], task.Owner) {
			t.Errorf("line %d: expected owner %q, got:\n%s", i, task.Owner, lines[i])
		}
	}
}

func TestRenderMilestoneLines_TagsColumn(t *testing.T) {
	milestones := []projectTask{
		{Title: "v1.0 release", Progress: 80, Tags: "release,stable"},
		{Title: "beta launch", Progress: 50, Tags: "beta"},
	}
	lines := renderMilestoneLines(milestones)
	if len(lines) != 2 {
		t.Fatalf("expected 2 lines, got %d", len(lines))
	}
	for i, m := range milestones {
		if !strings.Contains(lines[i], m.Tags) {
			t.Errorf("line %d: expected tags %q, got:\n%s", i, m.Tags, lines[i])
		}
		if !strings.Contains(lines[i], m.Title) {
			t.Errorf("line %d: expected title %q, got:\n%s", i, m.Title, lines[i])
		}
	}
}

func TestRenderTaskLines_LongSingleTokenPreservedWithoutEllipsis(t *testing.T) {
	longTitle := strings.Repeat("a", 120)
	tasks := []projectTask{
		{ID: "1", Title: longTitle, Status: "doing", Progress: 50, Owner: "alice", Tags: "long"},
	}
	lines := renderTaskLines(tasks)
	if len(lines) != 1 {
		t.Fatalf("expected 1 line for single-token title, got %d", len(lines))
	}
	if strings.Contains(lines[0], "...") {
		t.Errorf("expected long single-token title to be preserved without ellipsis, got:\n%s", lines[0])
	}
	if !strings.Contains(lines[0], longTitle) {
		t.Errorf("expected full single-token title to be preserved, got:\n%s", lines[0])
	}
}

func TestTruncateStr(t *testing.T) {
	tests := []struct {
		input string
		max   int
		want  string
	}{
		{"short", 10, "short"},
		{"exactly ten", 11, "exactly ten"},
		{"this is a long string", 10, "this is..."},
		{"abc", 3, "abc"},
		{"abcd", 3, "abc"},
		{"", 5, ""},
		{"hello", 0, ""},
		{"hello", -1, ""},
		{"hello", 4, "h..."},
	}
	for _, tt := range tests {
		got := truncateStr(tt.input, tt.max)
		if got != tt.want {
			t.Errorf("truncateStr(%q, %d) = %q, want %q", tt.input, tt.max, got, tt.want)
		}
	}
}
