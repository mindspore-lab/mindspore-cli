package server

import (
	"database/sql"
	"fmt"
	"strings"
	"time"

	"github.com/vigo999/mindspore-cli/internal/bugs"
	issuepkg "github.com/vigo999/mindspore-cli/internal/issues"
	"github.com/vigo999/mindspore-cli/internal/project"
)

type Store struct {
	db *sql.DB
}

func NewStore(db *sql.DB) (*Store, error) {
	s := &Store{db: db}
	if err := s.migrate(); err != nil {
		return nil, fmt.Errorf("migrate: %w", err)
	}
	return s, nil
}

func (s *Store) migrate() error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS bugs (
			id         INTEGER PRIMARY KEY AUTOINCREMENT,
			title      TEXT    NOT NULL,
			tags       TEXT    NOT NULL DEFAULT '',
			status     TEXT    NOT NULL DEFAULT 'open',
			lead       TEXT    NOT NULL DEFAULT '',
			reporter   TEXT    NOT NULL,
			created_at TEXT    NOT NULL,
			updated_at TEXT    NOT NULL
		)`,
		`CREATE TABLE IF NOT EXISTS notes (
			id         INTEGER PRIMARY KEY AUTOINCREMENT,
			bug_id     INTEGER NOT NULL REFERENCES bugs(id),
			author     TEXT    NOT NULL,
			content    TEXT    NOT NULL,
			created_at TEXT    NOT NULL
		)`,
		`CREATE TABLE IF NOT EXISTS activities (
			id         INTEGER PRIMARY KEY AUTOINCREMENT,
			bug_id     INTEGER NOT NULL REFERENCES bugs(id),
			actor      TEXT    NOT NULL,
			type       TEXT    NOT NULL,
			text       TEXT    NOT NULL DEFAULT '',
			created_at TEXT    NOT NULL
		)`,
		`CREATE TABLE IF NOT EXISTS issues (
			id         INTEGER PRIMARY KEY AUTOINCREMENT,
			title      TEXT    NOT NULL,
			kind       TEXT    NOT NULL,
			status     TEXT    NOT NULL DEFAULT 'ready',
			lead       TEXT    NOT NULL DEFAULT '',
			reporter   TEXT    NOT NULL,
			summary    TEXT    NOT NULL DEFAULT '',
			created_at TEXT    NOT NULL,
			updated_at TEXT    NOT NULL
		)`,
		`CREATE TABLE IF NOT EXISTS issue_notes (
			id         INTEGER PRIMARY KEY AUTOINCREMENT,
			issue_id   INTEGER NOT NULL REFERENCES issues(id),
			author     TEXT    NOT NULL,
			content    TEXT    NOT NULL,
			created_at TEXT    NOT NULL
		)`,
		`CREATE TABLE IF NOT EXISTS issue_activities (
			id         INTEGER PRIMARY KEY AUTOINCREMENT,
			issue_id   INTEGER NOT NULL REFERENCES issues(id),
			actor      TEXT    NOT NULL,
			type       TEXT    NOT NULL,
			text       TEXT    NOT NULL DEFAULT '',
			created_at TEXT    NOT NULL
		)`,
		`CREATE TABLE IF NOT EXISTS project_overview (
			id         INTEGER PRIMARY KEY CHECK (id = 1),
			phase      TEXT NOT NULL DEFAULT '',
			owner      TEXT NOT NULL DEFAULT '',
			repo       TEXT NOT NULL DEFAULT '',
			branch     TEXT NOT NULL DEFAULT '',
			updated_at TEXT NOT NULL
		)`,
		`INSERT OR IGNORE INTO project_overview (id, phase, owner, repo, branch, updated_at)
		 VALUES (1, '', '', '', '', datetime('now'))`,
		`CREATE TABLE IF NOT EXISTS project_tasks (
			id          INTEGER PRIMARY KEY AUTOINCREMENT,
			section     TEXT    NOT NULL DEFAULT 'tasks',
			title       TEXT    NOT NULL,
			status      TEXT    NOT NULL DEFAULT 'todo',
			progress    INTEGER NOT NULL DEFAULT 0,
			owner       TEXT    NOT NULL DEFAULT '',
			due         TEXT    NOT NULL DEFAULT '',
			tags        TEXT    NOT NULL DEFAULT '',
			created_by  TEXT    NOT NULL,
			created_at  TEXT    NOT NULL,
			updated_at  TEXT    NOT NULL
		)`,
		`CREATE TABLE IF NOT EXISTS user_sessions (
			user       TEXT PRIMARY KEY,
			last_seen  TEXT NOT NULL
		)`,
	}
	for _, stmt := range stmts {
		if _, err := s.db.Exec(stmt); err != nil {
			return fmt.Errorf("exec %q: %w", stmt[:40], err)
		}
	}
	if _, err := s.db.Exec(`ALTER TABLE bugs ADD COLUMN tags TEXT NOT NULL DEFAULT ''`); err != nil && !strings.Contains(strings.ToLower(err.Error()), "duplicate column name") {
		return fmt.Errorf("ensure bugs.tags column: %w", err)
	}
	if _, err := s.db.Exec(`ALTER TABLE project_tasks ADD COLUMN tags TEXT NOT NULL DEFAULT ''`); err != nil && !strings.Contains(strings.ToLower(err.Error()), "duplicate column name") {
		return fmt.Errorf("ensure project_tasks.tags column: %w", err)
	}
	return nil
}

func (s *Store) CreateBug(title, reporter string, tags []string) (*bugs.Bug, error) {
	now := time.Now().UTC().Format(time.RFC3339)
	tagValue := encodeBugTags(tags)
	res, err := s.db.Exec(
		`INSERT INTO bugs (title, tags, status, reporter, created_at, updated_at) VALUES (?, ?, 'open', ?, ?, ?)`,
		title, tagValue, reporter, now, now,
	)
	if err != nil {
		return nil, err
	}
	id, _ := res.LastInsertId()
	if _, err := s.db.Exec(
		`INSERT INTO activities (bug_id, actor, type, text, created_at) VALUES (?, ?, 'report', ?, ?)`,
		id, reporter, fmt.Sprintf("reported bug: %s", title), now,
	); err != nil {
		return nil, err
	}
	return s.GetBug(int(id))
}

func (s *Store) ListBugs(status string) ([]bugs.Bug, error) {
	query := `SELECT id, title, tags, status, lead, reporter, created_at, updated_at FROM bugs`
	var args []any
	if status != "" {
		query += ` WHERE status = ?`
		args = append(args, status)
	}
	query += ` ORDER BY updated_at DESC`
	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var bugList []bugs.Bug
	for rows.Next() {
		var b bugs.Bug
		var createdAt, updatedAt, tags string
		if err := rows.Scan(&b.ID, &b.Title, &tags, &b.Status, &b.Lead, &b.Reporter, &createdAt, &updatedAt); err != nil {
			return nil, err
		}
		b.Tags = decodeBugTags(tags)
		b.CreatedAt, _ = time.Parse(time.RFC3339, createdAt)
		b.UpdatedAt, _ = time.Parse(time.RFC3339, updatedAt)
		bugList = append(bugList, b)
	}
	return bugList, rows.Err()
}

func (s *Store) GetBug(id int) (*bugs.Bug, error) {
	var b bugs.Bug
	var createdAt, updatedAt, tags string
	err := s.db.QueryRow(
		`SELECT id, title, tags, status, lead, reporter, created_at, updated_at FROM bugs WHERE id = ?`, id,
	).Scan(&b.ID, &b.Title, &tags, &b.Status, &b.Lead, &b.Reporter, &createdAt, &updatedAt)
	if err != nil {
		return nil, err
	}
	b.Tags = decodeBugTags(tags)
	b.CreatedAt, _ = time.Parse(time.RFC3339, createdAt)
	b.UpdatedAt, _ = time.Parse(time.RFC3339, updatedAt)
	return &b, nil
}

func (s *Store) ClaimBug(id int, lead string) error {
	now := time.Now().UTC().Format(time.RFC3339)
	res, err := s.db.Exec(
		`UPDATE bugs SET lead = ?, status = 'doing', updated_at = ? WHERE id = ?`,
		lead, now, id,
	)
	if err != nil {
		return err
	}
	n, _ := res.RowsAffected()
	if n == 0 {
		return fmt.Errorf("bug %d not found", id)
	}
	_, err = s.db.Exec(
		`INSERT INTO activities (bug_id, actor, type, text, created_at) VALUES (?, ?, 'claim', ?, ?)`,
		id, lead, fmt.Sprintf("%s claimed bug", lead), now,
	)
	return err
}

func (s *Store) CloseBug(id int, user string) error {
	now := time.Now().UTC().Format(time.RFC3339)
	res, err := s.db.Exec(
		`UPDATE bugs SET status = 'closed', updated_at = ? WHERE id = ?`,
		now, id,
	)
	if err != nil {
		return err
	}
	n, _ := res.RowsAffected()
	if n == 0 {
		return fmt.Errorf("bug %d not found", id)
	}
	_, err = s.db.Exec(
		`INSERT INTO activities (bug_id, actor, type, text, created_at) VALUES (?, ?, 'close', ?, ?)`,
		id, user, fmt.Sprintf("%s closed bug", user), now,
	)
	return err
}

func (s *Store) AddNote(bugID int, author, content string) (*bugs.Note, error) {
	now := time.Now().UTC().Format(time.RFC3339)
	res, err := s.db.Exec(
		`INSERT INTO notes (bug_id, author, content, created_at) VALUES (?, ?, ?, ?)`,
		bugID, author, content, now,
	)
	if err != nil {
		return nil, err
	}
	noteID, _ := res.LastInsertId()
	if _, err := s.db.Exec(
		`INSERT INTO activities (bug_id, actor, type, text, created_at) VALUES (?, ?, 'note', ?, ?)`,
		bugID, author, fmt.Sprintf("added note: %s", content), now,
	); err != nil {
		return nil, err
	}
	if _, err := s.db.Exec(`UPDATE bugs SET updated_at = ? WHERE id = ?`, now, bugID); err != nil {
		return nil, err
	}
	createdAt, _ := time.Parse(time.RFC3339, now)
	return &bugs.Note{
		ID:        int(noteID),
		BugID:     bugID,
		Author:    author,
		Content:   content,
		CreatedAt: createdAt,
	}, nil
}

func (s *Store) ListActivity(bugID int) ([]bugs.Activity, error) {
	rows, err := s.db.Query(
		`SELECT id, bug_id, actor, type, text, created_at FROM activities WHERE bug_id = ? ORDER BY created_at ASC`,
		bugID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var acts []bugs.Activity
	for rows.Next() {
		var a bugs.Activity
		var createdAt string
		if err := rows.Scan(&a.ID, &a.BugID, &a.Actor, &a.Type, &a.Text, &createdAt); err != nil {
			return nil, err
		}
		a.CreatedAt, _ = time.Parse(time.RFC3339, createdAt)
		acts = append(acts, a)
	}
	return acts, rows.Err()
}

func (s *Store) DockSummary() (*bugs.DockData, error) {
	var openCount int
	if err := s.db.QueryRow(`SELECT COUNT(*) FROM bugs WHERE status IN ('open','doing')`).Scan(&openCount); err != nil {
		return nil, err
	}
	readyBugs, err := s.ListBugs("open")
	if err != nil {
		return nil, err
	}
	rows, err := s.db.Query(
		`SELECT id, bug_id, actor, type, text, created_at FROM activities ORDER BY created_at DESC LIMIT 10`,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var feed []bugs.Activity
	for rows.Next() {
		var a bugs.Activity
		var createdAt string
		if err := rows.Scan(&a.ID, &a.BugID, &a.Actor, &a.Type, &a.Text, &createdAt); err != nil {
			return nil, err
		}
		a.CreatedAt, _ = time.Parse(time.RFC3339, createdAt)
		feed = append(feed, a)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	onlineCount, _ := s.RecentUserCount(24 * time.Hour)

	return &bugs.DockData{
		OpenCount:   openCount,
		OnlineCount: onlineCount,
		ReadyBugs:   readyBugs,
		RecentFeed:  feed,
	}, nil
}

func (s *Store) CreateIssue(title string, kind issuepkg.Kind, reporter string) (*issuepkg.Issue, error) {
	now := time.Now().UTC().Format(time.RFC3339)
	res, err := s.db.Exec(
		`INSERT INTO issues (title, kind, status, reporter, summary, created_at, updated_at) VALUES (?, ?, 'ready', ?, '', ?, ?)`,
		title, string(kind), reporter, now, now,
	)
	if err != nil {
		return nil, err
	}
	id, _ := res.LastInsertId()
	if _, err := s.db.Exec(
		`INSERT INTO issue_activities (issue_id, actor, type, text, created_at) VALUES (?, ?, 'report', ?, ?)`,
		id, reporter, fmt.Sprintf("reported issue: %s", title), now,
	); err != nil {
		return nil, err
	}
	return s.GetIssue(int(id))
}

func (s *Store) ListIssues(status string) ([]issuepkg.Issue, error) {
	query := `SELECT id, title, kind, status, lead, reporter, summary, created_at, updated_at FROM issues`
	var args []any
	if status != "" {
		query += ` WHERE status = ?`
		args = append(args, status)
	}
	query += ` ORDER BY updated_at DESC`

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var issueList []issuepkg.Issue
	for rows.Next() {
		issue, err := scanIssue(rows)
		if err != nil {
			return nil, err
		}
		issueList = append(issueList, *issue)
	}
	return issueList, rows.Err()
}

func (s *Store) GetIssue(id int) (*issuepkg.Issue, error) {
	row := s.db.QueryRow(
		`SELECT id, title, kind, status, lead, reporter, summary, created_at, updated_at FROM issues WHERE id = ?`,
		id,
	)
	return scanIssue(row)
}

func (s *Store) AddIssueNote(issueID int, author, content string) (*issuepkg.Note, error) {
	now := time.Now().UTC().Format(time.RFC3339)
	res, err := s.db.Exec(
		`INSERT INTO issue_notes (issue_id, author, content, created_at) VALUES (?, ?, ?, ?)`,
		issueID, author, content, now,
	)
	if err != nil {
		return nil, err
	}
	noteID, _ := res.LastInsertId()
	if _, err := s.db.Exec(
		`INSERT INTO issue_activities (issue_id, actor, type, text, created_at) VALUES (?, ?, 'note', ?, ?)`,
		issueID, author, fmt.Sprintf("%s added note", author), now,
	); err != nil {
		return nil, err
	}
	if _, err := s.db.Exec(`UPDATE issues SET updated_at = ? WHERE id = ?`, now, issueID); err != nil {
		return nil, err
	}
	createdAt, _ := time.Parse(time.RFC3339, now)
	return &issuepkg.Note{
		ID:        int(noteID),
		IssueID:   issueID,
		Author:    author,
		Content:   content,
		CreatedAt: createdAt,
	}, nil
}

func (s *Store) ListIssueNotes(issueID int) ([]issuepkg.Note, error) {
	rows, err := s.db.Query(
		`SELECT id, issue_id, author, content, created_at FROM issue_notes WHERE issue_id = ? ORDER BY created_at ASC`,
		issueID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var notes []issuepkg.Note
	for rows.Next() {
		var note issuepkg.Note
		var createdAt string
		if err := rows.Scan(&note.ID, &note.IssueID, &note.Author, &note.Content, &createdAt); err != nil {
			return nil, err
		}
		note.CreatedAt, _ = time.Parse(time.RFC3339, createdAt)
		notes = append(notes, note)
	}
	return notes, rows.Err()
}

func (s *Store) ListIssueActivity(issueID int) ([]issuepkg.Activity, error) {
	rows, err := s.db.Query(
		`SELECT id, issue_id, actor, type, text, created_at FROM issue_activities WHERE issue_id = ? ORDER BY created_at ASC`,
		issueID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var acts []issuepkg.Activity
	for rows.Next() {
		var act issuepkg.Activity
		var createdAt string
		if err := rows.Scan(&act.ID, &act.IssueID, &act.Actor, &act.Type, &act.Text, &createdAt); err != nil {
			return nil, err
		}
		act.CreatedAt, _ = time.Parse(time.RFC3339, createdAt)
		acts = append(acts, act)
	}
	return acts, rows.Err()
}

func (s *Store) ClaimIssue(id int, lead string) (*issuepkg.Issue, error) {
	now := time.Now().UTC().Format(time.RFC3339)
	res, err := s.db.Exec(
		`UPDATE issues SET lead = ?, status = 'doing', updated_at = ? WHERE id = ?`,
		lead, now, id,
	)
	if err != nil {
		return nil, err
	}
	n, _ := res.RowsAffected()
	if n == 0 {
		return nil, fmt.Errorf("issue %d not found", id)
	}
	if _, err := s.db.Exec(
		`INSERT INTO issue_activities (issue_id, actor, type, text, created_at) VALUES (?, ?, 'claim', ?, ?)`,
		id, lead, fmt.Sprintf("%s took lead", lead), now,
	); err != nil {
		return nil, err
	}
	return s.GetIssue(id)
}

func (s *Store) UpdateIssueStatus(id int, status string, actor string) (*issuepkg.Issue, error) {
	normalized, err := issuepkg.NormalizeStatus(status)
	if err != nil {
		return nil, err
	}
	now := time.Now().UTC().Format(time.RFC3339)
	res, err := s.db.Exec(
		`UPDATE issues SET status = ?, updated_at = ? WHERE id = ?`,
		normalized, now, id,
	)
	if err != nil {
		return nil, err
	}
	n, _ := res.RowsAffected()
	if n == 0 {
		return nil, fmt.Errorf("issue %d not found", id)
	}
	if _, err := s.db.Exec(
		`INSERT INTO issue_activities (issue_id, actor, type, text, created_at) VALUES (?, ?, 'status', ?, ?)`,
		id, actor, fmt.Sprintf("%s changed status to %s", actor, normalized), now,
	); err != nil {
		return nil, err
	}
	return s.GetIssue(id)
}

func encodeBugTags(tags []string) string {
	return strings.Join(bugs.NormalizeTags(tags), ",")
}

func decodeBugTags(raw string) []string {
	return bugs.NormalizeTags(strings.Split(raw, ","))
}

type issueScanner interface {
	Scan(dest ...any) error
}

func scanIssue(scanner issueScanner) (*issuepkg.Issue, error) {
	var issue issuepkg.Issue
	var kind string
	var createdAt, updatedAt string
	if err := scanner.Scan(&issue.ID, &issue.Title, &kind, &issue.Status, &issue.Lead, &issue.Reporter, &issue.Summary, &createdAt, &updatedAt); err != nil {
		return nil, err
	}
	issue.Kind = issuepkg.Kind(kind)
	issue.Key = issuepkg.IssueKey(issue.ID)
	issue.CreatedAt, _ = time.Parse(time.RFC3339, createdAt)
	issue.UpdatedAt, _ = time.Parse(time.RFC3339, updatedAt)
	return &issue, nil
}

// --- Session methods ---

func (s *Store) TouchSession(user string) {
	now := time.Now().UTC().Format(time.RFC3339)
	s.db.Exec(`INSERT INTO user_sessions (user, last_seen) VALUES (?, ?)
		ON CONFLICT(user) DO UPDATE SET last_seen = ?`, user, now, now)
}

func (s *Store) RecentUserCount(since time.Duration) (int, error) {
	cutoff := time.Now().UTC().Add(-since).Format(time.RFC3339)
	var count int
	err := s.db.QueryRow(`SELECT COUNT(*) FROM user_sessions WHERE last_seen >= ?`, cutoff).Scan(&count)
	return count, err
}

// --- Project methods ---

func (s *Store) GetProjectSnapshot() (*project.Snapshot, error) {
	var ov project.Overview
	err := s.db.QueryRow(`SELECT phase, owner, repo, branch FROM project_overview WHERE id = 1`).
		Scan(&ov.Phase, &ov.Owner, &ov.Repo, &ov.Branch)
	if err != nil {
		return nil, err
	}

	rows, err := s.db.Query(
		`SELECT id, section, title, status, progress, owner, due, tags, created_by, created_at, updated_at
		 FROM project_tasks ORDER BY id ASC`,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var tasks []project.Task
	for rows.Next() {
		var t project.Task
		var createdAt, updatedAt string
		if err := rows.Scan(&t.ID, &t.Section, &t.Title, &t.Status, &t.Progress, &t.Owner, &t.Due, &t.Tags, &t.CreatedBy, &createdAt, &updatedAt); err != nil {
			return nil, err
		}
		t.CreatedAt, _ = time.Parse(time.RFC3339, createdAt)
		t.UpdatedAt, _ = time.Parse(time.RFC3339, updatedAt)
		tasks = append(tasks, t)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if tasks == nil {
		tasks = []project.Task{}
	}
	return &project.Snapshot{Overview: ov, Tasks: tasks}, nil
}

func (s *Store) CreateProjectTask(section, title, owner, createdBy, due, tags string, progress *int) (*project.Task, error) {
	now := time.Now().UTC().Format(time.RFC3339)
	prog := 0
	if progress != nil {
		prog = *progress
	}
	res, err := s.db.Exec(
		`INSERT INTO project_tasks (section, title, status, progress, owner, due, tags, created_by, created_at, updated_at)
		 VALUES (?, ?, 'todo', ?, ?, ?, ?, ?, ?, ?)`,
		section, title, prog, owner, due, tags, createdBy, now, now,
	)
	if err != nil {
		return nil, err
	}
	id, _ := res.LastInsertId()
	return s.getProjectTask(int(id))
}

func (s *Store) UpdateProjectTask(id int, title, owner, status, due, tags *string, progress *int) (*project.Task, error) {
	now := time.Now().UTC().Format(time.RFC3339)
	if title != nil {
		if _, err := s.db.Exec(`UPDATE project_tasks SET title = ?, updated_at = ? WHERE id = ?`, *title, now, id); err != nil {
			return nil, err
		}
	}
	if owner != nil {
		if _, err := s.db.Exec(`UPDATE project_tasks SET owner = ?, updated_at = ? WHERE id = ?`, *owner, now, id); err != nil {
			return nil, err
		}
	}
	if status != nil {
		if _, err := s.db.Exec(`UPDATE project_tasks SET status = ?, updated_at = ? WHERE id = ?`, *status, now, id); err != nil {
			return nil, err
		}
	}
	if due != nil {
		if _, err := s.db.Exec(`UPDATE project_tasks SET due = ?, updated_at = ? WHERE id = ?`, *due, now, id); err != nil {
			return nil, err
		}
	}
	if tags != nil {
		if _, err := s.db.Exec(`UPDATE project_tasks SET tags = ?, updated_at = ? WHERE id = ?`, *tags, now, id); err != nil {
			return nil, err
		}
	}
	if progress != nil {
		if _, err := s.db.Exec(`UPDATE project_tasks SET progress = ?, updated_at = ? WHERE id = ?`, *progress, now, id); err != nil {
			return nil, err
		}
	}
	return s.getProjectTask(id)
}

func (s *Store) DeleteProjectTask(id int) error {
	res, err := s.db.Exec(`DELETE FROM project_tasks WHERE id = ?`, id)
	if err != nil {
		return err
	}
	n, _ := res.RowsAffected()
	if n == 0 {
		return fmt.Errorf("task %d not found", id)
	}
	return nil
}

func (s *Store) UpdateProjectOverview(phase, owner, repo, branch string) (*project.Overview, error) {
	now := time.Now().UTC().Format(time.RFC3339)
	if phase != "" {
		if _, err := s.db.Exec(`UPDATE project_overview SET phase = ?, updated_at = ? WHERE id = 1`, phase, now); err != nil {
			return nil, err
		}
	}
	if owner != "" {
		if _, err := s.db.Exec(`UPDATE project_overview SET owner = ?, updated_at = ? WHERE id = 1`, owner, now); err != nil {
			return nil, err
		}
	}
	if repo != "" {
		if _, err := s.db.Exec(`UPDATE project_overview SET repo = ?, updated_at = ? WHERE id = 1`, repo, now); err != nil {
			return nil, err
		}
	}
	if branch != "" {
		if _, err := s.db.Exec(`UPDATE project_overview SET branch = ?, updated_at = ? WHERE id = 1`, branch, now); err != nil {
			return nil, err
		}
	}
	var ov project.Overview
	err := s.db.QueryRow(`SELECT phase, owner, repo, branch FROM project_overview WHERE id = 1`).
		Scan(&ov.Phase, &ov.Owner, &ov.Repo, &ov.Branch)
	if err != nil {
		return nil, err
	}
	return &ov, nil
}

func (s *Store) getProjectTask(id int) (*project.Task, error) {
	var t project.Task
	var createdAt, updatedAt string
	err := s.db.QueryRow(
		`SELECT id, section, title, status, progress, owner, due, tags, created_by, created_at, updated_at
		 FROM project_tasks WHERE id = ?`, id,
	).Scan(&t.ID, &t.Section, &t.Title, &t.Status, &t.Progress, &t.Owner, &t.Due, &t.Tags, &t.CreatedBy, &createdAt, &updatedAt)
	if err != nil {
		return nil, err
	}
	t.CreatedAt, _ = time.Parse(time.RFC3339, createdAt)
	t.UpdatedAt, _ = time.Parse(time.RFC3339, updatedAt)
	return &t, nil
}
