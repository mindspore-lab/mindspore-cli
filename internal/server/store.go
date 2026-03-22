package server

import (
	"database/sql"
	"fmt"
	"time"

	"github.com/vigo999/ms-cli/internal/issues"
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
	}
	for _, stmt := range stmts {
		if _, err := s.db.Exec(stmt); err != nil {
			return fmt.Errorf("exec %q: %w", stmt[:40], err)
		}
	}
	return nil
}

func (s *Store) CreateBug(title, reporter string) (*issues.Bug, error) {
	now := time.Now().UTC().Format(time.RFC3339)
	res, err := s.db.Exec(
		`INSERT INTO bugs (title, status, reporter, created_at, updated_at) VALUES (?, 'open', ?, ?, ?)`,
		title, reporter, now, now,
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

func (s *Store) ListBugs(status string) ([]issues.Bug, error) {
	query := `SELECT id, title, status, lead, reporter, created_at, updated_at FROM bugs`
	var args []any
	if status != "" {
		query += ` WHERE status = ?`
		args = append(args, status)
	}
	query += ` ORDER BY created_at DESC`
	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var bugs []issues.Bug
	for rows.Next() {
		var b issues.Bug
		var createdAt, updatedAt string
		if err := rows.Scan(&b.ID, &b.Title, &b.Status, &b.Lead, &b.Reporter, &createdAt, &updatedAt); err != nil {
			return nil, err
		}
		b.CreatedAt, _ = time.Parse(time.RFC3339, createdAt)
		b.UpdatedAt, _ = time.Parse(time.RFC3339, updatedAt)
		bugs = append(bugs, b)
	}
	return bugs, rows.Err()
}

func (s *Store) GetBug(id int) (*issues.Bug, error) {
	var b issues.Bug
	var createdAt, updatedAt string
	err := s.db.QueryRow(
		`SELECT id, title, status, lead, reporter, created_at, updated_at FROM bugs WHERE id = ?`, id,
	).Scan(&b.ID, &b.Title, &b.Status, &b.Lead, &b.Reporter, &createdAt, &updatedAt)
	if err != nil {
		return nil, err
	}
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

func (s *Store) AddNote(bugID int, author, content string) (*issues.Note, error) {
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
	return &issues.Note{
		ID:        int(noteID),
		BugID:     bugID,
		Author:    author,
		Content:   content,
		CreatedAt: createdAt,
	}, nil
}

func (s *Store) ListActivity(bugID int) ([]issues.Activity, error) {
	rows, err := s.db.Query(
		`SELECT id, bug_id, actor, type, text, created_at FROM activities WHERE bug_id = ? ORDER BY created_at ASC`,
		bugID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var acts []issues.Activity
	for rows.Next() {
		var a issues.Activity
		var createdAt string
		if err := rows.Scan(&a.ID, &a.BugID, &a.Actor, &a.Type, &a.Text, &createdAt); err != nil {
			return nil, err
		}
		a.CreatedAt, _ = time.Parse(time.RFC3339, createdAt)
		acts = append(acts, a)
	}
	return acts, rows.Err()
}

func (s *Store) DockSummary() (*issues.DockData, error) {
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
	var feed []issues.Activity
	for rows.Next() {
		var a issues.Activity
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
	return &issues.DockData{
		OpenCount:  openCount,
		ReadyBugs:  readyBugs,
		RecentFeed: feed,
	}, nil
}
