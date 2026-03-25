package issues

import (
	"fmt"
	"strings"
	"time"
)

type Kind string

const (
	KindFailure     Kind = "failure"
	KindAccuracy    Kind = "accuracy"
	KindPerformance Kind = "performance"
)

type Issue struct {
	ID        int       `json:"id"`
	Key       string    `json:"key"`
	Title     string    `json:"title"`
	Kind      Kind      `json:"kind"`
	Status    string    `json:"status"`
	Lead      string    `json:"lead,omitempty"`
	Reporter  string    `json:"reporter"`
	Summary   string    `json:"summary"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

type Note struct {
	ID        int       `json:"id"`
	IssueID   int       `json:"issue_id"`
	Author    string    `json:"author"`
	Content   string    `json:"content"`
	CreatedAt time.Time `json:"created_at"`
}

type Activity struct {
	ID        int       `json:"id"`
	IssueID   int       `json:"issue_id"`
	Actor     string    `json:"actor"`
	Type      string    `json:"type"`
	Text      string    `json:"text"`
	CreatedAt time.Time `json:"created_at"`
}

func NormalizeKind(kind string) (Kind, error) {
	switch Kind(strings.ToLower(strings.TrimSpace(kind))) {
	case KindFailure:
		return KindFailure, nil
	case KindAccuracy:
		return KindAccuracy, nil
	case KindPerformance:
		return KindPerformance, nil
	default:
		return "", fmt.Errorf("invalid issue kind: %s", kind)
	}
}

func NormalizeStatus(status string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "ready":
		return "ready", nil
	case "doing":
		return "doing", nil
	case "closed":
		return "closed", nil
	default:
		return "", fmt.Errorf("invalid issue status: %s", status)
	}
}

func IssueKey(id int) string {
	return fmt.Sprintf("ISSUE-%d", id)
}
