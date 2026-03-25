package bugs

import (
	"strings"
	"time"
)

type Bug struct {
	ID        int       `json:"id"`
	Title     string    `json:"title"`
	Tags      []string  `json:"tags,omitempty"`
	Status    string    `json:"status"`
	Lead      string    `json:"lead,omitempty"`
	Reporter  string    `json:"reporter"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

type Note struct {
	ID        int       `json:"id"`
	BugID     int       `json:"bug_id"`
	Author    string    `json:"author"`
	Content   string    `json:"content"`
	CreatedAt time.Time `json:"created_at"`
}

type Activity struct {
	ID        int       `json:"id"`
	BugID     int       `json:"bug_id"`
	Actor     string    `json:"actor"`
	Type      string    `json:"type"`
	Text      string    `json:"text"`
	CreatedAt time.Time `json:"created_at"`
}

type DockData struct {
	OpenCount   int        `json:"open_count"`
	OnlineCount int        `json:"online_count"`
	ReadyBugs   []Bug      `json:"ready_bugs"`
	RecentFeed  []Activity `json:"recent_feed"`
}

func NormalizeTags(tags []string) []string {
	if len(tags) == 0 {
		return nil
	}

	seen := make(map[string]struct{}, len(tags))
	out := make([]string, 0, len(tags))
	for _, tag := range tags {
		tag = strings.ToLower(strings.TrimSpace(tag))
		tag = strings.Join(strings.Fields(tag), "-")
		if tag == "" {
			continue
		}
		if _, ok := seen[tag]; ok {
			continue
		}
		seen[tag] = struct{}{}
		out = append(out, tag)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
