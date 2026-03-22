package issues

import "time"

type Bug struct {
	ID        int       `json:"id"`
	Title     string    `json:"title"`
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
	OpenCount  int        `json:"open_count"`
	ReadyBugs  []Bug      `json:"ready_bugs"`
	RecentFeed []Activity `json:"recent_feed"`
}
