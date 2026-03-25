package project

import "time"

type Task struct {
	ID        int       `json:"id"`
	Section   string    `json:"section"`
	Title     string    `json:"title"`
	Status    string    `json:"status"`
	Progress  int       `json:"progress"`
	Owner     string    `json:"owner,omitempty"`
	Due       string    `json:"due,omitempty"`
	Tags      string    `json:"tags,omitempty"`
	CreatedBy string    `json:"created_by"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

type Overview struct {
	Phase  string `json:"phase"`
	Owner  string `json:"owner"`
	Repo   string `json:"repo"`
	Branch string `json:"branch"`
}

type Snapshot struct {
	Overview Overview `json:"overview"`
	Tasks    []Task   `json:"tasks"`
}
