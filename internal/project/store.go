package project

type Store interface {
	GetSnapshot() (*Snapshot, error)
	CreateTask(section, title, owner, createdBy, due, tags string, progress *int) (*Task, error)
	UpdateTask(id int, title, owner, status, due, tags *string, progress *int) (*Task, error)
	DeleteTask(id int) error
	UpdateOverview(phase, owner, repo, branch string) (*Overview, error)
}
