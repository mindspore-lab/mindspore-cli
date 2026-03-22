package issues

type Store interface {
	CreateBug(title, reporter string) (*Bug, error)
	ListBugs(status string) ([]Bug, error)
	GetBug(id int) (*Bug, error)
	ClaimBug(id int, lead string) error
	AddNote(bugID int, author, content string) (*Note, error)
	ListActivity(bugID int) ([]Activity, error)
	DockSummary() (*DockData, error)
}
