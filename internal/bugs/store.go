package bugs

type Store interface {
	CreateBug(title, reporter string, tags []string) (*Bug, error)
	ListBugs(status string) ([]Bug, error)
	GetBug(id int) (*Bug, error)
	ClaimBug(id int, lead string) error
	CloseBug(id int) error
	AddNote(bugID int, author, content string) (*Note, error)
	ListActivity(bugID int) ([]Activity, error)
	DockSummary() (*DockData, error)
}
