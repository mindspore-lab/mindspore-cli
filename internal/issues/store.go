package issues

type Store interface {
	CreateIssue(title string, kind Kind, reporter string) (*Issue, error)
	ListIssues(status string) ([]Issue, error)
	GetIssue(id int) (*Issue, error)
	AddNote(issueID int, author, content string) (*Note, error)
	ListNotes(issueID int) ([]Note, error)
	ListActivity(issueID int) ([]Activity, error)
	ClaimIssue(id int, lead string) (*Issue, error)
	UpdateStatus(id int, status string, actor string) (*Issue, error)
}
