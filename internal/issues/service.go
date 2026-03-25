package issues

type Service struct {
	store Store
}

func NewService(store Store) *Service {
	return &Service{store: store}
}

func (s *Service) CreateIssue(title string, kind Kind, reporter string) (*Issue, error) {
	return s.store.CreateIssue(title, kind, reporter)
}

func (s *Service) ListIssues(status string) ([]Issue, error) {
	return s.store.ListIssues(status)
}

func (s *Service) GetIssue(id int) (*Issue, error) {
	return s.store.GetIssue(id)
}

func (s *Service) AddNote(issueID int, author, content string) (*Note, error) {
	return s.store.AddNote(issueID, author, content)
}

func (s *Service) ListNotes(issueID int) ([]Note, error) {
	return s.store.ListNotes(issueID)
}

func (s *Service) GetActivity(issueID int) ([]Activity, error) {
	return s.store.ListActivity(issueID)
}

func (s *Service) ClaimIssue(id int, lead string) (*Issue, error) {
	return s.store.ClaimIssue(id, lead)
}

func (s *Service) UpdateStatus(id int, status string, actor string) (*Issue, error) {
	return s.store.UpdateStatus(id, status, actor)
}
