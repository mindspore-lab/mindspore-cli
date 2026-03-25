package bugs

type Service struct {
	store Store
}

func NewService(store Store) *Service {
	return &Service{store: store}
}

func (s *Service) ReportBug(title, reporter string, tags []string) (*Bug, error) {
	return s.store.CreateBug(title, reporter, tags)
}

func (s *Service) ListBugs(status string) ([]Bug, error) {
	return s.store.ListBugs(status)
}

func (s *Service) GetBug(id int) (*Bug, error) {
	return s.store.GetBug(id)
}

func (s *Service) ClaimBug(id int, lead string) error {
	return s.store.ClaimBug(id, lead)
}

func (s *Service) CloseBug(id int) error {
	return s.store.CloseBug(id)
}

func (s *Service) AddNote(bugID int, author, content string) (*Note, error) {
	return s.store.AddNote(bugID, author, content)
}

func (s *Service) GetActivity(bugID int) ([]Activity, error) {
	return s.store.ListActivity(bugID)
}

func (s *Service) DockSummary() (*DockData, error) {
	return s.store.DockSummary()
}
