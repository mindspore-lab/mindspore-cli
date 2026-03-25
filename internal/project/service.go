package project

type Service struct {
	store Store
}

func NewService(store Store) *Service {
	return &Service{store: store}
}

func (s *Service) Snapshot() (*Snapshot, error) {
	return s.store.GetSnapshot()
}

func (s *Service) AddTask(section, title, owner, createdBy, due, tags string, progress *int) (*Task, error) {
	return s.store.CreateTask(section, title, owner, createdBy, due, tags, progress)
}

func (s *Service) UpdateTask(id int, title, owner, status, due, tags *string, progress *int) (*Task, error) {
	return s.store.UpdateTask(id, title, owner, status, due, tags, progress)
}

func (s *Service) RemoveTask(id int) error {
	return s.store.DeleteTask(id)
}

func (s *Service) UpdateOverview(phase, owner, repo, branch string) (*Overview, error) {
	return s.store.UpdateOverview(phase, owner, repo, branch)
}
