package app

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
)

var (
	authStatePathOverride  string
	modelStatePathOverride string
)

type providerAuthState struct {
	Providers map[string]providerAuthEntry `json:"providers,omitempty"`
}

type providerAuthEntry struct {
	ProviderID string `json:"provider_id"`
	APIKey     string `json:"api_key,omitempty"`
}

type modelSelectionState struct {
	Active    *modelRef  `json:"active,omitempty"`
	Recents   []modelRef `json:"recent,omitempty"`
	Favorites []modelRef `json:"favorites,omitempty"`
}

type modelRef struct {
	ProviderID string `json:"provider_id"`
	ModelID    string `json:"model_id"`
}

func authStatePath() string {
	if authStatePathOverride != "" {
		return authStatePathOverride
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".mscli", "auth.json")
}

func modelStatePath() string {
	if modelStatePathOverride != "" {
		return modelStatePathOverride
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".mscli", "model.json")
}

func loadProviderAuthState() (*providerAuthState, error) {
	data, err := os.ReadFile(authStatePath())
	if err != nil {
		if os.IsNotExist(err) {
			return emptyProviderAuthState(), nil
		}
		return nil, err
	}

	var state providerAuthState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, err
	}
	state.normalize()
	return &state, nil
}

func saveProviderAuthState(state *providerAuthState) error {
	if state == nil {
		state = emptyProviderAuthState()
	}
	state.normalize()

	path := authStatePath()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o600)
}

func loadModelSelectionState() (*modelSelectionState, error) {
	data, err := os.ReadFile(modelStatePath())
	if err != nil {
		if os.IsNotExist(err) {
			return emptyModelSelectionState(), nil
		}
		return nil, err
	}

	var state modelSelectionState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, err
	}
	state.normalize()
	return &state, nil
}

func saveModelSelectionState(state *modelSelectionState) error {
	if state == nil {
		state = emptyModelSelectionState()
	}
	state.normalize()

	path := modelStatePath()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o600)
}

func emptyProviderAuthState() *providerAuthState {
	return &providerAuthState{
		Providers: map[string]providerAuthEntry{},
	}
}

func emptyModelSelectionState() *modelSelectionState {
	return &modelSelectionState{
		Recents:   []modelRef{},
		Favorites: []modelRef{},
	}
}

func (s *providerAuthState) normalize() {
	if s == nil {
		return
	}
	if s.Providers == nil {
		s.Providers = map[string]providerAuthEntry{}
		return
	}

	normalized := make(map[string]providerAuthEntry, len(s.Providers))
	for id, entry := range s.Providers {
		entry.ProviderID = normalizedProviderID(entry.ProviderID)
		key := normalizedProviderID(id)
		if key == "" {
			key = entry.ProviderID
		}
		if key == "" {
			continue
		}
		if entry.ProviderID == "" {
			entry.ProviderID = key
		}
		entry.APIKey = strings.TrimSpace(entry.APIKey)
		normalized[key] = entry
	}
	s.Providers = normalized
}

func (s *modelSelectionState) normalize() {
	if s == nil {
		return
	}
	if s.Active != nil {
		normalized := s.Active.normalized()
		if normalized.empty() {
			s.Active = nil
		} else {
			s.Active = &normalized
		}
	}
	s.Recents = normalizeModelRefs(s.Recents)
	s.Favorites = normalizeModelRefs(s.Favorites)
}

func normalizeModelRefs(refs []modelRef) []modelRef {
	if len(refs) == 0 {
		return []modelRef{}
	}

	seen := make(map[string]struct{}, len(refs))
	out := make([]modelRef, 0, len(refs))
	for _, ref := range refs {
		normalized := ref.normalized()
		if normalized.empty() {
			continue
		}
		key := normalized.key()
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, normalized)
	}
	return out
}

func (r modelRef) normalized() modelRef {
	return modelRef{
		ProviderID: normalizedProviderID(r.ProviderID),
		ModelID:    strings.TrimSpace(r.ModelID),
	}
}

func (r modelRef) key() string {
	normalized := r.normalized()
	return normalized.ProviderID + ":" + normalized.ModelID
}

func (r modelRef) empty() bool {
	normalized := r.normalized()
	return normalized.ProviderID == "" || normalized.ModelID == ""
}

func normalizedProviderID(v string) string {
	return strings.ToLower(strings.TrimSpace(v))
}
