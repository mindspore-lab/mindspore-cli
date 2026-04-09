package app

import (
	"path/filepath"
	"testing"
)

func TestAuthStateRoundTrip(t *testing.T) {
	dir := t.TempDir()
	origPath := authStatePathOverride
	authStatePathOverride = filepath.Join(dir, "auth.json")
	t.Cleanup(func() { authStatePathOverride = origPath })

	state := &providerAuthState{
		Providers: map[string]providerAuthEntry{
			"openrouter": {
				ProviderID: "openrouter",
				APIKey:     "sk-test",
			},
		},
	}
	if err := saveProviderAuthState(state); err != nil {
		t.Fatalf("saveProviderAuthState() error = %v", err)
	}

	loaded, err := loadProviderAuthState()
	if err != nil {
		t.Fatalf("loadProviderAuthState() error = %v", err)
	}
	entry, ok := loaded.Providers["openrouter"]
	if !ok {
		t.Fatalf("loaded auth providers = %#v, want openrouter entry", loaded.Providers)
	}
	if got, want := entry.APIKey, "sk-test"; got != want {
		t.Fatalf("entry.APIKey = %q, want %q", got, want)
	}
}

func TestLoadProviderAuthStateMissingReturnsEmpty(t *testing.T) {
	dir := t.TempDir()
	origPath := authStatePathOverride
	authStatePathOverride = filepath.Join(dir, "missing", "auth.json")
	t.Cleanup(func() { authStatePathOverride = origPath })

	loaded, err := loadProviderAuthState()
	if err != nil {
		t.Fatalf("loadProviderAuthState() error = %v", err)
	}
	if len(loaded.Providers) != 0 {
		t.Fatalf("len(loaded.Providers) = %d, want 0", len(loaded.Providers))
	}
}

func TestModelStateRoundTrip(t *testing.T) {
	dir := t.TempDir()
	origPath := modelStatePathOverride
	modelStatePathOverride = filepath.Join(dir, "model.json")
	t.Cleanup(func() { modelStatePathOverride = origPath })

	state := &modelSelectionState{
		Active: &modelRef{
			ProviderID: "mindspore-cli-free",
			ModelID:    "kimi-k2.5",
		},
		Recents: []modelRef{
			{ProviderID: "mindspore-cli-free", ModelID: "kimi-k2.5"},
			{ProviderID: "openrouter", ModelID: "anthropic/claude-sonnet-4.5"},
		},
		Favorites: []modelRef{
			{ProviderID: "openrouter", ModelID: "anthropic/claude-sonnet-4.5"},
		},
	}
	if err := saveModelSelectionState(state); err != nil {
		t.Fatalf("saveModelSelectionState() error = %v", err)
	}

	loaded, err := loadModelSelectionState()
	if err != nil {
		t.Fatalf("loadModelSelectionState() error = %v", err)
	}
	if loaded.Active == nil {
		t.Fatal("loaded.Active = nil, want value")
	}
	if got, want := loaded.Active.ProviderID, "mindspore-cli-free"; got != want {
		t.Fatalf("loaded.Active.ProviderID = %q, want %q", got, want)
	}
	if got, want := len(loaded.Recents), 2; got != want {
		t.Fatalf("len(loaded.Recents) = %d, want %d", got, want)
	}
	if got, want := len(loaded.Favorites), 1; got != want {
		t.Fatalf("len(loaded.Favorites) = %d, want %d", got, want)
	}
}

func TestLoadModelSelectionStateMissingReturnsEmpty(t *testing.T) {
	dir := t.TempDir()
	origPath := modelStatePathOverride
	modelStatePathOverride = filepath.Join(dir, "missing", "model.json")
	t.Cleanup(func() { modelStatePathOverride = origPath })

	loaded, err := loadModelSelectionState()
	if err != nil {
		t.Fatalf("loadModelSelectionState() error = %v", err)
	}
	if loaded.Active != nil {
		t.Fatalf("loaded.Active = %#v, want nil", loaded.Active)
	}
	if len(loaded.Recents) != 0 {
		t.Fatalf("len(loaded.Recents) = %d, want 0", len(loaded.Recents))
	}
	if len(loaded.Favorites) != 0 {
		t.Fatalf("len(loaded.Favorites) = %d, want 0", len(loaded.Favorites))
	}
}

func TestModelSelectionStateNormalizeDeduplicatesRecentAndFavorites(t *testing.T) {
	state := &modelSelectionState{
		Recents: []modelRef{
			{ProviderID: "openrouter", ModelID: "gpt-4o"},
			{ProviderID: "openrouter", ModelID: "gpt-4o"},
			{ProviderID: " mindspore-cli-free ", ModelID: " kimi-k2.5 "},
		},
		Favorites: []modelRef{
			{ProviderID: " mindspore-cli-free ", ModelID: " kimi-k2.5 "},
			{ProviderID: "mindspore-cli-free", ModelID: "kimi-k2.5"},
		},
	}

	state.normalize()

	if got, want := len(state.Recents), 2; got != want {
		t.Fatalf("len(state.Recents) = %d, want %d", got, want)
	}
	if got, want := state.Recents[1].ProviderID, "mindspore-cli-free"; got != want {
		t.Fatalf("state.Recents[1].ProviderID = %q, want %q", got, want)
	}
	if got, want := state.Recents[1].ModelID, "kimi-k2.5"; got != want {
		t.Fatalf("state.Recents[1].ModelID = %q, want %q", got, want)
	}
	if got, want := len(state.Favorites), 1; got != want {
		t.Fatalf("len(state.Favorites) = %d, want %d", got, want)
	}
}
