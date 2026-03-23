package slash

import "testing"

func TestDefaultRegistryIncludesQuit(t *testing.T) {
	r := NewRegistry()

	if _, ok := r.Get("/quit"); !ok {
		t.Fatal("expected /quit command to be registered")
	}
}

func TestSuggestionsIncludesQuitPrefix(t *testing.T) {
	r := NewRegistry()

	suggestions := r.Suggestions("/q")
	if !contains(suggestions, "/quit") {
		t.Fatalf("expected /quit in suggestions for /q, got %v", suggestions)
	}
}

func contains(items []string, target string) bool {
	for _, item := range items {
		if item == target {
			return true
		}
	}
	return false
}
