package app

import "testing"

func TestDetectProviderImportSuggestionsMatchesClaudeCodeKimiConfig(t *testing.T) {
	t.Setenv("ANTHROPIC_BASE_URL", "https://api.kimi.com/coding/")
	t.Setenv("ANTHROPIC_API_KEY", "sk-kimi-test")

	catalog := &providerCatalog{
		Providers: []providerCatalogEntry{
			{
				ID:       "kimi-for-coding",
				Label:    "Kimi For Coding",
				BaseURL:  "https://api.kimi.com/coding/v1",
				Protocol: "anthropic-messages",
			},
		},
	}

	suggestions := detectProviderImportSuggestions(catalog, emptyProviderAuthState())
	if got, want := len(suggestions), 1; got != want {
		t.Fatalf("len(suggestions) = %d, want %d", got, want)
	}
	suggestion := suggestions[0]
	if got, want := suggestion.ProviderID, "kimi-for-coding"; got != want {
		t.Fatalf("suggestion.ProviderID = %q, want %q", got, want)
	}
	if got, want := suggestion.Protocol, "anthropic-messages"; got != want {
		t.Fatalf("suggestion.Protocol = %q, want %q", got, want)
	}
	if got, want := suggestion.APIKeyEnvVar, "ANTHROPIC_API_KEY"; got != want {
		t.Fatalf("suggestion.APIKeyEnvVar = %q, want %q", got, want)
	}
	if got, want := suggestion.BaseURLEnvVar, "ANTHROPIC_BASE_URL"; got != want {
		t.Fatalf("suggestion.BaseURLEnvVar = %q, want %q", got, want)
	}
}

func TestDetectProviderImportSuggestionsMatchesClaudeCodeKimiConfigWithoutAPIKey(t *testing.T) {
	t.Setenv("ANTHROPIC_BASE_URL", "https://api.kimi.com/coding/")
	t.Setenv("ANTHROPIC_API_KEY", "")

	catalog := &providerCatalog{
		Providers: []providerCatalogEntry{
			{
				ID:       "kimi-for-coding",
				Label:    "Kimi For Coding",
				BaseURL:  "https://api.kimi.com/coding/v1",
				Protocol: "anthropic-messages",
			},
		},
	}

	suggestions := detectProviderImportSuggestions(catalog, emptyProviderAuthState())
	if got, want := len(suggestions), 1; got != want {
		t.Fatalf("len(suggestions) = %d, want %d", got, want)
	}
	if got, want := suggestions[0].APIKey, ""; got != want {
		t.Fatalf("suggestions[0].APIKey = %q, want empty", got)
	}
}
