package app

import (
	"testing"
)

func TestBuildModelPickerOptionsIncludesRecentGroup(t *testing.T) {
	catalog := &providerCatalog{
		Providers: []providerCatalogEntry{
			{
				ID:       "openai",
				Label:    "OpenAI",
				Protocol: "openai-chat",
				Models: []modelCatalogEntry{
					{ProviderID: "openai", ID: "gpt-4.1", Label: "GPT-4.1"},
					{ProviderID: "openai", ID: "gpt-4.1-mini", Label: "GPT-4.1 Mini"},
					{ProviderID: "openai", ID: "gpt-4o", Label: "GPT-4o"},
					{ProviderID: "openai", ID: "gpt-4o-mini", Label: "GPT-4o Mini"},
					{ProviderID: "openai", ID: "o3", Label: "o3"},
					{ProviderID: "openai", ID: "o4-mini", Label: "o4-mini"},
				},
			},
		},
	}
	authState := &providerAuthState{
		Providers: map[string]providerAuthEntry{
			"openai": {ProviderID: "openai", APIKey: "sk-test"},
		},
	}
	modelState := &modelSelectionState{
		Recents: []modelRef{
			{ProviderID: "missing", ModelID: "gone"},
			{ProviderID: "openai", ModelID: "gpt-4.1"},
			{ProviderID: "openai", ModelID: "gpt-4.1-mini"},
			{ProviderID: "openai", ModelID: "gpt-4o"},
			{ProviderID: "openai", ModelID: "gpt-4o-mini"},
			{ProviderID: "openai", ModelID: "o3"},
			{ProviderID: "openai", ModelID: "o4-mini"},
		},
	}

	options := buildModelPickerOptions(catalog, authState, modelState, false, nil)

	wantIDs := []string{
		"__header__Recent",
		"openai:gpt-4.1",
		"openai:gpt-4.1-mini",
		"openai:gpt-4o",
		"openai:gpt-4o-mini",
		"openai:o3",
		"__separator__provider:openai",
		"__provider__openai",
	}
	if len(options) < len(wantIDs) {
		t.Fatalf("len(options) = %d, want at least %d", len(options), len(wantIDs))
	}
	for i, want := range wantIDs {
		if got := options[i].ID; got != want {
			t.Fatalf("options[%d].ID = %q, want %q", i, got, want)
		}
	}
	if got, want := options[1].Label, "GPT-4.1"; got != want {
		t.Fatalf("options[1].Label = %q, want %q", got, want)
	}
	if got, want := options[1].Desc, "· OpenAI"; got != want {
		t.Fatalf("options[1].Desc = %q, want %q", got, want)
	}
	if got := options[7].Desc; got != "" {
		t.Fatalf("options[7].Desc = %q, want empty provider-group entry desc", got)
	}
	if !options[7].ProviderRow {
		t.Fatal("expected provider group entry to be selectable provider row")
	}
	if got, want := options[7].DeleteProviderID, "openai"; got != want {
		t.Fatalf("options[7].DeleteProviderID = %q, want %q", got, want)
	}
}

func TestBuildModelPickerOptionsDisablesFreeProviderRowInModelsList(t *testing.T) {
	catalog := &providerCatalog{
		Providers: []providerCatalogEntry{
			{
				ID:       mindsporeCLIFreeProviderID,
				Label:    "MindSpore CLI Free",
				Protocol: "mindspore-cli-free",
				Models: []modelCatalogEntry{
					{ProviderID: mindsporeCLIFreeProviderID, ID: "kimi-k2.5", Label: "Kimi K2.5"},
				},
			},
		},
	}

	options := buildModelPickerOptions(catalog, emptyProviderAuthState(), &modelSelectionState{}, true, nil)
	if len(options) < 2 {
		t.Fatalf("len(options) = %d, want at least 2", len(options))
	}
	if got, want := options[0].ID, "__provider__"+mindsporeCLIFreeProviderID; got != want {
		t.Fatalf("options[0].ID = %q, want %q", got, want)
	}
	if !options[0].ProviderRow {
		t.Fatal("expected free provider row marker")
	}
	if !options[0].Disabled {
		t.Fatal("expected free provider row to be non-selectable")
	}
	if got := options[0].DeleteProviderID; got != "" {
		t.Fatalf("options[0].DeleteProviderID = %q, want empty", got)
	}
}

func TestPartitionConnectProvidersUsesConfiguredPopularOrder(t *testing.T) {
	providers := []providerCatalogEntry{
		{ID: "openrouter", Label: "OpenRouter"},
		{ID: "deepseek", Label: "DeepSeek"},
		{ID: "openai", Label: "OpenAI"},
		{ID: "kimi-for-coding", Label: "Kimi for Coding"},
		{ID: "anthropic", Label: "Anthropic"},
	}

	popular, other := partitionConnectProviders(providers, false)

	wantPopularIDs := []string{"anthropic", "openai", "kimi-for-coding", "deepseek"}
	if len(popular) != len(wantPopularIDs) {
		t.Fatalf("len(popular) = %d, want %d", len(popular), len(wantPopularIDs))
	}
	for i, want := range wantPopularIDs {
		if got := popular[i].ID; got != want {
			t.Fatalf("popular[%d].ID = %q, want %q", i, got, want)
		}
	}
	if len(other) != 1 || other[0].ID != "openrouter" {
		t.Fatalf("other = %#v, want only openrouter", other)
	}
}
