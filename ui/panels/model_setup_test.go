package panels

import (
	"fmt"
	"regexp"
	"strings"
	"testing"

	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

var selectionPopupANSIPattern = regexp.MustCompile(`\x1b\[[0-9;]*m`)

func TestRenderSetupPopupModeSelect(t *testing.T) {
	popup := &model.SetupPopup{
		Screen:       model.SetupScreenModeSelect,
		ModeSelected: 0,
		CanEscape:    true,
	}
	result := RenderSetupPopup(popup)
	if !strings.Contains(result, "mscli-provided") {
		t.Error("expected 'mscli-provided' in output")
	}
	if !strings.Contains(result, "your own model") {
		t.Error("expected 'your own model' in output")
	}
}

func TestRenderSetupPopupPresetPicker(t *testing.T) {
	popup := &model.SetupPopup{
		Screen: model.SetupScreenPresetPicker,
		PresetOptions: []model.SelectionOption{
			{ID: "kimi-k2.5-free", Label: "kimi-k2.5 [free]"},
			{ID: "deepseek-v3", Label: "deepseek-v3"},
			{ID: "glm-4.7", Label: "glm-4.7 (coming soon)", Disabled: true},
		},
		PresetSelected: 0,
		CanEscape:      true,
	}
	result := RenderSetupPopup(popup)
	if !strings.Contains(result, "kimi-k2.5 [free]") {
		t.Error("expected active preset label in output")
	}
	if !strings.Contains(result, "deepseek-v3") {
		t.Error("expected deepseek preset in output")
	}
	if !strings.Contains(result, "glm-4.7 (coming soon)") {
		t.Error("expected coming-soon preset in output")
	}
}

func TestRenderSetupPopupTokenInput(t *testing.T) {
	popup := &model.SetupPopup{
		Screen:     model.SetupScreenTokenInput,
		TokenValue: "sk-abc",
		CanEscape:  true,
		SelectedPreset: model.SelectionOption{
			ID:    "kimi-k2.5-free",
			Label: "kimi-k2.5 [free]",
		},
	}
	result := RenderSetupPopup(popup)
	if !strings.Contains(result, "Token") {
		t.Error("expected 'Token' label in output")
	}
}

func TestRenderSetupPopupEnvInfo(t *testing.T) {
	popup := &model.SetupPopup{
		Screen:    model.SetupScreenEnvInfo,
		CanEscape: true,
	}
	result := RenderSetupPopup(popup)
	if !strings.Contains(result, "MSCLI_PROVIDER") {
		t.Error("expected env var example in output")
	}
	if !strings.Contains(result, "MSCLI_API_KEY") {
		t.Error("expected MSCLI_API_KEY in output")
	}
}

func TestRenderSetupPopupPresetPickerUsesWindowedOptions(t *testing.T) {
	options := make([]model.SelectionOption, 0, 20)
	for i := 0; i < 20; i++ {
		options = append(options, model.SelectionOption{
			ID:    "provider-" + strings.Repeat("x", i%3+1),
			Label: fmt.Sprintf("provider option item-%02d", i),
		})
	}
	popup := &model.SetupPopup{
		Screen:         model.SetupScreenPresetPicker,
		PresetOptions:  options,
		PresetSelected: 15,
		CanEscape:      true,
	}
	result := RenderSetupPopup(popup)
	if !strings.Contains(result, "provider option item-15") {
		t.Fatalf("expected selected window item in output, got:\n%s", result)
	}
	if strings.Contains(result, "provider option item-00") {
		t.Fatalf("expected earliest option to be clipped from output, got:\n%s", result)
	}
}

func TestRenderSelectionPopupUsesWindowedOptions(t *testing.T) {
	options := make([]model.SelectionOption, 0, 20)
	for i := 0; i < 20; i++ {
		options = append(options, model.SelectionOption{
			ID:    "model-id",
			Label: fmt.Sprintf("model option item-%02d", i),
		})
	}
	result := RenderSelectionPopup(&model.SelectionPopup{
		Title:    "Select model",
		Options:  options,
		Selected: 14,
	})
	if !strings.Contains(result, "model option item-14") {
		t.Fatalf("expected selected window item in output, got:\n%s", result)
	}
	if strings.Contains(result, "model option item-00") {
		t.Fatalf("expected earliest option to be clipped from output, got:\n%s", result)
	}
}

func TestRenderSelectionPopupIncludesConnectProviderShortcut(t *testing.T) {
	result := RenderSelectionPopup(&model.SelectionPopup{
		Title:    "Select model",
		ActionID: "model_picker",
		Options: []model.SelectionOption{
			{ID: "__header__Recent", Label: "Recent", Header: true, Disabled: true},
			{ID: "mindspore-cli-free:kimi-k2.5", Label: "Kimi K2.5"},
		},
		Selected: 1,
	})
	if !strings.Contains(result, "Connect Provider ctrl+a") {
		t.Fatalf("expected connect provider shortcut in output, got:\n%s", result)
	}
}

func TestRenderSelectionPopupModelPickerFooterOnlyShowsConnectShortcut(t *testing.T) {
	result := RenderSelectionPopup(&model.SelectionPopup{
		Title:    "Select model",
		ActionID: "model_picker",
		Options: []model.SelectionOption{
			{ID: "__header__Recent", Label: "Recent", Header: true, Disabled: true},
			{ID: "mindspore-cli-free:kimi-k2.5", Label: "Kimi K2.5"},
		},
		Selected: 1,
	})
	if strings.Contains(result, "↑/↓ select · enter confirm · esc cancel") {
		t.Fatalf("expected model picker footer to omit nav hint, got:\n%s", result)
	}
	if strings.Contains(result, "↑/↓ scroll · enter confirm · esc cancel") {
		t.Fatalf("expected model picker footer to omit scroll hint, got:\n%s", result)
	}
	if !strings.Contains(result, "Connect Provider ctrl+a") {
		t.Fatalf("expected model picker footer to keep connect shortcut, got:\n%s", result)
	}
}

func TestRenderSelectionPopupModelPickerShowsProviderAfterRecentModel(t *testing.T) {
	result := RenderSelectionPopup(&model.SelectionPopup{
		Title:    "Select model",
		ActionID: "model_picker",
		Options: []model.SelectionOption{
			{ID: "__header__Recent", Label: "Recent", Header: true, Disabled: true},
			{ID: "openai:gpt-4.1", Label: "GPT-4.1", Desc: "· OpenAI"},
		},
		Selected: 1,
	})

	plain := selectionPopupANSIPattern.ReplaceAllString(result, "")
	if !strings.Contains(plain, "GPT-4.1 · OpenAI") {
		t.Fatalf("expected recent model to render provider suffix, got:\n%s", plain)
	}
}

func TestRenderModelBrowserPopupFocusProviderShowsOnlyProviderCardContent(t *testing.T) {
	result := RenderModelBrowserPopup(&model.ModelBrowserPopup{
		Providers: model.SelectionPopup{
			Title: "Providers",
			Options: []model.SelectionOption{
				{ID: "__header__detected", Label: "Import", Header: true, Disabled: true},
				{
					ID:    "kimi-for-coding",
					Label: "Kimi For Coding",
				},
				{ID: "__detail__kimi-for-coding__source", Label: "from Claude Code environment detected:", Disabled: true, DetailRow: true},
				{ID: "__detail__kimi-for-coding__base_url", Label: "- ANTHROPIC_BASE_URL=https://api.kimi.com/coding/", Disabled: true, DetailRow: true},
				{ID: "__detail__kimi-for-coding__api_key", Label: "- ANTHROPIC_API_KEY=sk-kimi-xxxx****xxxx", Disabled: true, DetailRow: true},
				{ID: "openrouter", Label: "OpenRouter", RequiresInput: true},
			},
			Selected: 1,
		},
		Models: model.SelectionPopup{
			Title: "Models",
			Options: []model.SelectionOption{
				{ID: "openrouter:openai/gpt-4o-mini", Label: "GPT-4o mini"},
			},
			Selected: 0,
		},
		Focus:            model.ModelBrowserFocusProvider,
		ProvidersVisible: true,
	})

	plain := selectionPopupANSIPattern.ReplaceAllString(result, "")
	if !strings.Contains(plain, "Select Providers") {
		t.Fatalf("expected provider header, got:\n%s", plain)
	}
	if !strings.Contains(plain, "Import") {
		t.Fatalf("expected import section header in provider list, got:\n%s", plain)
	}
	if !strings.Contains(plain, "Kimi For Coding") {
		t.Fatalf("expected import candidate in provider list, got:\n%s", plain)
	}
	if !strings.Contains(plain, "from Claude Code environment detected:") {
		t.Fatalf("expected muted import hint in provider list, got:\n%s", plain)
	}
	if !strings.Contains(plain, "ANTHROPIC_BASE_URL=https://api.kimi.com/coding/") {
		t.Fatalf("expected base url hint in provider list, got:\n%s", plain)
	}
	if !strings.Contains(plain, "Select Models") {
		t.Fatalf("expected unified header to include models label, got:\n%s", plain)
	}
	if !strings.Contains(plain, "→ Select Models") {
		t.Fatalf("expected provider switch hint in header, got:\n%s", plain)
	}
	if !strings.Contains(plain, "↑/↓ select") {
		t.Fatalf("expected navigation hint in footer, got:\n%s", plain)
	}
	if strings.Contains(plain, "GPT-4o mini") {
		t.Fatalf("expected rear models card to hide content, got:\n%s", plain)
	}
	if strings.Contains(result, "───╮") || strings.Contains(result, "───╯") {
		t.Fatalf("expected provider focus to hide rear card edge fragments, got:\n%s", result)
	}
	lines := strings.Split(plain, "\n")
	if got := strings.TrimSpace(lines[len(lines)-1]); !strings.Contains(got, "↑/↓ select · enter choose · esc") {
		t.Fatalf("expected footer on final line, got last line %q in:\n%s", got, plain)
	}
}

func TestRenderModelBrowserPopupFocusModelsShowsOnlyModelCardContent(t *testing.T) {
	result := RenderModelBrowserPopup(&model.ModelBrowserPopup{
		Providers: model.SelectionPopup{
			Title: "Providers",
			Options: []model.SelectionOption{
				{ID: "openrouter", Label: "OpenRouter", RequiresInput: true},
			},
			Selected: 0,
		},
		Models: model.SelectionPopup{
			Title: "Models",
			Options: []model.SelectionOption{
				{ID: "__provider__openrouter", Label: "OpenRouter", ProviderRow: true, DeleteProviderID: "openrouter"},
				{ID: "openrouter:openai/gpt-4o-mini", Label: "GPT-4o mini"},
			},
			Selected: 0,
		},
		Focus:            model.ModelBrowserFocusModel,
		ProvidersVisible: false,
	})

	plain := selectionPopupANSIPattern.ReplaceAllString(result, "")
	if !strings.Contains(plain, "Select Providers") {
		t.Fatalf("expected unified header to include providers label, got:\n%s", plain)
	}
	if !strings.Contains(plain, "Select Models") {
		t.Fatalf("expected model header, got:\n%s", plain)
	}
	if !strings.Contains(plain, "Select Providers ←") {
		t.Fatalf("expected model switch hint in header, got:\n%s", plain)
	}
	if !strings.Contains(plain, "OpenRouter") {
		t.Fatalf("expected selectable provider row in model card, got:\n%s", plain)
	}
	if !strings.Contains(plain, "double-press d to delete") {
		t.Fatalf("expected provider delete hint, got:\n%s", plain)
	}
	if !strings.Contains(plain, "↑/↓ select") {
		t.Fatalf("expected navigation hint in footer, got:\n%s", plain)
	}
	if strings.Contains(result, "╭───") || strings.Contains(result, "╰───") {
		t.Fatalf("expected model focus to hide rear card edge fragments, got:\n%s", result)
	}
	lines := strings.Split(plain, "\n")
	if got := strings.TrimSpace(lines[len(lines)-1]); !strings.Contains(got, "↑/↓ select · enter choose · esc") {
		t.Fatalf("expected footer on final line, got last line %q in:\n%s", got, plain)
	}
}

func TestRenderModelBrowserPopupProviderInputKeepsCardChromeAndHidesSearch(t *testing.T) {
	result := RenderModelBrowserPopup(&model.ModelBrowserPopup{
		Providers: model.SelectionPopup{
			Title: "Providers",
			Options: []model.SelectionOption{
				{ID: "openrouter", Label: "OpenRouter", RequiresInput: true},
			},
			Selected: 0,
		},
		Models: model.SelectionPopup{
			Title: "Models",
			Options: []model.SelectionOption{
				{ID: "openrouter:openai/gpt-4o-mini", Label: "GPT-4o mini"},
			},
			Selected: 0,
		},
		Focus:            model.ModelBrowserFocusProvider,
		ProvidersVisible: true,
		ProviderInput: &model.ModelBrowserProviderInput{
			Option: model.SelectionOption{ID: "openrouter", Label: "OpenRouter", RequiresInput: true},
			Label:  "API key",
			Value:  "sk-demo",
		},
	})

	plain := selectionPopupANSIPattern.ReplaceAllString(result, "")
	if !strings.Contains(plain, "Select Providers") {
		t.Fatalf("expected provider header, got:\n%s", plain)
	}
	if !strings.Contains(plain, "Select Models") {
		t.Fatalf("expected unified header to include models label, got:\n%s", plain)
	}
	if !strings.Contains(plain, "API key") {
		t.Fatalf("expected input label, got:\n%s", plain)
	}
	if !strings.Contains(plain, "→ Select Models") {
		t.Fatalf("expected switch hint retained in input mode, got:\n%s", plain)
	}
	if !strings.Contains(plain, "↑/↓ select") {
		t.Fatalf("expected navigation hint retained in input mode, got:\n%s", plain)
	}
	if strings.Contains(plain, "Search") {
		t.Fatalf("expected search hidden in input mode, got:\n%s", plain)
	}
	lines := strings.Split(plain, "\n")
	if got := strings.TrimSpace(lines[len(lines)-1]); !strings.Contains(got, "↑/↓ select · enter choose · esc") {
		t.Fatalf("expected footer on final line, got last line %q in:\n%s", got, plain)
	}
}
