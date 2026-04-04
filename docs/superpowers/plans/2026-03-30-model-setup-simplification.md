# Model Setup Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the complex model/provider selection flow with a two-path startup: mscli-provided model (token input) vs your-own-model (env vars), with auto-detection that skips the popup when config already exists.

**Architecture:** New `config.json` file at `~/.mscli/config.json` stores model mode, preset ID, and model token. Startup detection in `Wire()` checks env vars first, then `config.json`, and only emits a setup popup event when neither is configured. The existing `SelectionPopup` is extended with a multi-step flow (mode selection -> preset picker -> token input / env info). `/model` command opens the same popup.

**Tech Stack:** Go, Bubble Tea TUI, existing `configs`, `llm`, `ui/model`, `ui/panels` packages.

---

### Task 1: Add `config.json` persistence layer

**Files:**
- Create: `internal/app/appconfig.go`
- Test: `internal/app/appconfig_test.go`

This task adds load/save functions for `~/.mscli/config.json` which stores the model mode and mscli-provided token. Separate from `credentials.json` (issue server auth) and `configs/` (YAML defaults + env overrides).

- [ ] **Step 1: Write the failing test for `loadAppConfig` and `saveAppConfig`**

```go
package app

import (
	"os"
	"path/filepath"
	"testing"
)

func TestAppConfigRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	origPath := appConfigPathOverride
	appConfigPathOverride = path
	t.Cleanup(func() { appConfigPathOverride = origPath })

	cfg := &appConfig{
		ModelMode:     "mscli-provided",
		ModelPresetID: "kimi-k2.5-free",
		ModelToken:    "sk-test-token-123",
	}
	if err := saveAppConfig(cfg); err != nil {
		t.Fatalf("saveAppConfig: %v", err)
	}

	loaded, err := loadAppConfig()
	if err != nil {
		t.Fatalf("loadAppConfig: %v", err)
	}
	if loaded.ModelMode != cfg.ModelMode {
		t.Errorf("ModelMode = %q, want %q", loaded.ModelMode, cfg.ModelMode)
	}
	if loaded.ModelPresetID != cfg.ModelPresetID {
		t.Errorf("ModelPresetID = %q, want %q", loaded.ModelPresetID, cfg.ModelPresetID)
	}
	if loaded.ModelToken != cfg.ModelToken {
		t.Errorf("ModelToken = %q, want %q", loaded.ModelToken, cfg.ModelToken)
	}
}

func TestLoadAppConfigMissing(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nonexistent", "config.json")
	origPath := appConfigPathOverride
	appConfigPathOverride = path
	t.Cleanup(func() { appConfigPathOverride = origPath })

	cfg, err := loadAppConfig()
	if err != nil {
		t.Fatalf("expected nil error for missing file, got: %v", err)
	}
	if cfg.ModelMode != "" {
		t.Errorf("expected empty ModelMode, got %q", cfg.ModelMode)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/app/ -run TestAppConfig -v`
Expected: FAIL — `appConfig` type and functions not defined.

- [ ] **Step 3: Write minimal implementation**

```go
package app

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// appConfig holds persistent local settings stored in ~/.mscli/config.json.
// Separate from credentials.json (issue server auth) and configs/ (YAML + env).
type appConfig struct {
	ModelMode     string `json:"model_mode,omitempty"`      // "mscli-provided" or "own" or ""
	ModelPresetID string `json:"model_preset_id,omitempty"` // e.g. "kimi-k2.5-free"
	ModelToken    string `json:"model_token,omitempty"`     // API token for mscli-provided models
}

// appConfigPathOverride allows tests to redirect the config path.
var appConfigPathOverride string

func appConfigPath() string {
	if appConfigPathOverride != "" {
		return appConfigPathOverride
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".mscli", "config.json")
}

func loadAppConfig() (*appConfig, error) {
	data, err := os.ReadFile(appConfigPath())
	if err != nil {
		if os.IsNotExist(err) {
			return &appConfig{}, nil
		}
		return nil, err
	}
	var cfg appConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}

func saveAppConfig(cfg *appConfig) error {
	dir := filepath.Dir(appConfigPath())
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(appConfigPath(), data, 0o600)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./internal/app/ -run TestAppConfig -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add internal/app/appconfig.go internal/app/appconfig_test.go
git commit -m "feat: add config.json persistence for model setup preferences"
```

---

### Task 2: Add `ComingSoon` flag and new presets to model presets

**Files:**
- Modify: `internal/app/model_presets.go`
- Test: `internal/app/model_presets_test.go` (create)

- [ ] **Step 1: Write the failing test**

```go
package app

import "testing"

func TestListBuiltinModelPresetsIncludesComingSoon(t *testing.T) {
	presets := listBuiltinModelPresets()
	if len(presets) < 4 {
		t.Fatalf("expected at least 4 presets (1 active + 3 coming soon), got %d", len(presets))
	}

	active := 0
	comingSoon := 0
	for _, p := range presets {
		if p.ComingSoon {
			comingSoon++
		} else {
			active++
		}
	}
	if active < 1 {
		t.Errorf("expected at least 1 active preset, got %d", active)
	}
	if comingSoon < 3 {
		t.Errorf("expected at least 3 coming-soon presets, got %d", comingSoon)
	}
}

func TestResolveBuiltinModelPresetSkipsComingSoon(t *testing.T) {
	// "glm-4.7" is coming soon and should NOT resolve as a usable preset.
	if _, ok := resolveBuiltinModelPreset("glm-4.7"); ok {
		t.Error("expected coming-soon preset to not resolve")
	}
	// "kimi-k2.5-free" is active and should resolve.
	if _, ok := resolveBuiltinModelPreset("kimi-k2.5-free"); !ok {
		t.Error("expected active preset to resolve")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/app/ -run TestListBuiltinModelPresets -v && go test ./internal/app/ -run TestResolveBuiltinModelPresetSkips -v`
Expected: FAIL — `ComingSoon` field not defined, fewer than 4 presets.

- [ ] **Step 3: Write implementation**

In `internal/app/model_presets.go`, add the `ComingSoon` field to `builtinModelPreset` and add three placeholder presets. Update `resolveBuiltinModelPreset` to skip coming-soon entries.

```go
// In builtinModelPreset struct, add:
ComingSoon bool

// Replace builtinModelPresets slice:
var builtinModelPresets = []builtinModelPreset{
	{
		ID:       "kimi-k2.5-free",
		Label:    "kimi-k2.5 [free]",
		Provider: "anthropic",
		BaseURL:  "https://api.kimi.com/coding/",
		Model:    "kimi-k2.5",
		Aliases:  []string{"kimi-k2.5", "kimi-k2.5 [free]"},
		Credential: modelCredentialSpec{
			Strategy: credentialStrategyMSCODEServer,
			Path:     "/model-presets/kimi-k2.5-free/credential",
		},
	},
	{
		ID:         "glm-4.7",
		Label:      "glm-4.7 (coming soon)",
		Model:      "glm-4.7",
		ComingSoon: true,
	},
	{
		ID:         "deepseek-v4",
		Label:      "deepseek-v4 (coming soon)",
		Model:      "deepseek-v4",
		ComingSoon: true,
	},
	{
		ID:         "minimax-m2.7",
		Label:      "minimax-m2.7 (coming soon)",
		Model:      "minimax-m2.7",
		ComingSoon: true,
	},
}

// Update resolveBuiltinModelPreset to skip ComingSoon:
func resolveBuiltinModelPreset(input string) (builtinModelPreset, bool) {
	needle := strings.ToLower(strings.TrimSpace(input))
	if needle == "" {
		return builtinModelPreset{}, false
	}

	for _, preset := range builtinModelPresets {
		if preset.ComingSoon {
			continue
		}
		if strings.EqualFold(needle, preset.ID) || strings.EqualFold(needle, preset.Label) {
			return preset, true
		}
		for _, alias := range preset.Aliases {
			if strings.EqualFold(needle, alias) {
				return preset, true
			}
		}
	}
	return builtinModelPreset{}, false
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./internal/app/ -run "TestListBuiltinModelPresets|TestResolveBuiltinModelPresetSkips" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add internal/app/model_presets.go internal/app/model_presets_test.go
git commit -m "feat: add ComingSoon flag and placeholder presets (glm-4.7, deepseek-v4, minimax-m2.7)"
```

---

### Task 3: Add new event types and extend `SelectionPopup` for multi-step flow

**Files:**
- Modify: `ui/model/model.go` (add `ModelSetupOpen` event type)
- Modify: `ui/model/train.go` (add `Disabled` field to `SelectionOption`, add `SetupPopup` struct)

- [ ] **Step 1: Write failing test for disabled option skipping in popup navigation**

Create `ui/model/popup_test.go`:

```go
package model

import "testing"

func TestSetupPopupNextSelectable(t *testing.T) {
	popup := &SetupPopup{
		Screen: SetupScreenPresetPicker,
		PresetOptions: []SelectionOption{
			{ID: "kimi-k2.5-free", Label: "kimi-k2.5 [free]"},
			{ID: "glm-4.7", Label: "glm-4.7 (coming soon)", Disabled: true},
			{ID: "deepseek-v4", Label: "deepseek-v4 (coming soon)", Disabled: true},
			{ID: "minimax-m2.7", Label: "minimax-m2.7 (coming soon)", Disabled: true},
		},
		PresetSelected: 0,
	}

	// Moving down from index 0 should wrap back to 0 (all others disabled)
	popup.MovePresetSelection(1)
	if popup.PresetSelected != 0 {
		t.Errorf("expected selection to stay at 0 (others disabled), got %d", popup.PresetSelected)
	}
}

func TestSetupPopupMoveMode(t *testing.T) {
	popup := &SetupPopup{
		Screen:       SetupScreenModeSelect,
		ModeSelected: 0,
	}
	popup.MoveModeSelection(1)
	if popup.ModeSelected != 1 {
		t.Errorf("expected 1, got %d", popup.ModeSelected)
	}
	popup.MoveModeSelection(1)
	if popup.ModeSelected != 0 {
		t.Errorf("expected wrap to 0, got %d", popup.ModeSelected)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./ui/model/ -run "TestSetupPopup" -v`
Expected: FAIL — `SetupPopup`, `SetupScreenPresetPicker`, etc. not defined.

- [ ] **Step 3: Write implementation**

Add to `ui/model/train.go` (after the existing `SelectionPopup`):

```go
// SetupScreen identifies which screen of the model setup popup is shown.
type SetupScreen int

const (
	SetupScreenModeSelect   SetupScreen = iota // "mscli-provided" vs "your own model"
	SetupScreenPresetPicker                     // pick from preset list
	SetupScreenTokenInput                       // enter token for selected preset
	SetupScreenEnvInfo                          // show env var examples
)

// SetupPopup holds the full state of the multi-step model setup popup.
type SetupPopup struct {
	Screen         SetupScreen
	ModeSelected   int    // 0 = mscli-provided, 1 = your own model
	PresetOptions  []SelectionOption
	PresetSelected int
	SelectedPreset SelectionOption // set when user picks a preset
	TokenValue     string
	TokenError     string // inline error message
	CurrentMode    string // "mscli-provided", "own", or "" — for (current) badge
	CurrentPreset  string // preset ID currently active — for (current) badge
	CanEscape      bool   // false on first boot (no config to fall back to)
}

// MoveModeSelection moves the mode cursor by delta, wrapping around 2 options.
func (p *SetupPopup) MoveModeSelection(delta int) {
	p.ModeSelected = (p.ModeSelected + delta + 2) % 2
}

// MovePresetSelection moves the preset cursor by delta, skipping disabled options.
func (p *SetupPopup) MovePresetSelection(delta int) {
	n := len(p.PresetOptions)
	if n == 0 {
		return
	}
	start := p.PresetSelected
	for i := 0; i < n; i++ {
		next := (p.PresetSelected + delta + n) % n
		p.PresetSelected = next
		if !p.PresetOptions[next].Disabled {
			return
		}
	}
	// All disabled — stay at start.
	p.PresetSelected = start
}
```

Add `Disabled` field to existing `SelectionOption`:

```go
type SelectionOption struct {
	ID       string
	Label    string
	Desc     string
	Disabled bool // grayed out, not selectable (e.g. coming soon)
}
```

Add event type in `ui/model/model.go`:

```go
ModelSetupOpen EventType = "ModelSetupOpen"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./ui/model/ -run "TestSetupPopup" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/model/model.go ui/model/train.go ui/model/popup_test.go
git commit -m "feat: add SetupPopup multi-step model and Disabled option support"
```

---

### Task 4: Render the multi-step setup popup

**Files:**
- Create: `ui/panels/model_setup.go`
- Test: `ui/panels/model_setup_test.go`

- [ ] **Step 1: Write failing test**

```go
package panels

import (
	"strings"
	"testing"

	"github.com/vigo999/mindspore-code/ui/model"
)

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
			{ID: "glm-4.7", Label: "glm-4.7 (coming soon)", Disabled: true},
		},
		PresetSelected: 0,
		CanEscape:      true,
	}
	result := RenderSetupPopup(popup)
	if !strings.Contains(result, "kimi-k2.5 [free]") {
		t.Error("expected active preset label in output")
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./ui/panels/ -run TestRenderSetupPopup -v`
Expected: FAIL — `RenderSetupPopup` not defined.

- [ ] **Step 3: Write implementation**

```go
package panels

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/mindspore-code/ui/model"
)

var (
	setupTitleStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("12")).Bold(true).Align(lipgloss.Center)
	setupNormalStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("7"))
	setupSelectedStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("14")).Bold(true)
	setupDisabledStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("8"))
	setupHintStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Italic(true)
	setupErrorStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("196"))
	setupLabelStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("252"))
	setupBadgeStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("244"))
	setupBorderStyle   = lipgloss.NewStyle().
				Border(lipgloss.RoundedBorder()).
				BorderForeground(lipgloss.Color("12")).
				Padding(0, 2)
)

// RenderSetupPopup renders the multi-step model setup popup.
func RenderSetupPopup(popup *model.SetupPopup) string {
	switch popup.Screen {
	case model.SetupScreenModeSelect:
		return renderModeSelect(popup)
	case model.SetupScreenPresetPicker:
		return renderPresetPicker(popup)
	case model.SetupScreenTokenInput:
		return renderTokenInput(popup)
	case model.SetupScreenEnvInfo:
		return renderEnvInfo(popup)
	default:
		return ""
	}
}

func renderModeSelect(popup *model.SetupPopup) string {
	modes := []struct {
		label string
		mode  string
	}{
		{"mscli-provided model", "mscli-provided"},
		{"your own model", "own"},
	}

	maxW := len("Model Setup")
	for _, m := range modes {
		if w := 2 + len(m.label) + 12; w > maxW { // room for (current) badge
			maxW = w
		}
	}

	var lines []string
	lines = append(lines, setupTitleStyle.Width(maxW).Render("Model Setup"))
	lines = append(lines, "")
	for i, m := range modes {
		marker := "  "
		style := setupNormalStyle
		if i == popup.ModeSelected {
			marker = "> "
			style = setupSelectedStyle
		}
		label := m.label
		if popup.CurrentMode == m.mode {
			label += setupBadgeStyle.Render("  (current)")
		}
		lines = append(lines, marker+style.Render(label))
	}
	lines = append(lines, "")
	hint := "↑/↓ select · enter confirm"
	if popup.CanEscape {
		hint += " · esc cancel"
	}
	lines = append(lines, setupHintStyle.Render(hint))

	return setupBorderStyle.Render(strings.Join(lines, "\n"))
}

func renderPresetPicker(popup *model.SetupPopup) string {
	maxW := len("mscli-provided")
	for _, opt := range popup.PresetOptions {
		if w := 2 + len(opt.Label) + 12; w > maxW {
			maxW = w
		}
	}

	var lines []string
	lines = append(lines, setupTitleStyle.Width(maxW).Render("mscli-provided"))
	lines = append(lines, "")
	for i, opt := range popup.PresetOptions {
		marker := "  "
		style := setupNormalStyle
		if opt.Disabled {
			style = setupDisabledStyle
		}
		if i == popup.PresetSelected {
			if !opt.Disabled {
				marker = "> "
				style = setupSelectedStyle
			}
		}
		label := opt.Label
		if opt.ID == popup.CurrentPreset {
			label += setupBadgeStyle.Render("  (current)")
		}
		lines = append(lines, marker+style.Render(label))
	}
	lines = append(lines, "")
	lines = append(lines, setupHintStyle.Render("↑/↓ select · enter · esc back"))

	return setupBorderStyle.Render(strings.Join(lines, "\n"))
}

func renderTokenInput(popup *model.SetupPopup) string {
	title := popup.SelectedPreset.Label
	if title == "" {
		title = "Enter Token"
	}

	var lines []string
	lines = append(lines, setupTitleStyle.Width(40).Render(title))
	lines = append(lines, "")
	lines = append(lines, setupLabelStyle.Render("Token: ")+maskToken(popup.TokenValue))
	if popup.TokenError != "" {
		lines = append(lines, "")
		lines = append(lines, setupErrorStyle.Render(popup.TokenError))
	}
	lines = append(lines, "")
	lines = append(lines, setupHintStyle.Render("enter apply · esc back"))

	return setupBorderStyle.Render(strings.Join(lines, "\n"))
}

func maskToken(token string) string {
	if len(token) <= 8 {
		return token + strings.Repeat("_", 20-len(token))
	}
	return token[:4] + strings.Repeat("·", len(token)-8) + token[len(token)-4:]
}

func renderEnvInfo(popup *model.SetupPopup) string {
	var lines []string
	lines = append(lines, setupTitleStyle.Width(50).Render("Your Own Model"))
	lines = append(lines, "")
	lines = append(lines, setupLabelStyle.Render("Set environment variables:"))
	lines = append(lines, "")
	lines = append(lines, setupNormalStyle.Render("  export MSCLI_PROVIDER=openai-completion"))
	lines = append(lines, setupNormalStyle.Render("  export MSCLI_BASE_URL=https://api.openai.com/v1"))
	lines = append(lines, setupNormalStyle.Render("  export MSCLI_API_KEY=sk-..."))
	lines = append(lines, setupNormalStyle.Render("  export MSCLI_MODEL=gpt-5.4"))
	lines = append(lines, "")
	lines = append(lines, setupHintStyle.Render("Then restart mscli."))
	lines = append(lines, "")
	lines = append(lines, setupHintStyle.Render("esc back"))

	return setupBorderStyle.Render(strings.Join(lines, "\n"))
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./ui/panels/ -run TestRenderSetupPopup -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/panels/model_setup.go ui/panels/model_setup_test.go
git commit -m "feat: render multi-step model setup popup (mode select, preset picker, token input, env info)"
```

---

### Task 5: Wire setup popup into TUI event handling

**Files:**
- Modify: `ui/app.go` (add `setupPopup *model.SetupPopup` field, handle `ModelSetupOpen`, keyboard navigation, token text input)

This is the core UI wiring. The setup popup replaces the model picker popup. The TUI handles keyboard events for all four screens.

- [ ] **Step 1: Write failing test**

Create `ui/app_model_setup_test.go`:

```go
package ui

import (
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/vigo999/mindspore-code/ui/model"
)

func TestSetupPopupModeSelectAndEnter(t *testing.T) {
	eventCh := make(chan model.Event, 64)
	userCh := make(chan string, 8)
	app := New(eventCh, userCh, "test", "/tmp", "", "", 128000)

	// Open setup popup
	app.setupPopup = &model.SetupPopup{
		Screen:       model.SetupScreenModeSelect,
		ModeSelected: 0,
		CanEscape:    true,
	}

	// Press down to select "your own model"
	app, _ = app.Update(tea.KeyMsg{Type: tea.KeyDown})
	if app.(App).setupPopup.ModeSelected != 1 {
		t.Errorf("expected mode 1, got %d", app.(App).setupPopup.ModeSelected)
	}

	// Press enter on "your own model" -> should go to env info screen
	app, _ = app.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if app.(App).setupPopup.Screen != model.SetupScreenEnvInfo {
		t.Errorf("expected env info screen, got %d", app.(App).setupPopup.Screen)
	}
}

func TestSetupPopupEscFromPresetPicker(t *testing.T) {
	eventCh := make(chan model.Event, 64)
	userCh := make(chan string, 8)
	app := New(eventCh, userCh, "test", "/tmp", "", "", 128000)

	app.setupPopup = &model.SetupPopup{
		Screen:    model.SetupScreenPresetPicker,
		CanEscape: true,
	}

	// Esc should go back to mode select
	app, _ = app.Update(tea.KeyMsg{Type: tea.KeyEscape})
	if app.(App).setupPopup.Screen != model.SetupScreenModeSelect {
		t.Errorf("expected mode select screen, got %d", app.(App).setupPopup.Screen)
	}
}
```

NOTE: The exact test structure may need adjusting based on how the `App` type is structured (it may be a struct, not interface). Adapt the assertions to match the actual `App` type. The key behavior to test is:
- Down arrow on mode select changes `ModeSelected`
- Enter on mode 0 goes to `SetupScreenPresetPicker`
- Enter on mode 1 goes to `SetupScreenEnvInfo`
- Esc from sub-screens goes back to mode select
- Esc from mode select closes popup (when `CanEscape` is true)
- Esc from mode select does nothing (when `CanEscape` is false)

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./ui/ -run TestSetupPopup -v`
Expected: FAIL — `setupPopup` field not defined on App.

- [ ] **Step 3: Write implementation**

In `ui/app.go`, add the `setupPopup` field to the `App` struct alongside the existing `modelPicker`:

```go
setupPopup  *model.SetupPopup
```

Add event handler for `model.ModelSetupOpen`:

```go
case model.ModelSetupOpen:
    a.setupPopup = msg.SetupPopup
    return a, nil
```

Add a new `SetupPopup` field to `model.Event`:

```go
// In ui/model/model.go Event struct:
SetupPopup *SetupPopup // non-nil for model setup popup events
```

Add keyboard handling block before the existing `modelPicker` block. The setup popup should be checked first since it replaces the model picker:

```go
if a.setupPopup != nil {
    switch a.setupPopup.Screen {
    case model.SetupScreenModeSelect:
        switch msg.String() {
        case "up", "left":
            a.setupPopup.MoveModeSelection(-1)
            return a, nil
        case "down", "right":
            a.setupPopup.MoveModeSelection(1)
            return a, nil
        case "enter":
            if a.setupPopup.ModeSelected == 0 {
                a.setupPopup.Screen = model.SetupScreenPresetPicker
            } else {
                a.setupPopup.Screen = model.SetupScreenEnvInfo
            }
            return a, nil
        case "esc":
            if a.setupPopup.CanEscape {
                a.setupPopup = nil
            }
            return a, nil
        }
    case model.SetupScreenPresetPicker:
        switch msg.String() {
        case "up", "left":
            a.setupPopup.MovePresetSelection(-1)
            return a, nil
        case "down", "right":
            a.setupPopup.MovePresetSelection(1)
            return a, nil
        case "enter":
            opt := a.setupPopup.PresetOptions[a.setupPopup.PresetSelected]
            if !opt.Disabled {
                a.setupPopup.SelectedPreset = opt
                a.setupPopup.Screen = model.SetupScreenTokenInput
                a.setupPopup.TokenValue = ""
                a.setupPopup.TokenError = ""
            }
            return a, nil
        case "esc":
            a.setupPopup.Screen = model.SetupScreenModeSelect
            return a, nil
        }
    case model.SetupScreenTokenInput:
        switch msg.String() {
        case "enter":
            if a.userCh != nil && strings.TrimSpace(a.setupPopup.TokenValue) != "" {
                // Send command to backend: __model_setup <presetID> <token>
                cmd := fmt.Sprintf("__model_setup %s %s",
                    a.setupPopup.SelectedPreset.ID,
                    strings.TrimSpace(a.setupPopup.TokenValue))
                select {
                case a.userCh <- cmd:
                default:
                }
            }
            return a, nil
        case "esc":
            a.setupPopup.Screen = model.SetupScreenPresetPicker
            return a, nil
        case "backspace":
            if len(a.setupPopup.TokenValue) > 0 {
                a.setupPopup.TokenValue = a.setupPopup.TokenValue[:len(a.setupPopup.TokenValue)-1]
            }
            return a, nil
        default:
            // Append printable characters to token value.
            if len(msg.String()) == 1 && msg.String()[0] >= 32 {
                a.setupPopup.TokenValue += msg.String()
            }
            return a, nil
        }
    case model.SetupScreenEnvInfo:
        if msg.String() == "esc" {
            a.setupPopup.Screen = model.SetupScreenModeSelect
            return a, nil
        }
        return a, nil
    }
    return a, nil
}
```

In the `View()` method, render the setup popup (overlay it like the existing model picker):

```go
if a.setupPopup != nil {
    popupStr := panels.RenderSetupPopup(a.setupPopup)
    return overlayPopup(fullView, popupStr, a.width, a.height)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./ui/ -run TestSetupPopup -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/app.go ui/model/model.go ui/app_model_setup_test.go
git commit -m "feat: wire multi-step setup popup into TUI with keyboard navigation"
```

---

### Task 6: Backend handler for setup popup token submission

**Files:**
- Modify: `internal/app/commands.go` (add `__model_setup` handler)
- Modify: `internal/app/run.go` (route `__model_setup` in `processInput`)
- Test: `internal/app/commands_model_setup_test.go`

When the user enters a token in the setup popup, the TUI sends `__model_setup <presetID> <token>` to the user channel. The backend resolves the preset, applies the token via `SetProvider`, saves to `config.json`, and emits success/failure events.

- [ ] **Step 1: Write failing test**

```go
package app

import (
	"strings"
	"testing"

	"github.com/vigo999/mindspore-code/ui/model"
)

func TestCmdModelSetup_SavesConfigAndSwitchesPreset(t *testing.T) {
	dir := t.TempDir()
	origPath := appConfigPathOverride
	appConfigPathOverride = dir + "/config.json"
	t.Cleanup(func() { appConfigPathOverride = origPath })

	eventCh := make(chan model.Event, 64)
	app := &Application{
		EventCh: eventCh,
		Config:  testConfig(),
	}

	app.cmdModelSetup([]string{"kimi-k2.5-free", "sk-test-token"})

	// Drain events and check for success
	var gotUpdate bool
	for len(eventCh) > 0 {
		ev := <-eventCh
		if ev.Type == model.ModelUpdate {
			gotUpdate = true
		}
	}
	if !gotUpdate {
		t.Error("expected ModelUpdate event after setup")
	}

	// Verify config.json was saved
	cfg, err := loadAppConfig()
	if err != nil {
		t.Fatalf("loadAppConfig: %v", err)
	}
	if cfg.ModelMode != "mscli-provided" {
		t.Errorf("ModelMode = %q, want 'mscli-provided'", cfg.ModelMode)
	}
	if cfg.ModelPresetID != "kimi-k2.5-free" {
		t.Errorf("ModelPresetID = %q, want 'kimi-k2.5-free'", cfg.ModelPresetID)
	}
	if cfg.ModelToken != "sk-test-token" {
		t.Errorf("ModelToken = %q, want 'sk-test-token'", cfg.ModelToken)
	}
}

func TestCmdModelSetup_InvalidPreset(t *testing.T) {
	eventCh := make(chan model.Event, 64)
	app := &Application{
		EventCh: eventCh,
		Config:  testConfig(),
	}

	app.cmdModelSetup([]string{"nonexistent-preset", "sk-token"})

	var gotError bool
	for len(eventCh) > 0 {
		ev := <-eventCh
		if ev.Type == model.ToolError && strings.Contains(ev.Message, "unknown preset") {
			gotError = true
		}
	}
	if !gotError {
		t.Error("expected error event for unknown preset")
	}
}
```

NOTE: `testConfig()` should return a `*configs.Config` with default values. If there is already a test helper for this in the codebase, use it. Otherwise create a minimal one:

```go
func testConfig() *configs.Config {
	return configs.DefaultConfig()
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/app/ -run TestCmdModelSetup -v`
Expected: FAIL — `cmdModelSetup` not defined.

- [ ] **Step 3: Write implementation**

In `internal/app/commands.go`, add the handler:

```go
func (a *Application) cmdModelSetup(args []string) {
	if len(args) < 2 {
		a.EventCh <- model.Event{
			Type:    model.ToolError,
			ToolName: "model",
			Message: "model setup requires preset ID and token",
		}
		return
	}
	presetID := args[0]
	token := strings.TrimSpace(args[1])

	preset, ok := resolveBuiltinModelPreset(presetID)
	if !ok {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "model",
			Message:  fmt.Sprintf("unknown preset: %s", presetID),
		}
		return
	}

	a.EventCh <- model.Event{Type: model.AgentThinking}

	if a.modelBeforePreset == nil {
		a.modelBeforePreset = copyModelConfig(a.Config.Model)
	}

	previous := a.Config.Model
	a.Config.Model.URL = preset.BaseURL
	err := a.SetProvider(preset.Provider, preset.Model, token)
	if err != nil {
		a.Config.Model = previous
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "model",
			Message:  fmt.Sprintf("Failed to apply preset: %v", err),
		}
		// Emit event to show error on token screen
		a.EventCh <- model.Event{
			Type:    model.ModelSetupTokenError,
			Message: fmt.Sprintf("Failed: %v", err),
		}
		return
	}
	a.activeModelPresetID = preset.ID

	// Save to config.json
	if err := saveAppConfig(&appConfig{
		ModelMode:     "mscli-provided",
		ModelPresetID: preset.ID,
		ModelToken:    token,
	}); err != nil {
		a.emitToolError("config", "model applied but failed to save config: %v", err)
	}

	a.EventCh <- model.Event{
		Type:    model.ModelUpdate,
		Message: a.Config.Model.Model,
		CtxMax:  a.Config.Context.Window,
	}

	// Close the setup popup
	a.EventCh <- model.Event{Type: model.ModelSetupClose}

	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("Model configured: %s", preset.Label),
	}
}
```

Add new event types in `ui/model/model.go`:

```go
ModelSetupClose      EventType = "ModelSetupClose"
ModelSetupTokenError EventType = "ModelSetupTokenError"
```

In `internal/app/run.go` `processInput`, add routing before the `/` command check:

```go
if strings.HasPrefix(trimmed, "__model_setup ") {
    parts := strings.Fields(trimmed)
    a.cmdModelSetup(parts[1:])
    return
}
```

In `ui/app.go`, handle the new events:

```go
case model.ModelSetupClose:
    a.setupPopup = nil
    return a, nil

case model.ModelSetupTokenError:
    if a.setupPopup != nil {
        a.setupPopup.TokenError = msg.Message
    }
    return a, nil
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./internal/app/ -run TestCmdModelSetup -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add internal/app/commands.go internal/app/commands_model_setup_test.go internal/app/run.go ui/app.go ui/model/model.go
git commit -m "feat: backend handler for setup popup token submission with config.json persistence"
```

---

### Task 7: Startup detection and auto-popup

**Files:**
- Modify: `internal/app/wire.go` (add detection logic, emit `ModelSetupOpen` when needed)
- Test: `internal/app/wire_setup_test.go`

This is the core startup logic change. After provider init, check whether setup should be shown.

- [ ] **Step 1: Write failing test**

```go
package app

import (
	"os"
	"testing"
)

func TestDetectModelMode_EnvWins(t *testing.T) {
	t.Setenv("MSCLI_PROVIDER", "openai-completion")
	t.Setenv("MSCLI_API_KEY", "sk-test")
	t.Setenv("MSCLI_MODEL", "gpt-4o")

	mode := detectModelMode()
	if mode != "own-env" {
		t.Errorf("expected 'own-env', got %q", mode)
	}
}

func TestDetectModelMode_SavedToken(t *testing.T) {
	// Clear env
	t.Setenv("MSCLI_PROVIDER", "")
	t.Setenv("MSCLI_API_KEY", "")
	t.Setenv("MSCLI_MODEL", "")

	dir := t.TempDir()
	origPath := appConfigPathOverride
	appConfigPathOverride = dir + "/config.json"
	t.Cleanup(func() { appConfigPathOverride = origPath })

	if err := saveAppConfig(&appConfig{
		ModelMode:     "mscli-provided",
		ModelPresetID: "kimi-k2.5-free",
		ModelToken:    "sk-saved",
	}); err != nil {
		t.Fatal(err)
	}

	mode := detectModelMode()
	if mode != "mscli-provided" {
		t.Errorf("expected 'mscli-provided', got %q", mode)
	}
}

func TestDetectModelMode_NothingConfigured(t *testing.T) {
	t.Setenv("MSCLI_PROVIDER", "")
	t.Setenv("MSCLI_API_KEY", "")
	t.Setenv("MSCLI_MODEL", "")

	dir := t.TempDir()
	origPath := appConfigPathOverride
	appConfigPathOverride = dir + "/nonexistent/config.json"
	t.Cleanup(func() { appConfigPathOverride = origPath })

	mode := detectModelMode()
	if mode != "" {
		t.Errorf("expected empty string, got %q", mode)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/app/ -run TestDetectModelMode -v`
Expected: FAIL — `detectModelMode` not defined.

- [ ] **Step 3: Write implementation**

Add to `internal/app/wire.go`:

```go
// detectModelMode checks whether model config is already available.
// Returns "own-env" if env vars are complete, "mscli-provided" if a saved
// token exists, or "" if neither is configured.
func detectModelMode() string {
	provider := strings.TrimSpace(os.Getenv("MSCLI_PROVIDER"))
	apiKey := strings.TrimSpace(os.Getenv("MSCLI_API_KEY"))
	model := strings.TrimSpace(os.Getenv("MSCLI_MODEL"))
	if provider != "" && apiKey != "" && model != "" {
		return "own-env"
	}

	cfg, err := loadAppConfig()
	if err != nil {
		return ""
	}
	if cfg.ModelMode == "mscli-provided" &&
		strings.TrimSpace(cfg.ModelPresetID) != "" &&
		strings.TrimSpace(cfg.ModelToken) != "" {
		return "mscli-provided"
	}
	return ""
}
```

In the `Wire` function, after provider init, if `llmReady` is false, check detection and either apply saved config or mark that the setup popup should open:

```go
// After the existing provider init block:
var needsSetupPopup bool

if !llmReady {
    mode := detectModelMode()
    switch mode {
    case "mscli-provided":
        appCfg, _ := loadAppConfig()
        if preset, ok := resolveBuiltinModelPreset(appCfg.ModelPresetID); ok {
            config.Model.URL = preset.BaseURL
            config.Model.Provider = preset.Provider
            config.Model.Model = preset.Model
            config.Model.Key = appCfg.ModelToken
            configs.RefreshModelTokenDefaults(config, previousModel)
            provider, err = initProvider(config.Model, llm.ResolveOptions{PreferConfigAPIKey: true})
            if err == nil {
                llmReady = true
            }
        }
    case "own-env":
        // Already handled above — env vars were applied by LoadWithEnv.
        // If we're here with own-env, the API key was set but init still failed
        // for another reason. Don't show the popup.
    default:
        needsSetupPopup = true
    }
}
```

Add `needsSetupPopup` to `Application` struct:

```go
needsSetupPopup bool
```

In `runReal()`, after the boot animation setup, emit the setup popup if needed:

```go
if a.needsSetupPopup {
    a.emitModelSetupPopup(false) // canEscape=false on first boot
}
```

Add the `emitModelSetupPopup` helper:

```go
func (a *Application) emitModelSetupPopup(canEscape bool) {
	presetOptions := []model.SelectionOption{}
	for _, preset := range listBuiltinModelPresets() {
		presetOptions = append(presetOptions, model.SelectionOption{
			ID:       preset.ID,
			Label:    preset.Label,
			Disabled: preset.ComingSoon,
		})
	}

	currentMode := ""
	currentPreset := ""
	if a.activeModelPresetID != "" {
		currentMode = "mscli-provided"
		currentPreset = a.activeModelPresetID
	} else if a.llmReady {
		currentMode = "own"
	}

	// Pre-fill token from saved config if available
	savedToken := ""
	if appCfg, err := loadAppConfig(); err == nil {
		savedToken = appCfg.ModelToken
	}

	popup := &model.SetupPopup{
		Screen:         model.SetupScreenModeSelect,
		PresetOptions:  presetOptions,
		CanEscape:      canEscape,
		CurrentMode:    currentMode,
		CurrentPreset:  currentPreset,
		TokenValue:     savedToken,
	}

	a.EventCh <- model.Event{
		Type:       model.ModelSetupOpen,
		SetupPopup: popup,
	}
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./internal/app/ -run TestDetectModelMode -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add internal/app/wire.go internal/app/wire_setup_test.go
git commit -m "feat: startup detection — auto-show setup popup when no model config exists"
```

---

### Task 8: Rewire `/model` command to open setup popup

**Files:**
- Modify: `internal/app/commands.go` (`cmdModel` and `openModelPicker`)
- Modify: `internal/app/commands_model_test.go` (update existing tests)

- [ ] **Step 1: Write failing test**

```go
// Add to existing commands_model_test.go:
func TestCmdModel_NoArgsOpensSetupPopup(t *testing.T) {
	eventCh := make(chan model.Event, 64)
	app := &Application{
		EventCh: eventCh,
		Config:  testConfig(),
	}

	app.cmdModel(nil)

	var gotSetup bool
	for len(eventCh) > 0 {
		ev := <-eventCh
		if ev.Type == model.ModelSetupOpen {
			gotSetup = true
			if ev.SetupPopup == nil {
				t.Error("expected non-nil SetupPopup")
			}
			if !ev.SetupPopup.CanEscape {
				t.Error("expected CanEscape=true from /model command")
			}
		}
	}
	if !gotSetup {
		t.Error("expected ModelSetupOpen event")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/app/ -run TestCmdModel_NoArgsOpensSetupPopup -v`
Expected: FAIL — `ModelSetupOpen` not emitted (still emitting `ModelPickerOpen`).

- [ ] **Step 3: Write implementation**

Replace `openModelPicker()` with call to `emitModelSetupPopup(true)`:

```go
func (a *Application) cmdModel(args []string) {
	if len(args) == 0 {
		a.emitModelSetupPopup(true) // canEscape=true from /model
		return
	}

	// Rest unchanged — direct preset/model switching still works.
	modelArg := strings.TrimSpace(strings.Join(args, " "))
	if preset, ok := resolveBuiltinModelPreset(modelArg); ok {
		a.switchToBuiltinModelPreset(preset)
		return
	}

	a.restoreModelConfigFromPreset()
	modelArg = args[0]
	if strings.Contains(modelArg, ":") {
		parts := strings.SplitN(modelArg, ":", 2)
		providerName := llm.NormalizeProvider(parts[0])
		modelName := strings.TrimSpace(parts[1])
		if !llm.IsSupportedProvider(providerName) {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Unsupported provider prefix: %s (supported: openai-completion, openai-responses, anthropic)", providerName),
			}
			return
		}
		a.switchModel(providerName, modelName)
		return
	}

	a.switchModel("", modelArg)
}
```

Remove `openModelPicker()` function entirely (it's now replaced by `emitModelSetupPopup`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./internal/app/ -run "TestCmdModel" -v`
Expected: PASS. Existing tests for `/model <preset>`, `/model provider:model`, and `/model model` should still pass unchanged.

- [ ] **Step 5: Commit**

```bash
git add internal/app/commands.go internal/app/commands_model_test.go
git commit -m "feat: /model command now opens unified setup popup instead of old model picker"
```

---

### Task 9: Update `/help` text and clean up old model picker code

**Files:**
- Modify: `internal/app/commands.go` (update help text)
- Modify: `ui/app.go` (remove old `modelPicker` field usage, redirect to `setupPopup`)
- Modify: `ui/slash/commands.go` (update `/model` description)

- [ ] **Step 1: Update `/help` text**

In `cmdHelp()` in `internal/app/commands.go`, update the model commands section:

```go
// Replace the Model Commands section with:
Model Commands:
  /model                  Open model setup (mscli-provided or your own)
  /model kimi-k2.5-free   Switch to built-in preset directly
  /model gpt-4o           Switch model (keeps current provider)
  /model openai-completion:gpt-4o  Switch provider and model
```

- [ ] **Step 2: Update slash command description**

In `ui/slash/commands.go`, update the `/model` command registration:

```go
Command{
    Name:        "/model",
    Description: "Open model setup or switch model",
    Usage:       "/model [preset-id|provider:model|model]",
}
```

- [ ] **Step 3: Remove old `modelPicker` references**

In `ui/app.go`:
- Remove the `modelPicker *model.SelectionPopup` field from the App struct.
- Remove the old `modelPicker` keyboard handling block (the `setupPopup` block from Task 5 replaces it).
- Remove the old `modelPicker` rendering in `View()`.
- Remove the `model.ModelPickerOpen` event handler.

Keep `model.ModelPickerOpen` event type defined in `ui/model/model.go` for now (to avoid breaking imports), but it will no longer be emitted.

- [ ] **Step 4: Run all tests**

Run: `go test ./... 2>&1 | head -50`
Expected: All tests pass. Fix any compilation errors from removed `modelPicker` references.

- [ ] **Step 5: Commit**

```bash
git add internal/app/commands.go ui/app.go ui/slash/commands.go
git commit -m "refactor: remove old model picker, update /help and /model description"
```

---

### Task 10: Integration test — full startup-to-model-switch flow

**Files:**
- Create: `internal/app/model_setup_integration_test.go`

End-to-end test that verifies the three startup paths work correctly.

- [ ] **Step 1: Write integration tests**

```go
package app

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vigo999/mindspore-code/ui/model"
)

func TestStartup_NoConfig_EmitsSetupPopup(t *testing.T) {
	t.Setenv("MSCLI_PROVIDER", "")
	t.Setenv("MSCLI_API_KEY", "")
	t.Setenv("MSCLI_MODEL", "")

	dir := t.TempDir()
	origPath := appConfigPathOverride
	appConfigPathOverride = filepath.Join(dir, "config.json")
	t.Cleanup(func() { appConfigPathOverride = origPath })

	mode := detectModelMode()
	if mode != "" {
		t.Errorf("expected empty mode, got %q", mode)
	}
}

func TestStartup_EnvComplete_SkipsPopup(t *testing.T) {
	t.Setenv("MSCLI_PROVIDER", "openai-completion")
	t.Setenv("MSCLI_API_KEY", "sk-test-key")
	t.Setenv("MSCLI_MODEL", "gpt-4o")

	mode := detectModelMode()
	if mode != "own-env" {
		t.Errorf("expected 'own-env', got %q", mode)
	}
}

func TestStartup_SavedToken_SkipsPopup(t *testing.T) {
	t.Setenv("MSCLI_PROVIDER", "")
	t.Setenv("MSCLI_API_KEY", "")
	t.Setenv("MSCLI_MODEL", "")

	dir := t.TempDir()
	origPath := appConfigPathOverride
	appConfigPathOverride = filepath.Join(dir, "config.json")
	t.Cleanup(func() { appConfigPathOverride = origPath })

	saveAppConfig(&appConfig{
		ModelMode:     "mscli-provided",
		ModelPresetID: "kimi-k2.5-free",
		ModelToken:    "sk-saved-token",
	})

	mode := detectModelMode()
	if mode != "mscli-provided" {
		t.Errorf("expected 'mscli-provided', got %q", mode)
	}
}

func TestStartup_BothEnvAndSavedToken_EnvWins(t *testing.T) {
	t.Setenv("MSCLI_PROVIDER", "openai-completion")
	t.Setenv("MSCLI_API_KEY", "sk-env-key")
	t.Setenv("MSCLI_MODEL", "gpt-4o")

	dir := t.TempDir()
	origPath := appConfigPathOverride
	appConfigPathOverride = filepath.Join(dir, "config.json")
	t.Cleanup(func() { appConfigPathOverride = origPath })

	saveAppConfig(&appConfig{
		ModelMode:     "mscli-provided",
		ModelPresetID: "kimi-k2.5-free",
		ModelToken:    "sk-saved-token",
	})

	mode := detectModelMode()
	if mode != "own-env" {
		t.Errorf("expected 'own-env' (env wins), got %q", mode)
	}
}

func TestModelSetup_TokenApplied_PresetActive(t *testing.T) {
	dir := t.TempDir()
	origPath := appConfigPathOverride
	appConfigPathOverride = filepath.Join(dir, "config.json")
	t.Cleanup(func() { appConfigPathOverride = origPath })

	eventCh := make(chan model.Event, 64)
	app := &Application{
		EventCh: eventCh,
		Config:  testConfig(),
	}

	app.cmdModelSetup([]string{"kimi-k2.5-free", "sk-my-token"})

	// Verify the preset is now active
	if app.activeModelPresetID != "kimi-k2.5-free" {
		t.Errorf("expected active preset 'kimi-k2.5-free', got %q", app.activeModelPresetID)
	}
	if app.Config.Model.Model != "kimi-k2.5" {
		t.Errorf("expected model 'kimi-k2.5', got %q", app.Config.Model.Model)
	}

	// Verify config.json saved
	cfg, _ := loadAppConfig()
	if cfg.ModelToken != "sk-my-token" {
		t.Errorf("expected saved token 'sk-my-token', got %q", cfg.ModelToken)
	}
}

func TestModelCommand_OpensSetupPopup(t *testing.T) {
	eventCh := make(chan model.Event, 64)
	app := &Application{
		EventCh:             eventCh,
		Config:              testConfig(),
		activeModelPresetID: "kimi-k2.5-free",
	}

	app.cmdModel(nil) // /model with no args

	var popup *model.SetupPopup
	for len(eventCh) > 0 {
		ev := <-eventCh
		if ev.Type == model.ModelSetupOpen {
			popup = ev.SetupPopup
		}
	}
	if popup == nil {
		t.Fatal("expected setup popup to open")
	}
	if !popup.CanEscape {
		t.Error("expected CanEscape=true from /model")
	}
	if popup.CurrentMode != "mscli-provided" {
		t.Errorf("expected current mode 'mscli-provided', got %q", popup.CurrentMode)
	}
	if popup.CurrentPreset != "kimi-k2.5-free" {
		t.Errorf("expected current preset 'kimi-k2.5-free', got %q", popup.CurrentPreset)
	}
	// Verify coming-soon presets are in the list but disabled
	disabledCount := 0
	for _, opt := range popup.PresetOptions {
		if opt.Disabled {
			disabledCount++
			if !strings.Contains(opt.Label, "coming soon") {
				t.Errorf("disabled option %q should contain 'coming soon'", opt.Label)
			}
		}
	}
	if disabledCount != 3 {
		t.Errorf("expected 3 disabled (coming soon) presets, got %d", disabledCount)
	}
}
```

- [ ] **Step 2: Run tests**

Run: `go test ./internal/app/ -run "TestStartup_|TestModelSetup_|TestModelCommand_" -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add internal/app/model_setup_integration_test.go
git commit -m "test: integration tests for startup detection and model setup flow"
```

---

### Task 11: Final cleanup and full test suite verification

**Files:**
- Modify: `ui/app_model_picker_test.go` (update or remove old model picker test)

- [ ] **Step 1: Update old model picker test**

The existing `TestModelPickerOpenAndEnterDispatchesModelCommand` in `ui/app_model_picker_test.go` tests the old popup. Update it to test the new setup popup, or delete it if Task 5's tests cover the same behavior.

- [ ] **Step 2: Run full test suite**

Run: `go test ./... 2>&1 | tail -30`
Expected: All tests pass. No compilation errors.

- [ ] **Step 3: Run vet and build**

Run: `go vet ./... && go build ./cmd/mscli/ && go build ./cmd/mscli-server/`
Expected: Clean.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "refactor: clean up old model picker test, verify full test suite"
```

---

## Summary of files changed

| File | Action | Purpose |
|------|--------|---------|
| `internal/app/appconfig.go` | Create | `config.json` load/save for model mode + token |
| `internal/app/appconfig_test.go` | Create | Tests for config.json round-trip |
| `internal/app/model_presets.go` | Modify | Add `ComingSoon` flag, 3 placeholder presets |
| `internal/app/model_presets_test.go` | Create | Tests for presets with ComingSoon |
| `ui/model/model.go` | Modify | Add `ModelSetupOpen`, `ModelSetupClose`, `ModelSetupTokenError` event types, `SetupPopup` field on `Event` |
| `ui/model/train.go` | Modify | Add `SetupPopup` struct, `SetupScreen` enum, `Disabled` on `SelectionOption` |
| `ui/model/popup_test.go` | Create | Tests for SetupPopup navigation |
| `ui/panels/model_setup.go` | Create | Render all 4 setup popup screens |
| `ui/panels/model_setup_test.go` | Create | Tests for popup rendering |
| `ui/app.go` | Modify | Wire setup popup events, keyboard handling, rendering; remove old model picker |
| `ui/app_model_setup_test.go` | Create | Tests for TUI setup popup interaction |
| `internal/app/commands.go` | Modify | Add `cmdModelSetup`, replace `openModelPicker` with `emitModelSetupPopup`, update help text |
| `internal/app/commands_model_setup_test.go` | Create | Tests for model setup backend handler |
| `internal/app/run.go` | Modify | Route `__model_setup` command |
| `internal/app/wire.go` | Modify | Add `detectModelMode`, apply saved config at startup, set `needsSetupPopup` |
| `internal/app/wire_setup_test.go` | Create | Tests for startup detection |
| `internal/app/model_setup_integration_test.go` | Create | End-to-end integration tests |
| `ui/slash/commands.go` | Modify | Update `/model` description |
| `ui/app_model_picker_test.go` | Modify | Update or remove old model picker test |
