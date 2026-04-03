package components

import (
	"strings"
	"unicode"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/textarea"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	rw "github.com/mattn/go-runewidth"
	"github.com/rivo/uniseg"
	// uirender "github.com/vigo999/mindspore-code/ui/render"
	"github.com/vigo999/mindspore-code/ui/slash"
)

// Style vars are populated by InitStyles() in styles.go.
// composerStyle is layout-only (not themed).
var (
	sugCmdStyle     lipgloss.Style
	sugDescStyle    lipgloss.Style
	sugSelCmdStyle  lipgloss.Style
	sugSelDescStyle lipgloss.Style
	separatorStyle  lipgloss.Style
	composerStyle   = lipgloss.NewStyle().PaddingLeft(2)
)

const (
	maxVisibleSuggestions = 8
	minComposerRows       = 1
	composerPrompt        = "❯ "
	composerContinue      = "  "
)

// TextInput wraps a multiline textarea for the chat composer.
type TextInput struct {
	Model            textarea.Model
	slashRegistry    *slash.Registry
	showSuggestions  bool
	slashMode        bool // true once suggestions have been shown, until submit/esc
	suggestions      []string
	selectedIdx      int
	suggestionOffset int
	history          []string
	historyIndex     int
	historyDraft     string
	width            int
	maxVisibleRows   int // 0 = unlimited; when set, the editor becomes scrollable
}

// NewTextInput creates a focused multiline composer with a prompt.
func NewTextInput() TextInput {
	ti := textarea.New()
	ti.ShowLineNumbers = false
	ti.CharLimit = 0
	ti.MaxHeight = 0
	ti.Prompt = composerPrompt
	ti.KeyMap.InsertNewline = key.NewBinding(
		key.WithKeys("shift+enter", "ctrl+j"),
		key.WithHelp("shift+enter", "insert newline"),
	)
	ti.FocusedStyle.CursorLine = lipgloss.NewStyle()
	ti.BlurredStyle.CursorLine = lipgloss.NewStyle()
	ti.FocusedStyle.Prompt = lipgloss.NewStyle().Foreground(lipgloss.Color("252")).Bold(true)
	ti.BlurredStyle.Prompt = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
	ti.FocusedStyle.Placeholder = lipgloss.NewStyle().Foreground(lipgloss.Color("241"))
	ti.BlurredStyle.Placeholder = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
	ti.SetPromptFunc(lipgloss.Width(composerPrompt), func(line int) string {
		if line == 0 {
			return composerPrompt
		}
		return composerContinue
	})
	ti.Focus()

	input := TextInput{
		Model:         ti,
		slashRegistry: slash.DefaultRegistry,
		historyIndex:  -1,
	}
	input.syncHeight()
	return input
}

// Value returns the current input text.
func (t TextInput) Value() string {
	return t.Model.Value()
}

// Reset clears the input.
func (t TextInput) Reset() TextInput {
	t.Model.Reset()
	t.syncHeight()
	t.showSuggestions = false
	// Keep slashMode — it gets cleared when the command result arrives.
	t.suggestions = nil
	t.selectedIdx = 0
	t.suggestionOffset = 0
	t.historyIndex = -1
	t.historyDraft = ""
	return t
}

// Focus gives the input focus.
func (t TextInput) Focus() (TextInput, tea.Cmd) {
	cmd := t.Model.Focus()
	return t, cmd
}

// Blur removes focus from the input.
func (t TextInput) Blur() TextInput {
	t.Model.Blur()
	return t
}

// SetWidth updates the rendered input width.
func (t TextInput) SetWidth(width int) TextInput {
	if width < 1 {
		width = 1
	}
	t.Model.SetWidth(width)
	t.width = width
	t.syncHeight()
	return t
}

// SetMaxVisibleRows sets the maximum number of editor rows before the
// composer becomes internally scrollable.  0 means unlimited.
func (t TextInput) SetMaxVisibleRows(rows int) TextInput {
	if rows < 0 {
		rows = 0
	}
	t.maxVisibleRows = rows
	t.syncHeight()
	return t
}

// Update handles key events.
func (t TextInput) Update(msg tea.Msg) (TextInput, tea.Cmd) {
	var isPaste bool
	switch msg := msg.(type) {
	case tea.KeyMsg:
		isPaste = msg.Paste
		t.maybeGrowHeightBeforeUpdate(msg)

		// Handle slash command suggestions navigation
		if t.showSuggestions && len(t.suggestions) > 0 {
			switch msg.String() {
			case "up":
				if t.selectedIdx > 0 {
					t.selectedIdx--
				} else {
					// Wrap to last
					t.selectedIdx = len(t.suggestions) - 1
				}
				t.syncSuggestionWindow()
				return t, nil
			case "down":
				if t.selectedIdx < len(t.suggestions)-1 {
					t.selectedIdx++
				} else {
					// Wrap to first
					t.selectedIdx = 0
				}
				t.syncSuggestionWindow()
				return t, nil
			case "tab", "enter":
				// Accept selected suggestion
				if t.selectedIdx < len(t.suggestions) {
					val := t.suggestions[t.selectedIdx] + " "
					t.Model.SetValue(val)
					t.Model.SetCursor(len(val))
					t.syncHeight()
					t.showSuggestions = false
					t.suggestions = nil
					t.suggestionOffset = 0
				}
				return t, nil
			case "esc":
				// Cancel suggestions
				t.showSuggestions = false
				t.slashMode = false
				t.suggestions = nil
				t.suggestionOffset = 0
				return t, nil
			}
		}
	}

	m, cmd := t.Model.Update(msg)
	t.Model = m

	// After a bracketed paste the textarea's internal viewport may have
	// scrolled to a stale offset.  Re-setting the value via SetValue
	// clears the viewport state (Reset → GotoTop), then we restore the
	// cursor to where the paste left it and let scrollToCursor position
	// the viewport so the cursor is visible.
	if isPaste {
		savedRow := t.Model.Line()
		li := t.Model.LineInfo()
		savedCol := li.StartColumn + li.ColumnOffset

		t.Model.SetValue(t.Model.Value())

		for i := 0; t.Model.Line() > savedRow && i < 10000; i++ {
			t.Model.CursorUp()
		}
		t.Model.SetCursor(savedCol)
	}

	t.syncHeight()

	// For content that exceeds the visible height (paste or any other
	// insertion that grew past the cap), ensure the cursor is visible.
	if isPaste {
		t.scrollToCursor()
	}

	// Update suggestions based on current input
	t.updateSuggestions()

	return t, cmd
}

// ConsumeEscapedEnter converts a backslash immediately before the cursor into a newline.
// It returns the updated input and whether the escape was consumed.
func (t TextInput) ConsumeEscapedEnter() (TextInput, bool) {
	row, col, lines := t.cursorPosition()
	if row < 0 || row >= len(lines) || col <= 0 {
		return t, false
	}

	current := []rune(lines[row])
	if col > len(current) || current[col-1] != '\\' {
		return t, false
	}

	left := string(current[:col-1])
	right := string(current[col:])

	updatedLines := make([]string, 0, len(lines)+1)
	updatedLines = append(updatedLines, lines[:row]...)
	updatedLines = append(updatedLines, left, right)
	updatedLines = append(updatedLines, lines[row+1:]...)

	t.Model.SetValue(strings.Join(updatedLines, "\n"))
	targetRow := row + 1
	t.Model.SetCursor(0)
	for t.Model.Line() > targetRow {
		t.Model.CursorUp()
	}
	for t.Model.Line() < targetRow {
		t.Model.CursorDown()
	}
	t.Model.SetCursor(0)
	t.syncHeight()
	t.updateSuggestions()

	return t, true
}

// PushHistory stores a submitted input line for later up/down recall.
func (t TextInput) PushHistory(value string) TextInput {
	t, _ = t.recordHistory(value)
	return t
}

// RecordHistory stores a submitted input line and reports whether it was added.
func (t TextInput) RecordHistory(value string) (TextInput, bool) {
	return t.recordHistory(value)
}

func (t TextInput) recordHistory(value string) (TextInput, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return t, false
	}
	if n := len(t.history); n > 0 && t.history[n-1] == value {
		t.historyIndex = -1
		t.historyDraft = ""
		return t, false
	}
	t.history = append(t.history, value)
	t.historyIndex = -1
	t.historyDraft = ""
	return t, true
}

// SeedHistory preloads older persisted history in oldest-first order.
func (t TextInput) SeedHistory(values []string) TextInput {
	if len(values) == 0 {
		return t
	}
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if n := len(t.history); n > 0 && t.history[n-1] == value {
			continue
		}
		t.history = append(t.history, value)
	}
	t.historyIndex = -1
	t.historyDraft = ""
	return t
}

// PrevHistory recalls the previous submitted line.
func (t TextInput) PrevHistory() TextInput {
	if len(t.history) == 0 {
		return t
	}
	if t.historyIndex == -1 {
		t.historyDraft = t.Model.Value()
		t.historyIndex = len(t.history) - 1
	} else if t.historyIndex > 0 {
		t.historyIndex--
	}
	t.Model.SetValue(t.history[t.historyIndex])
	t.syncHeight()
	t.scrollToCursor()
	t.showSuggestions = false
	t.slashMode = false
	t.suggestions = nil
	t.suggestionOffset = 0
	return t
}

// NextHistory moves forward in submitted-line history, restoring the draft at the end.
func (t TextInput) NextHistory() TextInput {
	if len(t.history) == 0 || t.historyIndex == -1 {
		return t
	}
	if t.historyIndex < len(t.history)-1 {
		t.historyIndex++
		t.Model.SetValue(t.history[t.historyIndex])
		t.syncHeight()
		t.scrollToCursor()
		t.showSuggestions = false
		t.slashMode = false
		t.suggestions = nil
		t.suggestionOffset = 0
		return t
	}
	t.historyIndex = -1
	t.Model.SetValue(t.historyDraft)
	t.syncHeight()
	t.scrollToCursor()
	t.historyDraft = ""
	t.showSuggestions = false
	t.slashMode = false
	t.suggestions = nil
	t.suggestionOffset = 0
	return t
}

// updateSuggestions updates the slash command suggestions based on current input.
func (t *TextInput) updateSuggestions() {
	val := t.Model.Value()
	val = strings.TrimSpace(val)

	// Only show suggestions if input starts with "/"
	if !strings.HasPrefix(val, "/") {
		t.showSuggestions = false
		t.slashMode = false
		t.suggestions = nil
		t.selectedIdx = 0
		t.suggestionOffset = 0
		return
	}

	// Get suggestions
	t.suggestions = t.slashRegistry.Suggestions(val)
	t.showSuggestions = len(t.suggestions) > 0
	if t.showSuggestions {
		t.slashMode = true
	}

	// Reset selection if it's out of bounds
	if t.selectedIdx >= len(t.suggestions) {
		t.selectedIdx = 0
	}
	if len(t.suggestions) == 0 {
		t.suggestionOffset = 0
		return
	}
	t.syncSuggestionWindow()
}

func (t TextInput) separator() string {
	width := t.width
	if width < 1 {
		width = 1
	}
	return separatorStyle.Render(strings.Repeat("─", width+4))
}

// View renders the input with optional suggestions.
func (t TextInput) View() string {
	sep := t.separator()
	inputView := composerStyle.Render(t.Model.View())

	if !t.showSuggestions || len(t.suggestions) == 0 {
		if t.slashMode {
			return sep + "\n" + inputView + strings.Repeat("\n", maxVisibleSuggestions) + "\n" + sep
		}
		return sep + "\n" + inputView + "\n" + sep
	}

	// Render suggestions below input
	var sb strings.Builder
	sb.WriteString(sep)
	sb.WriteString("\n")
	sb.WriteString(inputView)
	sb.WriteString("\n")

	start := t.suggestionOffset
	if start < 0 {
		start = 0
	}
	end := start + maxVisibleSuggestions
	if end > len(t.suggestions) {
		end = len(t.suggestions)
	}

	for i := start; i < end; i++ {
		sug := t.suggestions[i]

		// Get command description
		cmd, ok := t.slashRegistry.Get(sug)
		if !ok {
			continue
		}

		if i == t.selectedIdx {
			sb.WriteString("    ")
			sb.WriteString(sugSelCmdStyle.Render(sug))
			sb.WriteString("  ")
			sb.WriteString(sugSelDescStyle.Render(cmd.Description))
		} else {
			sb.WriteString("    ")
			sb.WriteString(sugCmdStyle.Render(sug))
			sb.WriteString("  ")
			sb.WriteString(sugDescStyle.Render(cmd.Description))
		}

		sb.WriteString("\n")
	}
	// Pad remaining rows to fill the fixed slash suggestion area.
	rendered := end - start
	for i := rendered; i < maxVisibleSuggestions; i++ {
		sb.WriteString("\n")
	}
	sb.WriteString(sep)

	return sb.String()
}

// Height returns the total height including suggestions area.
func (t TextInput) Height() int {
	return t.editorHeight() + t.ReservedHeight()
}

// ReservedHeight returns composer chrome outside the editor rows
// (separators, slash suggestion area).
func (t TextInput) ReservedHeight() int {
	h := 2 // top + bottom separator
	if t.slashMode {
		h += maxVisibleSuggestions
	}
	return h
}

// IsSlashMode returns true if showing slash suggestions.
func (t TextInput) IsSlashMode() bool {
	return t.showSuggestions
}

// ClearSlashMode exits the slash suggestion reserved area.
func (t TextInput) ClearSlashMode() TextInput {
	t.slashMode = false
	t.showSuggestions = false
	t.suggestions = nil
	t.suggestionOffset = 0
	return t
}

// HasSuggestions returns true if there are visible suggestion candidates.
func (t TextInput) HasSuggestions() bool {
	return t.showSuggestions && len(t.suggestions) > 0
}

func isExplicitNewlineKey(msg tea.KeyMsg) bool {
	switch msg.String() {
	case "ctrl+j", "shift+enter":
		return true
	default:
		return false
	}
}

// CanNavigateHistory reports whether up/down should switch prompt history
// instead of moving inside the editor.
func (t TextInput) CanNavigateHistory(direction string) bool {
	row, _, lines := t.cursorPosition()
	switch direction {
	case "up":
		return row == 0
	case "down":
		return len(lines) == 0 || row == len(lines)-1
	default:
		return false
	}
}

func (t *TextInput) syncSuggestionWindow() {
	if len(t.suggestions) == 0 {
		t.suggestionOffset = 0
		return
	}

	if t.selectedIdx < 0 {
		t.selectedIdx = 0
	}
	if t.selectedIdx >= len(t.suggestions) {
		t.selectedIdx = len(t.suggestions) - 1
	}

	if t.selectedIdx < t.suggestionOffset {
		t.suggestionOffset = t.selectedIdx
	}
	if t.selectedIdx >= t.suggestionOffset+maxVisibleSuggestions {
		t.suggestionOffset = t.selectedIdx - maxVisibleSuggestions + 1
	}

	maxOffset := len(t.suggestions) - maxVisibleSuggestions
	if maxOffset < 0 {
		maxOffset = 0
	}
	if t.suggestionOffset > maxOffset {
		t.suggestionOffset = maxOffset
	}
	if t.suggestionOffset < 0 {
		t.suggestionOffset = 0
	}
}

func (t *TextInput) syncHeight() {
	height := t.editorHeight()
	if t.Model.Height() == height {
		return
	}
	t.Model.SetHeight(height)
}

func (t TextInput) editorHeight() int {
	lines := t.visibleLineCountForValue(t.Model.Value())
	if lines < minComposerRows {
		lines = minComposerRows
	}
	if t.maxVisibleRows > 0 && lines > t.maxVisibleRows {
		return t.maxVisibleRows
	}
	return lines
}

func (t TextInput) visibleLineCountForValue(value string) int {
	width := t.Model.Width()
	if width < 1 {
		width = 1
	}

	total := 0
	for _, line := range splitLines(value) {
		total += wrappedLineCount([]rune(line), width)
	}
	if total < minComposerRows {
		return minComposerRows
	}
	return total
}

func (t *TextInput) maybeGrowHeightBeforeUpdate(msg tea.KeyMsg) {
	nextValue, ok := t.valueAfterKey(msg)
	if !ok {
		return
	}
	nextHeight := t.visibleLineCountForValue(nextValue)
	if t.maxVisibleRows > 0 && nextHeight > t.maxVisibleRows {
		nextHeight = t.maxVisibleRows
	}
	if nextHeight > t.Model.Height() {
		t.Model.SetHeight(nextHeight)
	}
}

// scrollToCursor makes the textarea's internal viewport show the cursor.
// When content exceeds maxVisibleRows the viewport must scroll; we force
// correct viewport content via View(), then run a lightweight Update(nil)
// which triggers the textarea's built-in repositionView().
func (t *TextInput) scrollToCursor() {
	if t.maxVisibleRows <= 0 {
		return // unlimited height — viewport always shows everything
	}
	if t.visibleLineCountForValue(t.Model.Value()) <= t.maxVisibleRows {
		return // all content fits — no scrolling needed
	}
	// Populate viewport.lines with current content so repositionView has
	// accurate data (without this, it operates on stale lines from the
	// previous render frame and the scroll math is wrong).
	_ = t.Model.View()
	// Run a minimal Update cycle — nil message triggers no key handling
	// but does run repositionView(), which scrolls the viewport to keep
	// the cursor visible.
	m, _ := t.Model.Update(nil)
	t.Model = m
}

func (t TextInput) valueAfterKey(msg tea.KeyMsg) (string, bool) {
	switch {
	case isExplicitNewlineKey(msg):
		return t.valueWithInsertedText("\n"), true
	case msg.Paste:
		return t.valueWithInsertedText(string(msg.Runes)), true
	case msg.Type == tea.KeyRunes:
		return t.valueWithInsertedText(string(msg.Runes)), true
	case msg.Type == tea.KeySpace:
		return t.valueWithInsertedText(" "), true
	default:
		return "", false
	}
}

func (t TextInput) valueWithInsertedText(text string) string {
	row, col, lines := t.cursorPosition()
	current := []rune(lines[row])
	lines[row] = string(current[:col]) + text + string(current[col:])
	return strings.Join(lines, "\n")
}

// wrappedLineCount matches bubbles/textarea soft-wrap behavior so the composer
// grows to the number of visible rows, not just the number of logical lines.
func wrappedLineCount(runes []rune, width int) int {
	return len(wrapRunes(runes, width))
}

func wrapRunes(runes []rune, width int) [][]rune {
	if width < 1 {
		return [][]rune{{}}
	}

	lines := [][]rune{{}}
	word := []rune{}
	row := 0
	spaces := 0

	for _, r := range runes {
		if unicode.IsSpace(r) {
			spaces++
		} else {
			word = append(word, r)
		}

		if spaces > 0 {
			wordWidth := uniseg.StringWidth(string(word))
			if uniseg.StringWidth(string(lines[row]))+wordWidth+spaces > width {
				row++
				lines = append(lines, []rune{})
				lines[row] = append(lines[row], word...)
				lines[row] = append(lines[row], repeatSpaces(spaces)...)
			} else {
				lines[row] = append(lines[row], word...)
				lines[row] = append(lines[row], repeatSpaces(spaces)...)
			}
			spaces = 0
			word = nil
			continue
		}

		if len(word) == 0 {
			continue
		}

		lastWidth := rw.RuneWidth(word[len(word)-1])
		if uniseg.StringWidth(string(word))+lastWidth > width {
			if len(lines[row]) > 0 {
				row++
				lines = append(lines, []rune{})
			}
			lines[row] = append(lines[row], word...)
			word = nil
		}
	}

	if uniseg.StringWidth(string(lines[row]))+uniseg.StringWidth(string(word))+spaces >= width {
		lines = append(lines, []rune{})
		lines[row+1] = append(lines[row+1], word...)
		spaces++
		lines[row+1] = append(lines[row+1], repeatSpaces(spaces)...)
	} else {
		lines[row] = append(lines[row], word...)
		spaces++
		lines[row] = append(lines[row], repeatSpaces(spaces)...)
	}

	return lines
}

func repeatSpaces(n int) []rune {
	return []rune(strings.Repeat(string(' '), n))
}

func (t TextInput) atInputStart() bool {
	row, col, lines := t.cursorPosition()
	return row == 0 && col == 0 && len(lines) > 0
}

func (t TextInput) atInputEnd() bool {
	row, col, lines := t.cursorPosition()
	if len(lines) == 0 {
		return true
	}
	return row == len(lines)-1 && col == len([]rune(lines[row]))
}

func (t TextInput) cursorPosition() (int, int, []string) {
	lines := t.lines()
	row := t.Model.Line()
	if row < 0 {
		row = 0
	}
	if row >= len(lines) {
		row = len(lines) - 1
	}

	info := t.Model.LineInfo()
	col := info.StartColumn + info.ColumnOffset
	if col < 0 {
		col = 0
	}
	maxCol := len([]rune(lines[row]))
	if col > maxCol {
		col = maxCol
	}
	return row, col, lines
}

func (t TextInput) lines() []string {
	return splitLines(t.Model.Value())
}

func splitLines(value string) []string {
	if value == "" {
		return []string{""}
	}
	return strings.Split(value, "\n")
}
