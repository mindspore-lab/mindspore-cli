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
	// uirender "github.com/vigo999/mindspore-cli/ui/render"
	"github.com/vigo999/mindspore-cli/ui/slash"
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

type suggestionKind int

const (
	suggestionKindNone suggestionKind = iota
	suggestionKindSlash
	suggestionKindFile
)

type suggestionItem struct {
	Value       string
	Display     string
	Description string
	Kind        suggestionKind
}

type tokenRange struct {
	start int
	end   int
}

// TextInput wraps a multiline textarea for the chat composer.
type TextInput struct {
	Model            textarea.Model
	slashRegistry    *slash.Registry
	fileSuggestion   *fileSuggestionProvider
	showSuggestions  bool
	suggestionKind   suggestionKind
	suggestionItems  []suggestionItem
	selectedIdx      int
	suggestionOffset int
	activeToken      tokenRange
	history          []string
	historyIndex     int
	historyDraft     string
	maskedPasteRaw   string
	maskedPasteLabel string
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

// WithFileSuggestions enables @file suggestions from the given workspace root.
func (t TextInput) WithFileSuggestions(workDir string) TextInput {
	t.fileSuggestion = newFileSuggestionProvider(workDir)
	return t
}

// Value returns the current input text.
func (t TextInput) Value() string {
	return t.Model.Value()
}

// Reset clears the input.
func (t TextInput) Reset() TextInput {
	t.Model.Reset()
	t.syncHeight()
	t = t.clearSuggestions()
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

		if t.showSuggestions && len(t.suggestionItems) > 0 {
			switch msg.String() {
			case "up":
				if t.selectedIdx > 0 {
					t.selectedIdx--
				} else {
					t.selectedIdx = len(t.suggestionItems) - 1
				}
				t.syncSuggestionWindow()
				return t, nil
			case "down":
				if t.selectedIdx < len(t.suggestionItems)-1 {
					t.selectedIdx++
				} else {
					t.selectedIdx = 0
				}
				t.syncSuggestionWindow()
				return t, nil
			case "tab", "enter":
				if t.selectedIdx < len(t.suggestionItems) {
					t = t.applySuggestion(t.suggestionItems[t.selectedIdx])
				}
				return t, nil
			case "esc":
				t = t.clearSuggestions()
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
	t.maskedPasteRaw = ""
	t.maskedPasteLabel = ""
	t = t.clearSuggestions()
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
		t.maskedPasteRaw = ""
		t.maskedPasteLabel = ""
		t = t.clearSuggestions()
		return t
	}
	t.historyIndex = -1
	t.Model.SetValue(t.historyDraft)
	t.syncHeight()
	t.scrollToCursor()
	t.historyDraft = ""
	t = t.clearSuggestions()
	return t
}

// updateSuggestions chooses between slash and @file suggestions based on the current token.
func (t *TextInput) updateSuggestions() {
	token, span, ok := t.currentToken()
	if !ok {
		*t = t.clearSuggestions()
		return
	}

	if items, ok := t.slashSuggestionItems(token, span); ok {
		t.setSuggestions(suggestionKindSlash, span, items)
		return
	}
	if items, ok := t.fileSuggestionItems(token, span); ok {
		t.setSuggestions(suggestionKindFile, span, items)
		return
	}

	*t = t.clearSuggestions()
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

	if !t.showSuggestions || len(t.suggestionItems) == 0 {
		if t.suggestionKind != suggestionKindNone {
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
	if end > len(t.suggestionItems) {
		end = len(t.suggestionItems)
	}

	for i := start; i < end; i++ {
		item := t.suggestionItems[i]

		if i == t.selectedIdx {
			sb.WriteString("    ")
			sb.WriteString(sugSelCmdStyle.Render(item.Display))
			sb.WriteString("  ")
			sb.WriteString(sugSelDescStyle.Render(item.Description))
		} else {
			sb.WriteString("    ")
			sb.WriteString(sugCmdStyle.Render(item.Display))
			sb.WriteString("  ")
			sb.WriteString(sugDescStyle.Render(item.Description))
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
	if t.suggestionKind != suggestionKindNone {
		h += maxVisibleSuggestions
	}
	return h
}

// IsSlashMode returns true if showing slash suggestions.
func (t TextInput) IsSlashMode() bool {
	return t.showSuggestions && t.suggestionKind == suggestionKindSlash
}

// ClearSlashMode exits the slash suggestion reserved area.
func (t TextInput) ClearSlashMode() TextInput {
	if t.suggestionKind == suggestionKindSlash {
		return t.clearSuggestions()
	}
	return t
}

// HasSuggestions returns true if there are visible suggestion candidates.
func (t TextInput) HasSuggestions() bool {
	return t.showSuggestions && len(t.suggestionItems) > 0
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
	if len(t.suggestionItems) == 0 {
		t.suggestionOffset = 0
		return
	}

	if t.selectedIdx < 0 {
		t.selectedIdx = 0
	}
	if t.selectedIdx >= len(t.suggestionItems) {
		t.selectedIdx = len(t.suggestionItems) - 1
	}

	if t.selectedIdx < t.suggestionOffset {
		t.suggestionOffset = t.selectedIdx
	}
	if t.selectedIdx >= t.suggestionOffset+maxVisibleSuggestions {
		t.suggestionOffset = t.selectedIdx - maxVisibleSuggestions + 1
	}

	maxOffset := len(t.suggestionItems) - maxVisibleSuggestions
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

func (t TextInput) clearSuggestions() TextInput {
	t.showSuggestions = false
	t.suggestionKind = suggestionKindNone
	t.suggestionItems = nil
	t.selectedIdx = 0
	t.suggestionOffset = 0
	t.activeToken = tokenRange{}
	return t
}

func (t *TextInput) setSuggestions(kind suggestionKind, span tokenRange, items []suggestionItem) {
	if len(items) == 0 {
		*t = t.clearSuggestions()
		return
	}
	t.showSuggestions = true
	t.suggestionKind = kind
	t.suggestionItems = items
	t.activeToken = span
	if t.selectedIdx >= len(items) {
		t.selectedIdx = 0
	}
	t.syncSuggestionWindow()
}

func (t TextInput) applySuggestion(item suggestionItem) TextInput {
	replacement := item.Value + " "
	switch item.Kind {
	case suggestionKindFile:
		replacement = "@" + item.Value + " "
	}

	t.Model.SetValue(replaceRunesInRange(t.Model.Value(), t.activeToken, replacement))
	t.Model.SetCursor(t.activeToken.start + len([]rune(replacement)))
	t.syncHeight()
	t.maskedPasteRaw = ""
	t.maskedPasteLabel = ""
	t = t.clearSuggestions()
	return t
}

func (t TextInput) slashSuggestionItems(token string, span tokenRange) ([]suggestionItem, bool) {
	if span.start != 0 || !strings.HasPrefix(token, "/") {
		return nil, false
	}

	names := t.slashRegistry.Suggestions(token)
	items := make([]suggestionItem, 0, len(names))
	for _, name := range names {
		cmd, ok := t.slashRegistry.Get(name)
		if !ok {
			continue
		}
		items = append(items, suggestionItem{
			Value:       name,
			Display:     name,
			Description: cmd.Description,
			Kind:        suggestionKindSlash,
		})
	}
	return items, true
}

func (t TextInput) fileSuggestionItems(token string, span tokenRange) ([]suggestionItem, bool) {
	if !isFileSuggestionToken(token) {
		return nil, false
	}
	return t.fileSuggestion.suggestions(token[1:]), true
}

func (t TextInput) currentToken() (string, tokenRange, bool) {
	value := t.Model.Value()
	runes := []rune(value)
	cursor := t.absoluteCursorPosition()
	if cursor < 0 {
		cursor = 0
	}
	if cursor > len(runes) {
		cursor = len(runes)
	}

	start := cursor
	for start > 0 && !unicode.IsSpace(runes[start-1]) {
		start--
	}
	end := cursor
	for end < len(runes) && !unicode.IsSpace(runes[end]) {
		end++
	}
	if start == end {
		return "", tokenRange{}, false
	}
	return string(runes[start:end]), tokenRange{start: start, end: end}, true
}

func (t TextInput) absoluteCursorPosition() int {
	row, col, lines := t.cursorPosition()
	offset := 0
	for i := 0; i < row && i < len(lines); i++ {
		offset += len([]rune(lines[i])) + 1
	}
	return offset + col
}

func replaceRunesInRange(value string, span tokenRange, replacement string) string {
	runes := []rune(value)
	prefix := string(runes[:span.start])
	suffix := string(runes[span.end:])
	if strings.HasSuffix(replacement, " ") {
		suffix = strings.TrimLeftFunc(suffix, unicode.IsSpace)
	}
	return prefix + replacement + suffix
}

func isFileSuggestionToken(token string) bool {
	switch {
	case token == "":
		return false
	case strings.HasPrefix(token, "@@"):
		return false
	case !strings.HasPrefix(token, "@") || len(token) == 1:
		return false
	}
	for _, r := range token[1:] {
		if !isAllowedFileSuggestionRune(r) {
			return false
		}
	}
	return true
}

func isAllowedFileSuggestionRune(r rune) bool {
	switch {
	case r >= 'a' && r <= 'z':
		return true
	case r >= 'A' && r <= 'Z':
		return true
	case r >= '0' && r <= '9':
		return true
	}
	switch r {
	case '.', '_', '/', '\\', '-':
		return true
	default:
		return false
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
