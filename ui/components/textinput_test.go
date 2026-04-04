package components

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/mindspore-lab/mindspore-cli/ui/slash"
)

const largePastedBlock = "line 01\nline 02\nline 03\nline 04\nline 05\nline 06\nline 07\nline 08\n"

func TestTextInputHistoryRecall(t *testing.T) {
	input := NewTextInput()
	input = input.PushHistory("first prompt")
	input = input.PushHistory("second prompt")
	input.Model.SetValue("draft")
	input.Model.SetCursor(len("draft"))

	input = input.PrevHistory()
	if got := input.Value(); got != "second prompt" {
		t.Fatalf("expected latest history entry, got %q", got)
	}

	input = input.PrevHistory()
	if got := input.Value(); got != "first prompt" {
		t.Fatalf("expected previous history entry, got %q", got)
	}

	input = input.NextHistory()
	if got := input.Value(); got != "second prompt" {
		t.Fatalf("expected forward history entry, got %q", got)
	}

	input = input.NextHistory()
	if got := input.Value(); got != "draft" {
		t.Fatalf("expected draft restoration after leaving history, got %q", got)
	}
}

func TestTextInputSeedHistoryPreservesRecallOrder(t *testing.T) {
	input := NewTextInput()
	input = input.SeedHistory([]string{"older prompt", "newer old prompt"})
	input = input.PushHistory("current prompt")

	input = input.PrevHistory()
	if got := input.Value(); got != "current prompt" {
		t.Fatalf("expected latest current-session prompt first, got %q", got)
	}

	input = input.PrevHistory()
	if got := input.Value(); got != "newer old prompt" {
		t.Fatalf("expected newest seeded old-history next, got %q", got)
	}

	input = input.PrevHistory()
	if got := input.Value(); got != "older prompt" {
		t.Fatalf("expected oldest seeded prompt last, got %q", got)
	}
}

func TestTextInputHistoryDoesNotBreakSlashSuggestions(t *testing.T) {
	input := NewTextInput()
	input = input.PushHistory("/project")
	var cmd tea.Cmd
	input, cmd = input.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'/'}})
	_ = cmd
	if !input.IsSlashMode() {
		t.Fatal("expected slash suggestions after typing slash")
	}
}

func TestTextInputCtrlJInsertsNewline(t *testing.T) {
	input := NewTextInput()
	if got := input.Height(); got != 3 {
		t.Fatalf("expected single-row composer block height 3, got %d", got)
	}

	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyCtrlJ})
	if got := input.Value(); got != "\n" {
		t.Fatalf("expected ctrl+j to insert newline, got %q", got)
	}
	if got := input.Height(); got != 4 {
		t.Fatalf("expected two-row composer block height 4, got %d", got)
	}

	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyCtrlJ})

	if got := input.Value(); got != "\n\n" {
		t.Fatalf("expected second ctrl+j to insert another newline, got %q", got)
	}
	if got := input.Height(); got != 5 {
		t.Fatalf("expected three-row composer block height 5, got %d", got)
	}
}

func TestTextInputConsumeEscapedEnterAtEnd(t *testing.T) {
	input := NewTextInput()
	input.Model.SetValue("alpha\\")
	input.Model.SetCursor(len([]rune("alpha\\")))

	input, consumed := input.ConsumeEscapedEnter()
	if !consumed {
		t.Fatal("expected escaped enter to be consumed")
	}
	if got := input.Value(); got != "alpha\n" {
		t.Fatalf("expected escaped enter to replace trailing backslash with newline, got %q", got)
	}
	if got := input.Height(); got != 4 {
		t.Fatalf("expected escaped enter to grow composer height, got %d", got)
	}
}

func TestTextInputConsumeEscapedEnterAtCursorInMiddle(t *testing.T) {
	input := NewTextInput()
	input.Model.SetValue("ab\\cd")
	input.Model.SetCursor(3)

	input, consumed := input.ConsumeEscapedEnter()
	if !consumed {
		t.Fatal("expected escaped enter to be consumed in the middle of the input")
	}
	if got := input.Value(); got != "ab\ncd" {
		t.Fatalf("expected escaped enter to preserve surrounding text, got %q", got)
	}
	row, col, _ := input.cursorPosition()
	if row != 1 || col != 0 {
		t.Fatalf("expected cursor to move to the start of the new line, got row=%d col=%d", row, col)
	}
}

func TestTextInputConsumeEscapedEnterKeepsCursorOnInsertedLineInMultilineDraft(t *testing.T) {
	input := NewTextInput()
	input.Model.SetValue("aaa\\\nbbb\nccc")
	input.Model.SetCursor(0)
	input.Model.CursorUp()
	input.Model.CursorUp()
	input.Model.SetCursor(len([]rune("aaa\\")))

	input, consumed := input.ConsumeEscapedEnter()
	if !consumed {
		t.Fatal("expected escaped enter to be consumed in multiline draft")
	}
	if got := input.Value(); got != "aaa\n\nbbb\nccc" {
		t.Fatalf("expected multiline draft to gain a blank line, got %q", got)
	}

	row, col, _ := input.cursorPosition()
	if row != 1 || col != 0 {
		t.Fatalf("expected cursor on inserted blank line, got row=%d col=%d", row, col)
	}

	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("x")})
	if got := input.Value(); got != "aaa\nx\nbbb\nccc" {
		t.Fatalf("expected follow-up typing to stay on inserted line, got %q", got)
	}
}

func TestTextInputConsumeEscapedEnterConsumesOnlyImmediateBackslash(t *testing.T) {
	input := NewTextInput()
	input.Model.SetValue("\\\\")
	input.Model.SetCursor(2)

	input, consumed := input.ConsumeEscapedEnter()
	if !consumed {
		t.Fatal("expected escaped enter to consume the immediate preceding backslash")
	}
	if got := input.Value(); got != "\\\n" {
		t.Fatalf("expected earlier backslash to remain literal, got %q", got)
	}
}

func TestTextInputConsumeEscapedEnterReturnsFalseWithoutBackslash(t *testing.T) {
	input := NewTextInput()
	input.Model.SetValue("alpha")
	input.Model.SetCursor(len([]rune("alpha")))

	updated, consumed := input.ConsumeEscapedEnter()
	if consumed {
		t.Fatal("expected escaped enter to be ignored when no backslash precedes the cursor")
	}
	if got := updated.Value(); got != "alpha" {
		t.Fatalf("expected input to stay unchanged, got %q", got)
	}
}

func TestTextInputBackslashStaysLiteralWhenFollowedByOtherRunes(t *testing.T) {
	input := NewTextInput()

	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'\\'}})
	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("beta")})

	if got := input.Value(); got != "\\beta" {
		t.Fatalf("expected backslash to stay literal when next key is not enter, got %q", got)
	}
}

func TestTextInputConsumeEscapedEnterPreservesCollapsedPasteValue(t *testing.T) {
	input := NewTextInput()
	input = input.SetWidth(60)

	input, _ = input.Update(tea.KeyMsg{
		Type:  tea.KeyRunes,
		Runes: []rune(largePastedBlock),
		Paste: true,
	})
	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'\\'}})

	input, consumed := input.ConsumeEscapedEnter()
	if !consumed {
		t.Fatal("expected escaped enter to be consumed after collapsed paste summary")
	}
	if got := input.Value(); got != largePastedBlock+"\n" {
		t.Fatalf("expected reconstructed value to preserve raw paste and newline, got %q", got)
	}
}

func TestTextInputConsumeEscapedEnterCanRunWhenSlashModeHasNoSuggestions(t *testing.T) {
	input := newSlashSuggestionInput(3)
	input.Model.SetValue("/\\")
	input.Model.SetCursor(len([]rune("/\\")))
	input.showSuggestions = false
	input.suggestionItems = nil
	input.suggestionKind = suggestionKindSlash

	input, consumed := input.ConsumeEscapedEnter()
	if !consumed {
		t.Fatal("expected escaped enter to be consumed when slash mode has no suggestions")
	}
	if got := input.Value(); got != "/\n" {
		t.Fatalf("expected escaped enter to insert newline when no suggestions are available, got %q", got)
	}
}

func TestTextInputUsesSinglePromptWithContinuationLines(t *testing.T) {
	input := NewTextInput()
	input = input.SetWidth(40)
	input.Model.SetValue("one\ntwo\nthree")
	input.syncHeight()

	view := input.View()
	if got := strings.Count(view, composerPrompt); got != 1 {
		t.Fatalf("expected one primary prompt, got %d in view %q", got, view)
	}
	if got := strings.Count(view, composerContinue+"two"); got != 1 {
		t.Fatalf("expected continuation prompt for second line, got view %q", view)
	}
	if got := strings.Count(view, composerContinue+"three"); got != 1 {
		t.Fatalf("expected continuation prompt for third line, got view %q", view)
	}
}

func TestTextInputKeepsFirstLineVisibleAfterExplicitNewlineGrowth(t *testing.T) {
	input := NewTextInput()
	input = input.SetWidth(40)
	input.Model.SetValue("alpha")

	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyCtrlJ})
	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("beta")})

	view := input.View()
	if !strings.Contains(view, composerPrompt+"alpha") {
		t.Fatalf("expected first line to remain visible after newline growth, got %q", view)
	}
	if !strings.Contains(view, composerContinue+"beta") {
		t.Fatalf("expected second line to remain visible after newline growth, got %q", view)
	}
}

func TestTextInputHeightGrowsForSoftWrappedLine(t *testing.T) {
	input := NewTextInput()
	input = input.SetWidth(11)
	input.Model.SetValue("alpha beta")
	input.syncHeight()

	if got := input.Height(); got != 4 {
		t.Fatalf("expected wrapped two-row composer block height 4, got %d", got)
	}

	view := input.View()
	if !strings.Contains(view, composerPrompt+"alpha ") {
		t.Fatalf("expected first wrapped row in view, got %q", view)
	}
	if !strings.Contains(view, composerContinue+"beta") {
		t.Fatalf("expected second wrapped row in view, got %q", view)
	}
}

func TestTextInputKeepsFirstWrappedLineVisibleAfterSoftWrapGrowth(t *testing.T) {
	input := NewTextInput()
	input = input.SetWidth(11)

	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("alpha beta")})

	view := input.View()
	if !strings.Contains(view, composerPrompt+"alpha ") {
		t.Fatalf("expected first wrapped row to remain visible after typing growth, got %q", view)
	}
	if !strings.Contains(view, composerContinue+"beta") {
		t.Fatalf("expected second wrapped row after typing growth, got %q", view)
	}
}

func TestTextInputPasteShowsFullContent(t *testing.T) {
	input := NewTextInput()
	input = input.SetWidth(60)

	input, _ = input.Update(tea.KeyMsg{
		Type:  tea.KeyRunes,
		Runes: []rune(largePastedBlock),
		Paste: true,
	})

	if got := input.Value(); got != largePastedBlock {
		t.Fatalf("expected pasted content to be stored verbatim, got %q", got)
	}

	view := input.View()
	if !strings.Contains(view, "line 01") {
		t.Fatalf("expected pasted content to stay visible in view, got %q", view)
	}
	if !strings.Contains(view, "line 07") {
		t.Fatalf("expected later pasted lines to stay visible in view, got %q", view)
	}
}

func TestTextInputPasteKeepsAllRowsVisibleAfterGrowth(t *testing.T) {
	input := NewTextInput()
	input = input.SetWidth(24)

	input, _ = input.Update(tea.KeyMsg{
		Type:  tea.KeyRunes,
		Runes: []rune("line 01\nline 02\nline 03\nline 04"),
		Paste: true,
	})

	view := input.View()
	for _, want := range []string{"line 01", "line 02", "line 03", "line 04"} {
		if !strings.Contains(view, want) {
			t.Fatalf("expected pasted line %q to stay visible after paste growth, got %q", want, view)
		}
	}
}

// Regression: pasting after a prior render frame must not hide the first
// lines.  The textarea's internal viewport could retain a stale scroll
// offset from the previous View() → SetContent() cycle, causing
// repositionView() inside textarea.Update to scroll past the top lines.
func TestTextInputPasteAfterPriorRenderShowsAllLines(t *testing.T) {
	input := NewTextInput()
	input = input.SetWidth(80)

	// Simulate a render frame BEFORE the paste (the app calls View() every
	// frame, populating the textarea's internal viewport with old content).
	_ = input.View()

	input, _ = input.Update(tea.KeyMsg{
		Type:  tea.KeyRunes,
		Runes: []rune("The first line.\nThe second line.\nThe third line.\nThe fourth line."),
		Paste: true,
	})

	// Simulate resizeActiveLayout → resizeInput that the app does after
	// every key event.
	input = input.SetWidth(80)

	view := input.View()
	for _, want := range []string{"first line", "second line", "third line", "fourth line"} {
		if !strings.Contains(view, want) {
			t.Fatalf("expected %q visible after paste (prior render), got view:\n%s", want, view)
		}
	}
}

// Pasting two lines must show both — the minimal reproduction of the
// reported bug where only the last line was visible.
func TestTextInputPasteTwoLinesShowsBoth(t *testing.T) {
	input := NewTextInput()
	input = input.SetWidth(60)
	_ = input.View()

	input, _ = input.Update(tea.KeyMsg{
		Type:  tea.KeyRunes,
		Runes: []rune("Line A\nLine B"),
		Paste: true,
	})
	input = input.SetWidth(60)

	view := input.View()
	if !strings.Contains(view, "Line A") {
		t.Fatalf("first pasted line missing:\n%s", view)
	}
	if !strings.Contains(view, "Line B") {
		t.Fatalf("second pasted line missing:\n%s", view)
	}
}

// After paste the cursor must be at the end of the pasted text (industry
// standard behaviour matching Claude Code / Codex CLI).
func TestTextInputPasteCursorAtEnd(t *testing.T) {
	input := NewTextInput()
	input = input.SetWidth(60)

	input, _ = input.Update(tea.KeyMsg{
		Type:  tea.KeyRunes,
		Runes: []rune("alpha\nbeta\ngamma"),
		Paste: true,
	})

	if !input.atInputEnd() {
		t.Fatal("expected cursor at the end of pasted text")
	}
}

// Large paste (many lines) must store all content and render at least the
// first and last lines in the view.
func TestTextInputLargePasteShowsContent(t *testing.T) {
	input := NewTextInput()
	input = input.SetWidth(60)
	_ = input.View()

	var lines []string
	for i := 0; i < 50; i++ {
		lines = append(lines, fmt.Sprintf("line %03d content", i))
	}
	payload := strings.Join(lines, "\n")

	input, _ = input.Update(tea.KeyMsg{
		Type:  tea.KeyRunes,
		Runes: []rune(payload),
		Paste: true,
	})
	input = input.SetWidth(60)

	if got := input.Value(); got != payload {
		t.Fatalf("not all lines stored; got %d runes, want %d", len(got), len(payload))
	}

	view := input.View()
	if !strings.Contains(view, "line 000") {
		t.Fatalf("first line not visible in large paste:\n%s", view)
	}
	if !strings.Contains(view, "line 049") {
		t.Fatalf("last line not visible in large paste:\n%s", view)
	}
}

func TestTextInputHistoryRecallOfSlashCommandDoesNotReopenSuggestions(t *testing.T) {
	input := NewTextInput()
	input = input.PushHistory("/project")
	input = input.PushHistory("hello")

	input = input.PrevHistory()
	if got := input.Value(); got != "hello" {
		t.Fatalf("expected latest history entry, got %q", got)
	}

	input = input.PrevHistory()
	if got := input.Value(); got != "/project" {
		t.Fatalf("expected slash command from history, got %q", got)
	}
	if input.IsSlashMode() {
		t.Fatal("expected slash suggestions to stay closed while browsing history")
	}

	input = input.NextHistory()
	if got := input.Value(); got != "hello" {
		t.Fatalf("expected down to continue history recall even in slash mode, got %q", got)
	}
}

func TestTextInputHistoryOnlyTriggersAtEditorBoundaries(t *testing.T) {
	input := NewTextInput()
	input.Model.SetValue("first line\nsecond line")
	input.syncHeight()

	if input.CanNavigateHistory("up") {
		t.Fatal("expected up to stay inside multiline editor while on the last line")
	}

	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyUp})

	if !input.CanNavigateHistory("up") {
		t.Fatal("expected up history at the top boundary of the editor")
	}
	if input.CanNavigateHistory("down") {
		t.Fatal("expected down history to stay disabled away from the bottom boundary")
	}

	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyDown})

	if !input.CanNavigateHistory("down") {
		t.Fatal("expected down history at the bottom boundary of multiline input")
	}
}

func TestTextInputSuggestionsScrollDownToKeepSelectionVisible(t *testing.T) {
	input := newSlashSuggestionInput(10)
	input.selectedIdx = 7
	input.suggestionOffset = 0

	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyDown})

	if input.selectedIdx != 8 {
		t.Fatalf("expected selected index 8, got %d", input.selectedIdx)
	}
	if input.suggestionOffset != 1 {
		t.Fatalf("expected suggestion offset 1, got %d", input.suggestionOffset)
	}

	view := input.View()
	if !strings.Contains(view, "/cmd08") {
		t.Fatalf("expected view to include newly selected command, got %q", view)
	}
	if strings.Contains(view, "/cmd00") {
		t.Fatalf("expected first command to scroll out of view, got %q", view)
	}
}

func TestTextInputSuggestionsScrollUpToKeepSelectionVisible(t *testing.T) {
	input := newSlashSuggestionInput(10)
	input.selectedIdx = 2
	input.suggestionOffset = 2

	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyUp})

	if input.selectedIdx != 1 {
		t.Fatalf("expected selected index 1, got %d", input.selectedIdx)
	}
	if input.suggestionOffset != 1 {
		t.Fatalf("expected suggestion offset 1, got %d", input.suggestionOffset)
	}

	view := input.View()
	if !strings.Contains(view, "/cmd01") {
		t.Fatalf("expected view to include newly selected command, got %q", view)
	}
	if strings.Contains(view, "/cmd09") {
		t.Fatalf("expected last command to scroll out of view, got %q", view)
	}
}

func TestTextInputSuggestionsWrapDownToTopPage(t *testing.T) {
	input := newSlashSuggestionInput(10)
	input.selectedIdx = 9
	input.suggestionOffset = 2

	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyDown})

	if input.selectedIdx != 0 {
		t.Fatalf("expected selected index 0 after wrap, got %d", input.selectedIdx)
	}
	if input.suggestionOffset != 0 {
		t.Fatalf("expected suggestion offset 0 after wrap, got %d", input.suggestionOffset)
	}

	view := input.View()
	if !strings.Contains(view, "/cmd00") {
		t.Fatalf("expected wrapped view to include first command, got %q", view)
	}
	if strings.Contains(view, "/cmd09") {
		t.Fatalf("expected last command to leave view after wrapping to top, got %q", view)
	}
}

func TestTextInputSuggestionsWrapUpToLastPage(t *testing.T) {
	input := newSlashSuggestionInput(10)
	input.selectedIdx = 0
	input.suggestionOffset = 0

	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyUp})

	if input.selectedIdx != 9 {
		t.Fatalf("expected selected index 9 after wrap, got %d", input.selectedIdx)
	}
	if input.suggestionOffset != 2 {
		t.Fatalf("expected suggestion offset 2 after wrap, got %d", input.suggestionOffset)
	}

	view := input.View()
	if !strings.Contains(view, "/cmd09") {
		t.Fatalf("expected wrapped view to include last command, got %q", view)
	}
	if strings.Contains(view, "/cmd00") {
		t.Fatalf("expected first command to leave view after wrapping to bottom, got %q", view)
	}
}

func TestTextInputShowsFileSuggestionsForAtToken(t *testing.T) {
	root := t.TempDir()
	writeSuggestionFile(t, root, "ctx.txt")
	writeSuggestionFile(t, root, "sub/first.md")
	writeSuggestionFile(t, root, "sub/final.md")

	input := NewTextInput().WithFileSuggestions(root)
	input.Model.SetValue("read @sub/f")
	input.Model.SetCursor(len("read @sub/f"))
	input.updateSuggestions()

	if !input.HasSuggestions() {
		t.Fatal("expected file suggestions for @ token")
	}
	if input.IsSlashMode() {
		t.Fatal("expected @file suggestions not to report slash mode")
	}
	if len(input.suggestionItems) != 2 {
		t.Fatalf("expected 2 matching file suggestions, got %d", len(input.suggestionItems))
	}
	if got := input.suggestionItems[0].Value; got != "sub/final.md" {
		t.Fatalf("expected lexicographic file suggestion, got %q", got)
	}
}

func TestTextInputEnterAcceptsFileSuggestionWithoutSubmitting(t *testing.T) {
	root := t.TempDir()
	writeSuggestionFile(t, root, "ctx.txt")

	input := NewTextInput().WithFileSuggestions(root)
	input.Model.SetValue("read @ct")
	input.Model.SetCursor(len("read @ct"))
	input.updateSuggestions()

	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if got := input.Value(); got != "read @ctx.txt " {
		t.Fatalf("expected enter to accept file suggestion, got %q", got)
	}
	if input.HasSuggestions() {
		t.Fatal("expected suggestions cleared after accepting file suggestion")
	}
}

func TestTextInputReplacesOnlyCurrentTokenWhenApplyingFileSuggestion(t *testing.T) {
	root := t.TempDir()
	writeSuggestionFile(t, root, "ctx.txt")

	input := NewTextInput().WithFileSuggestions(root)
	input.Model.SetValue("before @ct after")
	input.Model.SetCursor(len("before @ct"))
	input.updateSuggestions()

	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyTab})
	if got := input.Value(); got != "before @ctx.txt after" {
		t.Fatalf("expected current token replacement only, got %q", got)
	}
}

func TestTextInputLeavesInvalidAtFormsWithoutSuggestions(t *testing.T) {
	root := t.TempDir()
	writeSuggestionFile(t, root, "ctx.txt")

	cases := []string{
		"read @@ctx",
		"read @ctx.txt,",
		"read (@ctx.txt)",
	}

	for _, tc := range cases {
		input := NewTextInput().WithFileSuggestions(root)
		input.Model.SetValue(tc)
		input.Model.SetCursor(len(tc))
		input.updateSuggestions()
		if input.HasSuggestions() {
			t.Fatalf("expected no suggestions for %q", tc)
		}
	}
}

func TestTextInputReplacesCurrentTokenOnSecondLine(t *testing.T) {
	root := t.TempDir()
	writeSuggestionFile(t, root, "ctx.txt")

	input := NewTextInput().WithFileSuggestions(root)
	input.Model.SetValue("line one\nread @ct")
	input.Model.SetCursor(len("line one\nread @ct"))
	input.updateSuggestions()

	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyTab})
	if got := input.Value(); got != "line one\nread @ctx.txt " {
		t.Fatalf("expected second-line token replacement, got %q", got)
	}
}

func TestTextInputPrefersFileSuggestionsInSlashCommandArguments(t *testing.T) {
	root := t.TempDir()
	writeSuggestionFile(t, root, "ctx.txt")

	input := NewTextInput().WithFileSuggestions(root)
	input.Model.SetValue("/report accuracy @ct")
	input.Model.SetCursor(len("/report accuracy @ct"))
	input.updateSuggestions()

	if !input.HasSuggestions() {
		t.Fatal("expected file suggestions in slash command arguments")
	}
	if input.IsSlashMode() {
		t.Fatal("expected file suggestions to override slash suggestions outside the command token")
	}
}

func newSlashSuggestionInput(count int) TextInput {
	input := NewTextInput()
	registry := slash.NewRegistry()
	input.slashRegistry = registry
	input.showSuggestions = true
	input.suggestionKind = suggestionKindSlash
	input.suggestionItems = make([]suggestionItem, 0, count)

	for i := 0; i < count; i++ {
		name := fmt.Sprintf("/cmd%02d", i)
		registry.Register(slash.Command{
			Name:        name,
			Description: fmt.Sprintf("Command %02d", i),
			Usage:       name,
		})
		input.suggestionItems = append(input.suggestionItems, suggestionItem{
			Value:       name,
			Display:     name,
			Description: fmt.Sprintf("Command %02d", i),
			Kind:        suggestionKindSlash,
		})
	}

	return input
}

func writeSuggestionFile(t *testing.T, root, relative string) {
	t.Helper()
	path := filepath.Join(root, relative)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(relative), 0o644); err != nil {
		t.Fatal(err)
	}
}
