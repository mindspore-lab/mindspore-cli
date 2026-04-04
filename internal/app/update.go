package app

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/mindspore-cli/internal/update"
	"github.com/vigo999/mindspore-cli/internal/version"
)

var updatePromptSelectedStyle = lipgloss.NewStyle().
	Foreground(lipgloss.Color("252")).
	Background(lipgloss.Color("237")).
	Bold(true)

// updateChoice is the result of the update prompt.
type updateChoice int

const (
	updateChoiceUpdate updateChoice = iota
	updateChoiceSkip
)

// updatePrompt is a mini Bubble Tea model for the pre-TUI update screen.
type updatePrompt struct {
	result         *update.CheckResult
	program        *tea.Program
	cursor         int
	options        []string
	chosen         bool
	choice         updateChoice
	message        string // status message after selection
	quitting       bool
	downloaded     int64
	total          int64
	formattedNotes string // pre-computed from ReleaseNotes
}

func newUpdatePrompt(result *update.CheckResult) updatePrompt {
	options := []string{"Update now", "Skip this time"}
	if result.ForceUpdate {
		options = []string{"Update now"}
	}
	return updatePrompt{
		result:         result,
		options:        options,
		formattedNotes: formatReleaseNotes(result.ReleaseNotes),
	}
}

func (m *updatePrompt) Init() tea.Cmd { return nil }

func (m *updatePrompt) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	if m.quitting {
		return m, nil
	}

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			if m.result.ForceUpdate {
				// Can't quit on forced update, must update
				return m, nil
			}
			m.choice = updateChoiceSkip
			m.quitting = true
			return m, tea.Quit

		case "up", "k":
			if m.cursor > 0 {
				m.cursor--
			}
		case "down", "j":
			if m.cursor < len(m.options)-1 {
				m.cursor++
			}
		case "enter":
			m.chosen = true
			if m.cursor == 0 {
				m.choice = updateChoiceUpdate
				m.message = fmt.Sprintf("  Downloading mscli %s...", m.result.LatestVersion)
				return m, doUpdate(m.result, m.program)
			}
			// Skip (only reachable for non-forced)
			m.choice = updateChoiceSkip
			m.quitting = true
			return m, tea.Quit
		}

	case downloadProgressMsg:
		m.downloaded = msg.downloaded
		m.total = msg.total
		return m, nil

	case updateDoneMsg:
		m.quitting = true
		if msg.err != nil {
			m.message = fmt.Sprintf("  %v\n  Continuing with current version...", msg.err)
			m.choice = updateChoiceSkip
		} else {
			m.message = fmt.Sprintf("  Updated to %s. Please restart mscli.", m.result.LatestVersion)
			m.choice = updateChoiceUpdate
		}
		return m, tea.Quit
	}

	return m, nil
}

func (m *updatePrompt) View() string {
	var b strings.Builder

	b.WriteString("\n")
	if m.result.ForceUpdate {
		b.WriteString(fmt.Sprintf("  Required update: %s → %s\n", m.result.CurrentVersion, m.result.LatestVersion))
	} else {
		b.WriteString(fmt.Sprintf("  Update available: %s → %s\n", m.result.CurrentVersion, m.result.LatestVersion))
	}
	b.WriteString("\n")

	if m.formattedNotes != "" {
		b.WriteString("  What's new:\n")
		b.WriteString(m.formattedNotes)
		b.WriteString("\n")
	}

	if m.chosen {
		b.WriteString(m.message)
		b.WriteString("\n")
		if m.downloaded > 0 {
			b.WriteString(renderProgressBar(m.downloaded, m.total))
			b.WriteString("\n")
		}
		return b.String()
	}

	for i, opt := range m.options {
		if i == m.cursor {
			b.WriteString(updatePromptSelectedStyle.Render(fmt.Sprintf("  > %s", opt)))
			b.WriteString("\n")
		} else {
			b.WriteString(fmt.Sprintf("    %s\n", opt))
		}
	}
	b.WriteString("\n  Use ↑/↓ to select, Enter to confirm\n")

	return b.String()
}

// formatReleaseNotes strips markdown headers and formats notes for display.
func formatReleaseNotes(notes string) string {
	if notes == "" {
		return ""
	}
	var b strings.Builder
	for _, line := range strings.Split(strings.TrimSpace(notes), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// Strip markdown headers (e.g. "## What's new" → "What's new")
		if strings.HasPrefix(line, "#") {
			line = strings.TrimLeft(line, "#")
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
		}
		b.WriteString("  " + line + "\n")
	}
	return b.String()
}

// renderProgressBar renders a 30-char wide progress bar.
func renderProgressBar(downloaded, total int64) string {
	const barWidth = 30
	if total <= 0 {
		return fmt.Sprintf("  Downloading... (%.1f MB)", float64(downloaded)/1e6)
	}

	pct := float64(downloaded) / float64(total)
	if pct > 1 {
		pct = 1
	}
	filled := int(pct * barWidth)
	empty := barWidth - filled

	return fmt.Sprintf("  [%s%s] %d%% (%.1f/%.1f MB)",
		strings.Repeat("█", filled),
		strings.Repeat("░", empty),
		int(pct*100),
		float64(downloaded)/1e6,
		float64(total)/1e6,
	)
}

// downloadProgressMsg carries download progress updates.
type downloadProgressMsg struct {
	downloaded int64
	total      int64
}

// updateDoneMsg is sent when download+install completes.
type updateDoneMsg struct{ err error }

func doUpdate(result *update.CheckResult, p *tea.Program) tea.Cmd {
	return func() tea.Msg {
		progressFn := func(downloaded, total int64) {
			p.Send(downloadProgressMsg{downloaded: downloaded, total: total})
		}
		tmpPath, err := update.DownloadWithProgress(context.Background(), result.DownloadURL, progressFn)
		if err != nil {
			return updateDoneMsg{fmt.Errorf("download failed: %w", err)}
		}
		defer os.Remove(tmpPath)

		if err := update.Install(tmpPath); err != nil {
			return updateDoneMsg{fmt.Errorf("install failed: %w", err)}
		}
		return updateDoneMsg{}
	}
}

// checkAndPromptUpdate checks for updates before the TUI launches.
// Returns true if the caller should exit (user updated successfully).
func checkAndPromptUpdate() bool {
	if version.Version == "dev" || version.Version == "" {
		return false
	}

	result, err := update.Check(context.Background(), version.Version)
	if err != nil || result == nil || !result.UpdateAvailable {
		return false
	}

	result.ReleaseNotes = update.FetchReleaseNotes(context.Background(), result.LatestVersion)
	prompt := newUpdatePrompt(result)
	p := tea.NewProgram(&prompt)
	prompt.program = p
	finalModel, err := p.Run()
	if err != nil {
		return false
	}

	final := finalModel.(*updatePrompt)
	fmt.Println()
	return final.choice == updateChoiceUpdate && final.quitting
}

// cleanUpdateTmp removes leftover temp files from previous update attempts.
func cleanUpdateTmp() {
	tmpDir := update.ConfigDir() + "/tmp"
	entries, err := os.ReadDir(tmpDir)
	if err != nil {
		return
	}
	now := time.Now()
	for _, e := range entries {
		if !strings.HasPrefix(e.Name(), "mscli-update-") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		if now.Sub(info.ModTime()) > 24*time.Hour {
			os.Remove(tmpDir + "/" + e.Name())
		}
	}
}
