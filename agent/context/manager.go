package context

import (
	"fmt"
	"strings"
)

type Entry struct {
	Kind    string
	Content string
}

// Manager assembles model context with lightweight compaction.
type Manager struct {
	budget          Budget
	compactionRatio float64
	maxEntries      int
	entries         []Entry
}

func NewManager(maxTokens int, compactionRatio float64, maxEntries int) *Manager {
	if maxEntries <= 0 {
		maxEntries = 80
	}
	if compactionRatio <= 0 || compactionRatio >= 1 {
		compactionRatio = defaultCompactionRatio
	}
	return &Manager{
		budget:          Budget{MaxTokens: maxTokens},
		compactionRatio: compactionRatio,
		maxEntries:      maxEntries,
		entries:         make([]Entry, 0, maxEntries),
	}
}

func (m *Manager) Add(kind, content string) {
	k := strings.ToUpper(strings.TrimSpace(kind))
	c := strings.TrimSpace(content)
	if c == "" {
		return
	}
	if len(m.entries) > 0 {
		last := m.entries[len(m.entries)-1]
		if last.Kind == k && last.Content == c {
			return
		}
	}

	m.entries = append(m.entries, Entry{
		Kind:    k,
		Content: c,
	})
	if len(m.entries) > m.maxEntries {
		m.entries = append([]Entry{}, m.entries[len(m.entries)-m.maxEntries:]...)
	}
}

func (m *Manager) LastN(n int) []Entry {
	if n <= 0 || len(m.entries) == 0 {
		return nil
	}
	if n >= len(m.entries) {
		out := make([]Entry, len(m.entries))
		copy(out, m.entries)
		return out
	}
	out := make([]Entry, n)
	copy(out, m.entries[len(m.entries)-n:])
	return out
}

func (m *Manager) Render() string {
	if len(m.entries) == 0 {
		return "(none)"
	}

	lines := make([]string, 0, len(m.entries))
	for i, e := range m.entries {
		lines = append(lines, fmt.Sprintf("%d. [%s] %s", i+1, e.Kind, e.Content))
	}
	raw := strings.Join(lines, "\n")
	return CompactWithBudget(raw, m.budget.Limit(), m.compactionRatio)
}
