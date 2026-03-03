package context

// Budget controls context token usage.
type Budget struct {
	MaxTokens int
}

const (
	defaultMaxTokens = 12000
)

func (b Budget) Limit() int {
	if b.MaxTokens <= 0 {
		return defaultMaxTokens
	}
	return b.MaxTokens
}

// ApproxTokens provides a lightweight token estimate for prompt budgeting.
func ApproxTokens(text string) int {
	if text == "" {
		return 0
	}
	// A pragmatic approximation for mixed English/CJK text.
	return len([]rune(text))/4 + 1
}
