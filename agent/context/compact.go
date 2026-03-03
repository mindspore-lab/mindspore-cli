package context

import "strings"

const defaultCompactionRatio = 0.8

// Compact keeps backward compatibility for old call-sites.
func Compact(input string) string { return input }

// CompactWithBudget trims context when estimated tokens exceed maxTokens.
func CompactWithBudget(input string, maxTokens int, keepRatio float64) string {
	trimmed := strings.TrimSpace(input)
	if trimmed == "" {
		return ""
	}
	if maxTokens <= 0 || ApproxTokens(trimmed) <= maxTokens {
		return trimmed
	}
	if keepRatio <= 0 || keepRatio >= 1 {
		keepRatio = defaultCompactionRatio
	}

	runes := []rune(trimmed)
	maxRunes := maxTokens * 4
	if maxRunes < 80 {
		maxRunes = 80
	}
	if len(runes) <= maxRunes {
		return trimmed
	}

	headRunes := int(float64(maxRunes) * (1 - keepRatio))
	if headRunes < 24 {
		headRunes = 24
	}
	tailRunes := maxRunes - headRunes
	if tailRunes < 48 {
		tailRunes = 48
		if tailRunes+headRunes > maxRunes {
			headRunes = maxRunes - tailRunes
			if headRunes < 8 {
				headRunes = 8
			}
		}
	}

	head := string(runes[:headRunes])
	tail := string(runes[len(runes)-tailRunes:])
	return strings.TrimSpace(head) + "\n...[context compacted]...\n" + strings.TrimSpace(tail)
}
