package app

import (
	"path/filepath"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/mindspore-lab/mindspore-cli/internal/workspacefile"
)

var atFilePathPattern = regexp.MustCompile(`^[A-Za-z0-9._/\\-]+$`)

func (a *Application) expandInputText(text string) (string, error) {
	return expandAtFiles(a.WorkDir, text)
}

func expandAtFiles(workDir, text string) (string, error) {
	var out strings.Builder

	for i := 0; i < len(text); {
		r := rune(text[i])
		if r < utf8RuneSelf && !isASCIIWhitespace(byte(r)) {
			j := i + 1
			for j < len(text) && !isASCIIWhitespace(text[j]) {
				j++
			}
			token := text[i:j]
			replaced, err := replaceAtFileToken(workDir, token)
			if err != nil {
				return "", err
			}
			out.WriteString(replaced)
			i = j
			continue
		}

		runeValue, size := utf8.DecodeRuneInString(text[i:])
		if !unicode.IsSpace(runeValue) {
			j := i + size
			for j < len(text) {
				nextRune, nextSize := utf8.DecodeRuneInString(text[j:])
				if unicode.IsSpace(nextRune) {
					break
				}
				j += nextSize
			}
			token := text[i:j]
			replaced, err := replaceAtFileToken(workDir, token)
			if err != nil {
				return "", err
			}
			out.WriteString(replaced)
			i = j
			continue
		}

		out.WriteRune(runeValue)
		i += size
	}

	return out.String(), nil
}

func replaceAtFileToken(workDir, token string) (string, error) {
	switch {
	case token == "":
		return token, nil
	case strings.HasPrefix(token, "@@"):
		return token[1:], nil
	case !strings.HasPrefix(token, "@") || len(token) == 1:
		return token, nil
	}

	path := token[1:]
	if !atFilePathPattern.MatchString(path) {
		return token, nil
	}

	fullPath, err := workspacefile.ResolveExistingFilePath(workDir, path)
	if err != nil {
		return "", err
	}

	return formatExpandedFilePath(fullPath), nil
}

func formatExpandedFilePath(path string) string {
	return `[file path="` + filepath.ToSlash(filepath.Clean(path)) + `"]`
}

type rawCommand struct {
	Name      string
	Remainder string
}

func splitRawCommand(input string) (rawCommand, bool) {
	trimmed := strings.TrimSpace(input)
	if trimmed == "" || !strings.HasPrefix(trimmed, "/") {
		return rawCommand{}, false
	}

	if idx := strings.IndexAny(trimmed, " \t\r\n"); idx >= 0 {
		return rawCommand{
			Name:      trimmed[:idx],
			Remainder: strings.TrimSpace(trimmed[idx+1:]),
		}, true
	}

	return rawCommand{Name: trimmed}, true
}

func splitFirstToken(input string) (string, string) {
	trimmed := strings.TrimSpace(input)
	if trimmed == "" {
		return "", ""
	}

	for i, r := range trimmed {
		if unicode.IsSpace(r) {
			return trimmed[:i], strings.TrimSpace(trimmed[i:])
		}
	}

	return trimmed, ""
}

func (a *Application) expandIssueCommandInput(rawInput string) (string, error) {
	first, remainder := splitFirstToken(rawInput)
	if first == "" {
		return strings.TrimSpace(rawInput), nil
	}
	if !looksLikeIssueKey(first) {
		return a.expandInputText(strings.TrimSpace(rawInput))
	}
	if _, err := parseIssueRef(first); err != nil {
		return strings.TrimSpace(rawInput), nil
	}
	if remainder == "" {
		return first, nil
	}
	expanded, err := a.expandInputText(remainder)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(first + " " + expanded), nil
}

func (a *Application) expandReportInput(rawInput string) (string, error) {
	first, remainder := splitFirstToken(rawInput)
	if first == "" || remainder == "" {
		return strings.TrimSpace(rawInput), nil
	}
	expanded, err := a.expandInputText(remainder)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(first + " " + expanded), nil
}

func (a *Application) emitInputExpansionError(err error) {
	a.emitToolError("input", "Failed to expand @file input: %v", err)
}

func isASCIIWhitespace(b byte) bool {
	switch b {
	case ' ', '\t', '\n', '\r', '\f', '\v':
		return true
	default:
		return false
	}
}

const utf8RuneSelf = 0x80
