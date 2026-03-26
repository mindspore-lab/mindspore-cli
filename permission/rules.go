package permission

import (
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// PermissionRule is a parsed Tool/Tool(specifier) rule.
type PermissionRule struct {
	Raw       string
	Tool      string
	Specifier string
}

// RuleViewItem is a flattened view for UI and command output.
type RuleViewItem struct {
	Rule   string
	Level  PermissionLevel
	Source string
}

type compiledRule struct {
	Rule   PermissionRule
	Level  PermissionLevel
	Source string
	Order  int
}

// ParsePermissionRule parses rules in Tool or Tool(specifier) format.
func ParsePermissionRule(raw string) (PermissionRule, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return PermissionRule{}, fmt.Errorf("empty rule")
	}

	open := strings.Index(raw, "(")
	if open == -1 {
		tool := strings.ToLower(strings.TrimSpace(raw))
		if tool == "" {
			return PermissionRule{}, fmt.Errorf("invalid rule: %q", raw)
		}
		return PermissionRule{Raw: raw, Tool: tool}, nil
	}

	if !strings.HasSuffix(raw, ")") {
		return PermissionRule{}, fmt.Errorf("invalid rule syntax: %q", raw)
	}
	tool := strings.ToLower(strings.TrimSpace(raw[:open]))
	if tool == "" {
		return PermissionRule{}, fmt.Errorf("invalid rule syntax: %q", raw)
	}
	specifier := strings.TrimSpace(raw[open+1 : len(raw)-1])

	if tool == "bash" && strings.Contains(specifier, ":*") {
		specifier = strings.ReplaceAll(specifier, ":*", " *")
	}

	return PermissionRule{Raw: raw, Tool: tool, Specifier: specifier}, nil
}

func ruleFromToolLiteral(tool string) string {
	switch strings.ToLower(strings.TrimSpace(tool)) {
	case "shell":
		return "Bash"
	case "read", "grep", "glob":
		return "Read"
	case "edit":
		return "Edit"
	case "write":
		return "Write"
	case "webfetch":
		return "WebFetch"
	case "agent":
		return "Agent"
	default:
		return strings.TrimSpace(tool)
	}
}

func matchRule(rule PermissionRule, tool, action, path string) bool {
	tool = strings.ToLower(strings.TrimSpace(tool))
	action = strings.TrimSpace(action)
	path = normalizePath(path)

	switch rule.Tool {
	case "bash":
		if tool != "shell" {
			return false
		}
		return matchBashSpecifier(rule.Specifier, action)
	case "read":
		if !isReadTool(tool) {
			return false
		}
		return matchPathSpecifier(rule.Specifier, preferPath(path, action))
	case "edit":
		if tool != "edit" {
			return false
		}
		return matchPathSpecifier(rule.Specifier, preferPath(path, action))
	case "write":
		if tool != "write" {
			return false
		}
		return matchPathSpecifier(rule.Specifier, preferPath(path, action))
	case "webfetch":
		if tool != "webfetch" {
			return false
		}
		return matchWebSpecifier(rule.Specifier, action)
	case "agent":
		if tool != "agent" {
			return false
		}
		if rule.Specifier == "" {
			return true
		}
		return wildcardMatch(strings.ToLower(rule.Specifier), strings.ToLower(action))
	default:
		if strings.HasPrefix(rule.Tool, "mcp__") {
			return wildcardMatch(strings.ToLower(rule.Tool), tool)
		}
		if rule.Specifier == "" {
			return wildcardMatch(strings.ToLower(rule.Tool), tool)
		}
		return rule.Tool == tool && wildcardMatch(strings.ToLower(rule.Specifier), strings.ToLower(action))
	}
}

func isReadTool(tool string) bool {
	switch tool {
	case "read", "glob", "grep":
		return true
	default:
		return false
	}
}

func preferPath(path, action string) string {
	if strings.TrimSpace(path) != "" {
		return path
	}
	return action
}

func matchBashSpecifier(spec, command string) bool {
	spec = strings.TrimSpace(spec)
	command = strings.TrimSpace(command)

	if spec == "" || spec == "*" {
		return true
	}
	if command == "" {
		return false
	}
	if hasShellOperator(command) && !hasShellOperator(spec) {
		return false
	}
	return wildcardMatch(normalizeBashCommandPattern(spec), normalizeBashCommandPattern(command))
}

func hasShellOperator(s string) bool {
	return strings.Contains(s, "&&") || strings.Contains(s, "||") || strings.Contains(s, ";") || strings.Contains(s, "|")
}

func splitShellCommand(command string) []string {
	command = strings.TrimSpace(command)
	if command == "" {
		return nil
	}
	if !hasShellOperator(command) {
		return []string{command}
	}
	replacer := strings.NewReplacer("&&", "\n", "||", "\n", ";", "\n", "|", "\n")
	parts := strings.Split(replacer.Replace(command), "\n")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		out = append(out, p)
	}
	if len(out) == 0 {
		return []string{command}
	}
	return out
}

func normalizeBashCommandPattern(input string) string {
	s := strings.TrimSpace(input)
	if s == "" {
		return s
	}
	s = strings.ReplaceAll(s, "\r\n", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\t", " ")
	s = strings.Join(strings.Fields(s), " ")
	s = strings.ReplaceAll(s, `\(`, "(")
	s = strings.ReplaceAll(s, `\)`, ")")
	s = stripSimpleShellQuotes(s)
	s = normalizeFindGroupingParens(s)
	return s
}

func stripSimpleShellQuotes(s string) string {
	var b strings.Builder
	inSingle := false
	inDouble := false
	escaped := false
	for i := 0; i < len(s); i++ {
		ch := s[i]
		if escaped {
			b.WriteByte(ch)
			escaped = false
			continue
		}
		if ch == '\\' {
			escaped = true
			b.WriteByte(ch)
			continue
		}
		if ch == '\'' && !inDouble {
			inSingle = !inSingle
			continue
		}
		if ch == '"' && !inSingle {
			inDouble = !inDouble
			continue
		}
		b.WriteByte(ch)
	}
	return b.String()
}

func normalizeFindGroupingParens(s string) string {
	fields := strings.Fields(s)
	if len(fields) == 0 || fields[0] != "find" {
		return s
	}
	out := make([]string, 0, len(fields))
	for _, tok := range fields {
		if tok == "(" || tok == ")" {
			continue
		}
		out = append(out, tok)
	}
	return strings.Join(out, " ")
}

func matchPathSpecifier(spec, targetPath string) bool {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return true
	}
	absTarget, relTarget := pathForms(targetPath)
	if absTarget == "" && relTarget == "" {
		return false
	}

	mode, pattern := normalizePathPattern(spec)
	switch mode {
	case "absolute":
		return absTarget != "" && wildcardMatch(pattern, absTarget)
	case "relative":
		return relTarget != "" && gitignoreMatch(pattern, relTarget)
	default:
		if relTarget != "" && gitignoreMatch(pattern, relTarget) {
			return true
		}
		if absTarget != "" && wildcardMatch(pattern, absTarget) {
			return true
		}
		return false
	}
}

func normalizePathPattern(spec string) (mode string, pattern string) {
	spec = normalizePath(strings.TrimSpace(spec))
	if spec == "" {
		return "any", spec
	}
	if strings.HasPrefix(spec, "//") {
		return "absolute", cleanSlashPath("/" + strings.TrimPrefix(spec, "//"))
	}
	if strings.HasPrefix(spec, "~/") {
		home, err := os.UserHomeDir()
		if err == nil && strings.TrimSpace(home) != "" {
			home = normalizePath(home)
			return "absolute", cleanSlashPath(filepath.ToSlash(filepath.Join(home, spec[2:])))
		}
		return "absolute", cleanSlashPath("/" + strings.TrimPrefix(spec, "~/"))
	}
	if strings.HasPrefix(spec, "./") {
		return "relative", strings.TrimPrefix(spec, "./")
	}
	if strings.HasPrefix(spec, "/") {
		if cwd, err := os.Getwd(); err == nil && strings.TrimSpace(cwd) != "" {
			return "absolute", cleanSlashPath(filepath.ToSlash(filepath.Join(cwd, strings.TrimPrefix(spec, "/"))))
		}
		return "relative", strings.TrimPrefix(spec, "/")
	}
	return "relative", spec
}

func normalizePath(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	return filepath.ToSlash(path)
}

func pathForms(input string) (absolute string, relative string) {
	input = normalizePath(input)
	if input == "" {
		return "", ""
	}
	cwd := ""
	if wd, err := os.Getwd(); err == nil && strings.TrimSpace(wd) != "" {
		cwd = cleanSlashPath(filepath.ToSlash(wd))
	}

	if filepath.IsAbs(input) {
		absolute = cleanSlashPath(input)
		if cwd != "" {
			if rel, err := filepath.Rel(cwd, absolute); err == nil {
				rel = filepath.ToSlash(rel)
				if rel != "." && !strings.HasPrefix(rel, "../") {
					relative = rel
				}
			}
		}
		return absolute, relative
	}

	relative = strings.TrimPrefix(input, "./")
	if cwd != "" {
		absolute = cleanSlashPath(filepath.ToSlash(filepath.Join(cwd, relative)))
	}
	return absolute, relative
}

func cleanSlashPath(p string) string {
	p = filepath.ToSlash(filepath.Clean(p))
	if p == "." {
		return ""
	}
	return p
}

func gitignoreMatch(pattern, target string) bool {
	pattern = strings.TrimSpace(filepath.ToSlash(pattern))
	target = strings.TrimSpace(filepath.ToSlash(target))
	target = strings.TrimPrefix(target, "./")
	target = strings.TrimPrefix(target, "/")
	if pattern == "" || target == "" {
		return false
	}

	pattern = strings.TrimPrefix(pattern, "./")
	anchored := strings.HasPrefix(pattern, "/")
	if anchored {
		pattern = strings.TrimPrefix(pattern, "/")
	}

	dirOnly := strings.HasSuffix(pattern, "/")
	pattern = strings.TrimSuffix(pattern, "/")
	if pattern == "" {
		return false
	}

	hasSlash := strings.Contains(pattern, "/")
	if !hasSlash {
		segPattern, ok := globToRegex(pattern, true)
		if !ok {
			return false
		}
		for _, seg := range strings.Split(target, "/") {
			if segPattern.MatchString(seg) {
				return true
			}
		}
		return false
	}

	rx, ok := globToRegex(pattern, false)
	if !ok {
		return false
	}

	if anchored {
		if !rx.MatchString(target) {
			return false
		}
		if dirOnly {
			return target == pattern || strings.HasPrefix(target, pattern+"/")
		}
		return true
	}

	parts := strings.Split(target, "/")
	for i := 0; i < len(parts); i++ {
		sub := strings.Join(parts[i:], "/")
		if rx.MatchString(sub) {
			if !dirOnly {
				return true
			}
			return sub == pattern || strings.HasPrefix(sub, pattern+"/")
		}
	}
	return false
}

func globToRegex(glob string, noSlash bool) (*regexp.Regexp, bool) {
	var b strings.Builder
	b.WriteString("^")
	for i := 0; i < len(glob); {
		switch glob[i] {
		case '*':
			if i+1 < len(glob) && glob[i+1] == '*' {
				if noSlash {
					b.WriteString(".*")
				} else {
					b.WriteString(".*")
				}
				i += 2
			} else {
				if noSlash {
					b.WriteString("[^/]*")
				} else {
					b.WriteString("[^/]*")
				}
				i++
			}
		case '?':
			if noSlash {
				b.WriteString("[^/]")
			} else {
				b.WriteString("[^/]")
			}
			i++
		case '.', '+', '(', ')', '|', '^', '$', '{', '}', '\\':
			b.WriteByte('\\')
			b.WriteByte(glob[i])
			i++
		case '[':
			j := i + 1
			for j < len(glob) && glob[j] != ']' {
				j++
			}
			if j >= len(glob) {
				b.WriteString("\\[")
				i++
				continue
			}
			class := glob[i+1 : j]
			if strings.HasPrefix(class, "!") {
				class = "^" + class[1:]
			}
			b.WriteString("[")
			b.WriteString(class)
			b.WriteString("]")
			i = j + 1
		default:
			b.WriteByte(glob[i])
			i++
		}
	}
	b.WriteString("$")
	rx, err := regexp.Compile(b.String())
	return rx, err == nil
}

func matchWebSpecifier(spec, action string) bool {
	spec = strings.TrimSpace(spec)
	action = strings.TrimSpace(action)
	if spec == "" {
		return true
	}
	specLower := strings.ToLower(spec)
	if strings.HasPrefix(specLower, "domain:") {
		domain := strings.TrimSpace(strings.TrimPrefix(specLower, "domain:"))
		if domain == "" {
			return false
		}
		host := strings.ToLower(extractHost(action))
		if host == "" {
			return false
		}
		return host == domain || strings.HasSuffix(host, "."+domain)
	}
	return wildcardMatch(specLower, strings.ToLower(action))
}

func extractHost(action string) string {
	candidates := strings.Fields(action)
	if len(candidates) == 0 {
		candidates = []string{action}
	}
	for _, c := range candidates {
		c = strings.TrimSpace(c)
		if c == "" {
			continue
		}
		u, err := url.Parse(c)
		if err == nil && u != nil && strings.TrimSpace(u.Host) != "" {
			return strings.TrimSpace(u.Hostname())
		}
		if !strings.Contains(c, "://") {
			if u2, err := url.Parse("https://" + c); err == nil && u2 != nil && strings.TrimSpace(u2.Host) != "" {
				return strings.TrimSpace(u2.Hostname())
			}
		}
	}
	return ""
}

func wildcardMatch(pattern, s string) bool {
	if pattern == "" {
		return s == ""
	}
	if pattern == "*" {
		return true
	}

	parts := strings.Split(pattern, "*")
	if len(parts) == 1 {
		return pattern == s
	}

	if !strings.HasPrefix(pattern, "*") {
		if !strings.HasPrefix(s, parts[0]) {
			return false
		}
		s = s[len(parts[0]):]
		parts = parts[1:]
	}

	for i, part := range parts {
		if part == "" {
			continue
		}
		if i == len(parts)-1 && !strings.HasSuffix(pattern, "*") {
			idx := strings.LastIndex(s, part)
			return idx >= 0 && idx+len(part) == len(s)
		}
		idx := strings.Index(s, part)
		if idx < 0 {
			return false
		}
		s = s[idx+len(part):]
	}
	return true
}
