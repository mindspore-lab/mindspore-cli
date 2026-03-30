package panels

import (
	"regexp"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/mattn/go-runewidth"
)

var (
	ansiEscapePattern = regexp.MustCompile(`\x1b\[[0-9;]*m`)

	mdHeading1Style = lipgloss.NewStyle().
			Foreground(lipgloss.Color("230")).
			Bold(true)

	mdHeading2Style = lipgloss.NewStyle().
			Foreground(lipgloss.Color("223")).
			Bold(true)

	mdHeading3Style = lipgloss.NewStyle().
			Foreground(lipgloss.Color("252")).
			Bold(true)

	mdQuoteStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("245")).
			Italic(true)

	mdRuleStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("239"))

	mdCodeBlockStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("254")).
				BorderLeft(true).
				BorderStyle(lipgloss.ThickBorder()).
				BorderForeground(lipgloss.Color("67")).
				PaddingLeft(1).
				PaddingRight(1)

	mdCodeInlineStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("60"))

	mdBoldStyle = lipgloss.NewStyle().Bold(true)

	mdItalicStyle = lipgloss.NewStyle().Italic(true)

	mdStrikeStyle = lipgloss.NewStyle().Strikethrough(true)

	mdLinkStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("117")).
			Underline(true)

	mdLinkURLStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("244"))

	mdTableBorderStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("240"))

	mdTableHeaderStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("230")).
				Bold(true)

	mdCodeLangStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("214")).
			Background(lipgloss.Color("236")).
			Bold(true).
			PaddingLeft(1).
			PaddingRight(1)
)

func renderAgentContent(content string, width int) string {
	if content == "" {
		return ""
	}
	if ansiEscapePattern.MatchString(content) {
		return content
	}
	return renderMarkdown(content, width)
}

func renderMarkdown(content string, width int) string {
	lines := strings.Split(strings.ReplaceAll(content, "\r\n", "\n"), "\n")
	rendered := make([]string, 0, len(lines))
	inCodeBlock := false
	codeLang := ""

	for i := 0; i < len(lines); i++ {
		line := lines[i]
		trimmed := strings.TrimSpace(line)

		if strings.HasPrefix(trimmed, "```") {
			if !inCodeBlock {
				codeLang = strings.TrimSpace(strings.TrimPrefix(trimmed, "```"))
				if codeLang != "" {
					rendered = append(rendered, mdCodeLangStyle.Render(codeLang))
				}
			}
			inCodeBlock = !inCodeBlock
			if !inCodeBlock {
				codeLang = ""
			}
			continue
		}
		if inCodeBlock {
			rendered = append(rendered, renderCodeBlockLine(line, width))
			continue
		}
		if trimmed == "" {
			rendered = append(rendered, "")
			continue
		}
		if block, next, ok := markdownTable(lines, i, width); ok {
			rendered = append(rendered, block)
			i = next - 1
			continue
		}
		if heading, level, ok := markdownHeading(line); ok {
			rendered = append(rendered, markdownHeadingStyle(level).Render(renderInlineMarkdown(heading)))
			continue
		}
		if quote, ok := markdownQuote(line); ok {
			rendered = append(rendered, mdQuoteStyle.Render("│ "+renderInlineMarkdown(quote)))
			continue
		}
		if indent, checked, item, ok := markdownTaskItem(line); ok {
			rendered = append(rendered, renderListPrefix(indent, taskListMarker(checked))+renderInlineMarkdown(item))
			continue
		}
		if indent, item, ok := markdownBullet(line); ok {
			rendered = append(rendered, renderListPrefix(indent, "• ")+renderInlineMarkdown(item))
			continue
		}
		if indent, index, item, ok := markdownOrderedItem(line); ok {
			rendered = append(rendered, renderListPrefix(indent, index+". ")+renderInlineMarkdown(item))
			continue
		}
		if markdownRule(trimmed) {
			rendered = append(rendered, mdRuleStyle.Render("────────────────────"))
			continue
		}
		rendered = append(rendered, renderInlineMarkdown(line))
	}

	return strings.Join(rendered, "\n")
}

type tableAlignment int

const (
	alignLeft tableAlignment = iota
	alignCenter
	alignRight
)

func markdownTable(lines []string, start int, width int) (string, int, bool) {
	if start+1 >= len(lines) {
		return "", start, false
	}
	header := strings.TrimSpace(lines[start])
	separator := strings.TrimSpace(lines[start+1])
	if !isMarkdownTableRow(header) || !isMarkdownTableSeparator(separator) {
		return "", start, false
	}

	rows := [][]string{parseMarkdownTableRow(header)}
	i := start + 2
	for i < len(lines) {
		trimmed := strings.TrimSpace(lines[i])
		if trimmed == "" || !isMarkdownTableRow(trimmed) {
			break
		}
		rows = append(rows, parseMarkdownTableRow(trimmed))
		i++
	}
	if len(rows) < 2 {
		return "", start, false
	}
	return renderMarkdownTable(rows, parseMarkdownTableAlignment(separator), width), i, true
}

func isMarkdownTableRow(line string) bool {
	line = strings.TrimSpace(line)
	return strings.Count(line, "|") >= 2
}

func isMarkdownTableSeparator(line string) bool {
	if !isMarkdownTableRow(line) {
		return false
	}
	cells := parseMarkdownTableRow(line)
	if len(cells) == 0 {
		return false
	}
	for _, cell := range cells {
		cell = strings.TrimSpace(cell)
		if cell == "" {
			return false
		}
		for _, r := range cell {
			if r != '-' && r != ':' {
				return false
			}
		}
	}
	return true
}

func parseMarkdownTableRow(line string) []string {
	line = strings.TrimSpace(line)
	if strings.HasPrefix(line, "|") {
		line = strings.TrimPrefix(line, "|")
	}
	if strings.HasSuffix(line, "|") {
		line = strings.TrimSuffix(line, "|")
	}
	parts := strings.Split(line, "|")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		out = append(out, strings.TrimSpace(part))
	}
	return out
}

func renderMarkdownTable(rows [][]string, aligns []tableAlignment, width int) string {
	colCount := 0
	for _, row := range rows {
		if len(row) > colCount {
			colCount = len(row)
		}
	}
	widths := make([]int, colCount)
	for _, row := range rows {
		for col := 0; col < colCount; col++ {
			cell := ""
			if col < len(row) {
				cell = row[col]
			}
			if w := plainTextWidth(renderInlineMarkdown(cell)); w > widths[col] {
				widths[col] = w
			}
		}
	}
	widths = constrainTableWidths(widths, width)

	lines := []string{renderTableBorder("┌", "┬", "┐", widths), renderTableRow(rows[0], widths, true, aligns), renderTableBorder("├", "┼", "┤", widths)}
	for _, row := range rows[1:] {
		lines = append(lines, renderTableRow(row, widths, false, aligns))
	}
	lines = append(lines, renderTableBorder("└", "┴", "┘", widths))
	return strings.Join(lines, "\n")
}

func renderTableBorder(left, middle, right string, widths []int) string {
	parts := make([]string, 0, len(widths)*2+1)
	parts = append(parts, left)
	for i, width := range widths {
		if i > 0 {
			parts = append(parts, middle)
		}
		parts = append(parts, strings.Repeat("─", width+2))
	}
	parts = append(parts, right)
	return mdTableBorderStyle.Render(strings.Join(parts, ""))
}

func renderTableRow(row []string, widths []int, header bool, aligns ...[]tableAlignment) string {
	var b strings.Builder
	b.WriteString(mdTableBorderStyle.Render("│"))
	var colAligns []tableAlignment
	if len(aligns) > 0 {
		colAligns = aligns[0]
	}
	for i, width := range widths {
		text := ""
		if i < len(row) {
			text = row[i]
		}
		rendered := renderInlineMarkdown(truncatePlainText(text, width))
		leftPad, rightPad := tablePadding(width, plainTextWidth(rendered), alignmentAt(colAligns, i, header))
		b.WriteString(" ")
		b.WriteString(strings.Repeat(" ", leftPad))
		if header {
			b.WriteString(mdTableHeaderStyle.Render(rendered))
		} else {
			b.WriteString(rendered)
		}
		b.WriteString(strings.Repeat(" ", rightPad))
		b.WriteString(" ")
		b.WriteString(mdTableBorderStyle.Render("│"))
	}
	return b.String()
}

func constrainTableWidths(widths []int, maxWidth int) []int {
	if len(widths) == 0 {
		return widths
	}
	if maxWidth < minimumTableWidth(len(widths)) {
		maxWidth = minimumTableWidth(len(widths))
	}
	constrained := append([]int(nil), widths...)
	for tableRenderWidth(constrained) > maxWidth {
		col := widestColumnIndex(constrained)
		if constrained[col] <= 3 {
			break
		}
		constrained[col]--
	}
	return constrained
}

func tableRenderWidth(widths []int) int {
	width := 1
	for _, colWidth := range widths {
		width += colWidth + 3
	}
	return width
}

func minimumTableWidth(cols int) int {
	if cols <= 0 {
		return 1
	}
	return 1 + cols*3
}

func widestColumnIndex(widths []int) int {
	best := 0
	for i := 1; i < len(widths); i++ {
		if widths[i] > widths[best] {
			best = i
		}
	}
	return best
}

func truncatePlainText(text string, width int) string {
	if width <= 0 {
		return ""
	}
	if runewidth.StringWidth(text) <= width {
		return text
	}
	if width == 1 {
		return "…"
	}
	limit := width - 1
	var b strings.Builder
	used := 0
	for _, r := range text {
		rw := runewidth.RuneWidth(r)
		if used+rw > limit {
			break
		}
		b.WriteRune(r)
		used += rw
	}
	return b.String() + "…"
}

func renderCodeBlockLine(line string, width int) string {
	contentWidth := width - codeBlockDecorationWidth()
	if contentWidth < 1 {
		contentWidth = 1
	}
	return mdCodeBlockStyle.Render(truncatePlainText(line, contentWidth))
}

func codeBlockDecorationWidth() int {
	return mdCodeBlockStyle.GetHorizontalFrameSize()
}

func plainTextWidth(rendered string) int {
	return runewidth.StringWidth(ansiEscapePattern.ReplaceAllString(rendered, ""))
}

func markdownHeading(line string) (string, int, bool) {
	trimmed := strings.TrimLeft(line, " \t")
	level := 0
	for level < len(trimmed) && trimmed[level] == '#' {
		level++
	}
	if level == 0 || level > 6 || len(trimmed) <= level || trimmed[level] != ' ' {
		return "", 0, false
	}
	return strings.TrimSpace(trimmed[level:]), level, true
}

func markdownHeadingStyle(level int) lipgloss.Style {
	switch level {
	case 1:
		return mdHeading1Style
	case 2:
		return mdHeading2Style
	default:
		return mdHeading3Style
	}
}

func markdownQuote(line string) (string, bool) {
	trimmed := strings.TrimLeft(line, " \t")
	if !strings.HasPrefix(trimmed, ">") {
		return "", false
	}
	return strings.TrimSpace(strings.TrimPrefix(trimmed, ">")), true
}

func markdownBullet(line string) (int, string, bool) {
	indent, trimmed := markdownIndent(line)
	if len(trimmed) < 2 {
		return 0, "", false
	}
	switch trimmed[0] {
	case '-', '*', '+':
		if trimmed[1] == ' ' {
			return indent, strings.TrimSpace(trimmed[2:]), true
		}
	}
	return 0, "", false
}

func markdownTaskItem(line string) (int, bool, string, bool) {
	indent, trimmed := markdownIndent(line)
	if len(trimmed) < 6 {
		return 0, false, "", false
	}
	if (trimmed[0] != '-' && trimmed[0] != '*' && trimmed[0] != '+') || trimmed[1] != ' ' || trimmed[2] != '[' || trimmed[4] != ']' || trimmed[5] != ' ' {
		return 0, false, "", false
	}
	switch trimmed[3] {
	case ' ', 'x', 'X':
		return indent, trimmed[3] == 'x' || trimmed[3] == 'X', strings.TrimSpace(trimmed[6:]), true
	default:
		return 0, false, "", false
	}
}

func markdownOrderedItem(line string) (int, string, string, bool) {
	indent, trimmed := markdownIndent(line)
	i := 0
	for i < len(trimmed) && trimmed[i] >= '0' && trimmed[i] <= '9' {
		i++
	}
	if i == 0 || i+1 >= len(trimmed) || trimmed[i] != '.' || trimmed[i+1] != ' ' {
		return 0, "", "", false
	}
	return indent, trimmed[:i], strings.TrimSpace(trimmed[i+2:]), true
}

func markdownRule(line string) bool {
	if len(line) < 3 {
		return false
	}
	switch line[0] {
	case '-', '*', '_':
		for i := 1; i < len(line); i++ {
			if line[i] != line[0] {
				return false
			}
		}
		return true
	default:
		return false
	}
}

func renderInlineMarkdown(line string) string {
	var out strings.Builder
	for len(line) > 0 {
		switch {
		case strings.HasPrefix(line, "~~"):
			if text, rest, ok := markdownDelimited(line, "~~"); ok {
				out.WriteString(mdStrikeStyle.Render(renderInlineMarkdown(text)))
				line = rest
				continue
			}
		case strings.HasPrefix(line, "`"):
			end := strings.Index(line[1:], "`")
			if end >= 0 {
				end++
				out.WriteString(mdCodeInlineStyle.Render(line[1:end]))
				line = line[end+1:]
				continue
			}
		case strings.HasPrefix(line, "**"):
			if text, rest, ok := markdownDelimited(line, "**"); ok {
				out.WriteString(mdBoldStyle.Render(renderInlineMarkdown(text)))
				line = rest
				continue
			}
		case strings.HasPrefix(line, "__"):
			if text, rest, ok := markdownDelimited(line, "__"); ok {
				out.WriteString(mdBoldStyle.Render(renderInlineMarkdown(text)))
				line = rest
				continue
			}
		case strings.HasPrefix(line, "*"):
			if text, rest, ok := markdownDelimited(line, "*"); ok {
				out.WriteString(mdItalicStyle.Render(renderInlineMarkdown(text)))
				line = rest
				continue
			}
		case strings.HasPrefix(line, "_"):
			if text, rest, ok := markdownDelimited(line, "_"); ok {
				out.WriteString(mdItalicStyle.Render(renderInlineMarkdown(text)))
				line = rest
				continue
			}
		case strings.HasPrefix(line, "["):
			if label, url, rest, ok := markdownLink(line); ok {
				out.WriteString(mdLinkStyle.Render(renderInlineMarkdown(label)))
				if url != "" {
					out.WriteString(mdLinkURLStyle.Render(" (" + url + ")"))
				}
				line = rest
				continue
			}
		}

		next := nextMarkdownToken(line)
		plain := line
		if next > 0 {
			plain = line[:next]
			line = line[next:]
		} else {
			line = ""
		}
		out.WriteString(agentStyle.Render(plain))
	}
	return out.String()
}

func markdownDelimited(line, delim string) (string, string, bool) {
	if !strings.HasPrefix(line, delim) {
		return "", "", false
	}
	end := strings.Index(line[len(delim):], delim)
	if end < 0 {
		return "", "", false
	}
	end += len(delim)
	return line[len(delim):end], line[end+len(delim):], true
}

func markdownLink(line string) (string, string, string, bool) {
	closeLabel := strings.Index(line, "](")
	if closeLabel <= 1 {
		return "", "", "", false
	}
	closeURL := strings.Index(line[closeLabel+2:], ")")
	if closeURL < 0 {
		return "", "", "", false
	}
	closeURL += closeLabel + 2
	return line[1:closeLabel], line[closeLabel+2 : closeURL], line[closeURL+1:], true
}

func nextMarkdownToken(line string) int {
	indexes := []int{
		strings.Index(line, "~~"),
		strings.Index(line, "`"),
		strings.Index(line, "**"),
		strings.Index(line, "__"),
		strings.Index(line, "*"),
		strings.Index(line, "_"),
		strings.Index(line, "["),
	}
	best := -1
	for _, idx := range indexes {
		if idx < 0 {
			continue
		}
		if best < 0 || idx < best {
			best = idx
		}
	}
	return best
}

func markdownIndent(line string) (int, string) {
	spaces := 0
	for _, r := range line {
		if r == ' ' {
			spaces++
			continue
		}
		if r == '\t' {
			spaces += 2
			continue
		}
		break
	}
	return spaces / 2, strings.TrimLeft(line, " \t")
}

func renderListPrefix(indent int, marker string) string {
	return agentStyle.Render(strings.Repeat("  ", indent) + marker)
}

func taskListMarker(checked bool) string {
	if checked {
		return "[x] "
	}
	return "[ ] "
}

func parseMarkdownTableAlignment(line string) []tableAlignment {
	cells := parseMarkdownTableRow(line)
	aligns := make([]tableAlignment, 0, len(cells))
	for _, cell := range cells {
		cell = strings.TrimSpace(cell)
		left := strings.HasPrefix(cell, ":")
		right := strings.HasSuffix(cell, ":")
		switch {
		case left && right:
			aligns = append(aligns, alignCenter)
		case right:
			aligns = append(aligns, alignRight)
		default:
			aligns = append(aligns, alignLeft)
		}
	}
	return aligns
}

func alignmentAt(aligns []tableAlignment, col int, header bool) tableAlignment {
	if header {
		return alignCenter
	}
	if col >= 0 && col < len(aligns) {
		return aligns[col]
	}
	return alignLeft
}

func tablePadding(cellWidth, contentWidth int, align tableAlignment) (int, int) {
	if contentWidth >= cellWidth {
		return 0, 0
	}
	space := cellWidth - contentWidth
	switch align {
	case alignRight:
		return space, 0
	case alignCenter:
		left := space / 2
		return left, space - left
	default:
		return 0, space
	}
}
