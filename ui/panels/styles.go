package panels

import (
	"github.com/charmbracelet/lipgloss"
	"github.com/mindspore-lab/mindspore-cli/ui/theme"
)

// badge builds a bold badge style with the given foreground/background and standard padding.
func badge(fg, bg lipgloss.Color) lipgloss.Style {
	return lipgloss.NewStyle().Bold(true).Foreground(fg).Background(bg).Padding(0, 1)
}

// InitStyles rebuilds all package-level style vars from theme.Current.
// Must be called after theme.Apply() and before the TUI starts.
func InitStyles() {
	t := theme.Current

	// ── chat.go ──────────────────────────────────────────────────
	userStyle = lipgloss.NewStyle().Foreground(t.Accent).Bold(true)
	userBlockStyle = lipgloss.NewStyle().Foreground(t.TextPrimary).Background(lipgloss.Color("248"))
	agentStyle = lipgloss.NewStyle().Foreground(t.TextPrimary)
	thinkingStyle = lipgloss.NewStyle().Foreground(t.Thinking).Italic(true)
	toolBorderStyle = lipgloss.NewStyle().Foreground(t.TextMuted)
	toolHeaderStyle = lipgloss.NewStyle().Foreground(t.Warning).Bold(true)
	toolContentStyle = lipgloss.NewStyle().Foreground(t.TextPrimary).PaddingLeft(2)
	collapsedIconStyle = lipgloss.NewStyle().Foreground(t.TextMuted)
	collapsedNameStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)
	collapsedTitleStyle = lipgloss.NewStyle().Foreground(t.TextPrimary).Bold(true)
	collapsedSummaryStyle = lipgloss.NewStyle().Foreground(t.TextMuted).Italic(true)
	errorBorderStyle = lipgloss.NewStyle().Foreground(t.Error)
	errorHeaderStyle = lipgloss.NewStyle().Foreground(t.Error).Bold(true)
	errorContentStyle = lipgloss.NewStyle().Foreground(t.ErrorLight).PaddingLeft(2)
	diffAddStyle = lipgloss.NewStyle().Foreground(t.Success)
	diffRemoveStyle = lipgloss.NewStyle().Foreground(t.ErrorLight)
	diffNeutralStyle = lipgloss.NewStyle().Foreground(t.TextPrimary).PaddingLeft(2)
	toolPendingDotStyle = lipgloss.NewStyle().Foreground(t.TextMuted)
	toolSuccessDotStyle = lipgloss.NewStyle().Foreground(t.Success)
	toolWarningDotStyle = lipgloss.NewStyle().Foreground(t.Warning)
	toolErrorDotStyle = lipgloss.NewStyle().Foreground(t.Error)
	toolCallLineStyle = lipgloss.NewStyle().Foreground(t.TextPrimary)
	toolPendingStatusStyle = lipgloss.NewStyle().Foreground(t.TextSecondary).Italic(true)
	toolResultPrefixStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)
	toolResultSummaryStyle = lipgloss.NewStyle().Foreground(t.TextPrimary)
	toolResultDetailStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)
	toolResultWarningStyle = lipgloss.NewStyle().Foreground(t.Warning)
	toolResultErrorStyle = lipgloss.NewStyle().Foreground(t.ErrorLight)

	// ── topbar.go ────────────────────────────────────────────────
	brandStyle = lipgloss.NewStyle().Foreground(t.TextPrimary).Bold(true)
	infoStyle = lipgloss.NewStyle().Foreground(t.AccentAlt)
	sepStyle = lipgloss.NewStyle().Foreground(t.AccentAlt)
	dividerStyle = lipgloss.NewStyle().Foreground(t.AccentAlt)
	bannerLabelStyle = lipgloss.NewStyle().Foreground(t.AccentAlt)
	bannerValueStyle = lipgloss.NewStyle().Foreground(t.AccentAlt)
	bannerDimStyle = lipgloss.NewStyle().Foreground(t.AccentAlt).Italic(true)

	// ── hintbar.go ───────────────────────────────────────────────
	hintDividerStyle = lipgloss.NewStyle().Foreground(t.Border)
	hintTextStyle = lipgloss.NewStyle().Foreground(t.TextSecondary).PaddingLeft(1)
	hintKeyStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)
	hintDescStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)
	hintSepStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)

	// ── boot.go ──────────────────────────────────────────────────
	bootMessageBaseStyle = lipgloss.NewStyle().Foreground(t.TextMuted).Bold(true)
	bootMessageGlowStyle = lipgloss.NewStyle().Foreground(t.TextPrimary).Bold(true)
	bootMessageHotStyle = lipgloss.NewStyle().Foreground(t.BadgeFGBright).Bold(true)

	// ── model_setup.go ───────────────────────────────────────────
	setupTitleStyle = lipgloss.NewStyle().Foreground(t.Accent).Bold(true).Align(lipgloss.Center)
	setupNormalStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)
	setupSelectedStyle = lipgloss.NewStyle().Foreground(t.Accent).Bold(true)
	setupDisabledStyle = lipgloss.NewStyle().Foreground(t.TextMuted)
	setupHintStyle = lipgloss.NewStyle().Foreground(t.TextMuted).Italic(true)
	setupErrorStyle = lipgloss.NewStyle().Foreground(t.Error)
	setupLabelStyle = lipgloss.NewStyle().Foreground(t.TextPrimary)
	setupBadgeStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)
	modelPickerDescStyle = lipgloss.NewStyle().Foreground(t.TextMuted)
	setupBorderStyle = lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(t.TextPrimary).
		Padding(0, 2)
	tokenCursorStyle = lipgloss.NewStyle().Background(t.TextPrimary).Foreground(t.SurfaceDim)
	tokenTextStyle = lipgloss.NewStyle().Foreground(t.TextPrimary)

	// ── train_setup.go ───────────────────────────────────────────
	trainTitleStyle = lipgloss.NewStyle().Bold(true).Foreground(t.Warning)
	sectionHeaderStyle = lipgloss.NewStyle().Bold(true).Foreground(t.TextPrimary).Background(t.SurfaceDim).Padding(0, 1)
	phaseBadgeSetup = badge(t.BadgeFG, t.Warning)
	phaseBadgeReady = badge(t.BadgeFG, t.Success)
	phaseBadgeRunning = badge(t.BadgeFG, t.Accent)
	phaseBadgeCompleted = badge(t.BadgeFG, t.Success)
	phaseBadgeFailed = badge(t.BadgeFGBright, t.Error)
	phaseBadgeStopped = badge(t.BadgeFG, t.TextMuted)
	phaseBadgeDrift = badge(t.BadgeFGBright, t.Error)
	phaseBadgeAnalyzing = badge(t.BadgeFG, t.Warning)
	phaseBadgeRerunning = badge(t.BadgeFG, lipgloss.Color("69"))
	checkPassedStyle = lipgloss.NewStyle().Foreground(t.Success)
	checkFailedStyle = lipgloss.NewStyle().Foreground(t.Error)
	checkRunningStyle = lipgloss.NewStyle().Foreground(t.Warning)
	checkPendingStyle = lipgloss.NewStyle().Foreground(t.TextMuted)
	checkDetailStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)
	trainDividerStyle = lipgloss.NewStyle().Foreground(t.SurfaceDim)
	metricLabelStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)
	actionNormalStyle = lipgloss.NewStyle().Foreground(t.TextPrimary).Background(t.Border).Padding(0, 2)
	actionFocusedStyle = lipgloss.NewStyle().Foreground(t.BadgeFG).Background(t.Accent).Bold(true).Padding(0, 2)
	actionDangerStyle = lipgloss.NewStyle().Foreground(t.BadgeFGBright).Background(t.Error).Bold(true).Padding(0, 2)
	actionDangerFocusedStyle = lipgloss.NewStyle().Foreground(t.BadgeFG).Background(t.ErrorLight).Bold(true).Padding(0, 2)
	actionDisabledStyle = lipgloss.NewStyle().Foreground(t.Border).Background(t.SurfaceDim).Strikethrough(true).Padding(0, 2)

	// ── train_compare_summary.go ─────────────────────────────────
	cmpLabelStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)
	cmpGoodStyle = lipgloss.NewStyle().Foreground(t.Success).Bold(true)
	cmpBadStyle = lipgloss.NewStyle().Foreground(t.Error).Bold(true)
	cmpWarnStyle = lipgloss.NewStyle().Foreground(t.Warning).Bold(true)
	cmpNeutStyle = lipgloss.NewStyle().Foreground(t.TextPrimary).Bold(true)

	// ── train_logs.go ────────────────────────────────────────────
	logTitleStyle = lipgloss.NewStyle().Bold(true).Foreground(t.TextPrimary).Background(t.SurfaceDim).Padding(0, 1)
	logLineStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)
	logMetricLineStyle = lipgloss.NewStyle().Foreground(t.Accent)
	logErrorLineStyle = lipgloss.NewStyle().Foreground(t.Error).Bold(true)
	logHighlightStyle = lipgloss.NewStyle().Foreground(t.Success)
	logTimestampStyle = lipgloss.NewStyle().Foreground(t.TextMuted)
	logBorderStyle = lipgloss.NewStyle().Foreground(t.SurfaceDim)

	// ── train_issue.go ───────────────────────────────────────────
	issueKindStyle = lipgloss.NewStyle().Foreground(t.Warning).Bold(true)
	issueSummaryStyle = lipgloss.NewStyle().Foreground(t.TextPrimary)

	// ── train_lane_panel.go ──────────────────────────────────────
	laneHeaderStyle = lipgloss.NewStyle().Bold(true).Foreground(t.TextPrimary)
	laneSubStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)
	laneBadgeRunning = badge(t.BadgeFG, t.Accent)
	laneBadgeCompleted = badge(t.BadgeFG, t.Success)
	laneBadgeFailed = badge(t.BadgeFGBright, t.Error)
	laneBadgePending = badge(t.BadgeFG, t.TextMuted)
	laneMetricLabel = lipgloss.NewStyle().Foreground(t.TextSecondary)

	// ── train_workspace.go ───────────────────────────────────────
	panelStubStyle = lipgloss.NewStyle().Foreground(t.TextSecondary).Padding(0, 1)
	runBarStyle = lipgloss.NewStyle().Foreground(t.TextPrimary).Padding(0, 1)
}
