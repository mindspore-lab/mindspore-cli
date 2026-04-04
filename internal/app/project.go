package app

import (
	"fmt"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	projectpkg "github.com/vigo999/mindspore-cli/internal/project"
	"github.com/vigo999/mindspore-cli/ui/model"
)

var runProjectGit = func(workDir string, args ...string) (string, error) {
	cmd := exec.Command("git", append([]string{"-C", workDir}, args...)...)
	out, err := cmd.CombinedOutput()
	text := strings.TrimSpace(string(out))
	if err != nil {
		if text == "" {
			return "", err
		}
		return text, fmt.Errorf("%w: %s", err, text)
	}
	return text, nil
}

type projectCard struct {
	Status     model.ProjectStatusView
	Overview   []string
	Milestones []projectTask
	Tasks      []projectTask
	Support    []string
}

type projectTask struct {
	ID       string
	Title    string
	Status   string
	Progress int
	Owner    string
	Due      string
	Tags     string
}

func (a *Application) cmdProjectInput(raw string) {
	args, err := splitProjectArgs(raw)
	if err != nil {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: fmt.Sprintf("project command parse error: %v", err),
		}
		return
	}
	a.cmdProject(args)
}

func (a *Application) cmdProject(args []string) {
	action := "status"
	if len(args) > 0 {
		action = strings.ToLower(strings.TrimSpace(args[0]))
	}

	switch action {
	case "", "status", "show", "open", "refresh":
		a.emitProjectSnapshot()
	case "tasks":
		tag := ""
		if len(args) > 1 {
			tag = strings.TrimSpace(args[1])
		}
		a.emitProjectTasksByTag(tag)
	case "close", "exit":
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "project status is stream-only now. Run /project again to refresh the snapshot.",
		}
	case "add":
		if !a.ensureAdmin() {
			return
		}
		if err := a.cmdProjectAdd(args[1:]); err != nil {
			a.emitProjectCommandError(err)
		}
	case "update":
		if !a.ensureAdmin() {
			return
		}
		if err := a.cmdProjectUpdate(args[1:]); err != nil {
			a.emitProjectCommandError(err)
		}
	case "rm", "remove", "delete":
		if !a.ensureAdmin() {
			return
		}
		if err := a.cmdProjectRemove(args[1:]); err != nil {
			a.emitProjectCommandError(err)
		}
	case "overview":
		if !a.ensureAdmin() {
			return
		}
		if err := a.cmdProjectOverview(args[1:]); err != nil {
			a.emitProjectCommandError(err)
		}
	default:
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Usage: /project [status] | /project add <section> \"<title>\" [--id id] [--owner owner] [--progress pct] | /project update <section> <target> [--title title] [--owner owner] [--progress pct] | /project rm <section> <target>",
		}
	}
}

func (a *Application) emitProjectSnapshot() {
	status, err := collectProjectStatus(a.WorkDir)
	if err != nil {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "project",
			Message:  fmt.Sprintf("project status failed: %v", err),
		}
		return
	}

	var card projectCard
	card.Status = status

	if a.ensureProjectService() {
		snap, err := a.projectService.Snapshot()
		if err == nil {
			card = buildCardFromSnapshot(snap, status)
		}
	}

	// Fallback overview when not logged in or server returns empty
	if len(card.Overview) == 0 {
		card.Overview = fallbackOverview(status)
	}

	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		RawANSI: true,
		Message: renderProjectCard(card),
	}
}

func buildCardFromSnapshot(snap *projectpkg.Snapshot, status model.ProjectStatusView) projectCard {
	card := projectCard{Status: status}

	ov := snap.Overview
	if ov.Phase != "" {
		card.Overview = append(card.Overview, "  phase: "+ov.Phase)
	}
	if ov.Owner != "" {
		card.Overview = append(card.Overview, "  owner: "+ov.Owner)
	}
	if ov.Repo != "" {
		card.Overview = append(card.Overview, "  repo: "+ov.Repo)
	}
	if ov.Branch != "" {
		card.Overview = append(card.Overview, "  branch: "+ov.Branch)
	}

	for _, t := range snap.Tasks {
		pt := projectTask{
			ID:       strconv.Itoa(t.ID),
			Title:    t.Title,
			Status:   t.Status,
			Progress: t.Progress,
			Owner:    t.Owner,
			Due:      t.Due,
			Tags:     t.Tags,
		}
		switch t.Section {
		case "milestones":
			card.Milestones = append(card.Milestones, pt)
		case "support":
			card.Support = append(card.Support, t.Title)
		default:
			card.Tasks = append(card.Tasks, pt)
		}
	}
	return card
}

func taskToChecklist(t projectTask) string {
	marker := "[ ] "
	if normalizeTaskStatus(t.Status) == "done" || t.Progress >= 100 {
		marker = "[x] "
	}
	return marker + "`#" + t.ID + "` " + t.Title
}

func (a *Application) emitProjectTasksByTag(tag string) {
	if !a.ensureProjectService() {
		return
	}
	snap, err := a.projectService.Snapshot()
	if err != nil {
		a.emitProjectCommandError(err)
		return
	}
	var tasks []projectTask
	for _, t := range snap.Tasks {
		if t.Section != "tasks" {
			continue
		}
		if tag != "" && !containsTag(t.Tags, tag) {
			continue
		}
		tasks = append(tasks, projectTask{
			ID:       strconv.Itoa(t.ID),
			Title:    t.Title,
			Status:   t.Status,
			Progress: t.Progress,
			Owner:    t.Owner,
			Due:      t.Due,
			Tags:     t.Tags,
		})
	}
	if len(tasks) == 0 {
		msg := "no tasks found"
		if tag != "" {
			msg += " with tag: " + tag
		}
		a.EventCh <- model.Event{Type: model.AgentReply, Message: msg}
		return
	}
	header := "[ TASKS ]"
	if tag != "" {
		header = "[ TASKS: " + tag + " ]"
	}
	lines := []string{header}
	lines = append(lines, renderTaskLines(tasks)...)
	a.EventCh <- model.Event{Type: model.AgentReply, RawANSI: true, Message: strings.Join(lines, "\n")}
}

func containsTag(tags, target string) bool {
	for _, t := range strings.Split(tags, ",") {
		if strings.EqualFold(strings.TrimSpace(t), target) {
			return true
		}
	}
	return false
}

// extractTags parses "[tag1,tag2] title" into tags and title.
// If no brackets, returns empty tags and the original string.
func extractTags(s string) (tags, title string) {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "[") {
		return "", s
	}
	idx := strings.Index(s, "]")
	if idx < 0 {
		return "", s
	}
	tags = strings.TrimSpace(s[1:idx])
	title = strings.TrimSpace(s[idx+1:])
	return tags, title
}

func (a *Application) emitProjectCommandError(err error) {
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("project command failed: %v", err),
	}
}

func (a *Application) cmdProjectAdd(args []string) error {
	if len(args) < 2 {
		return fmt.Errorf("usage: /project add <section> \"<title>\" [--owner owner] [--progress pct]")
	}
	if !a.ensureProjectService() {
		return fmt.Errorf("not logged in. Run /login <token> first")
	}
	section := canonicalSection(args[0])
	rawTitle := strings.TrimSpace(args[1])
	if rawTitle == "" {
		return fmt.Errorf("title cannot be empty")
	}
	inlineTags, title := extractTags(rawTitle)
	if title == "" {
		return fmt.Errorf("title cannot be empty")
	}
	opts, err := parseProjectTaskOptions(args[2:], false)
	if err != nil {
		return err
	}
	tags := opts.Tags
	if inlineTags != "" {
		if tags != "" {
			tags = inlineTags + "," + tags
		} else {
			tags = inlineTags
		}
	}
	owner := opts.Owner
	if owner == "" {
		owner = a.issueUser
	}
	task, err := a.projectService.AddTask(section, title, owner, a.issueUser, opts.Due, tags, opts.Progress)
	if err != nil {
		return err
	}
	var msg string
	switch section {
	case "tasks":
		msg = fmt.Sprintf("created task #%d: %s", task.ID, task.Title)
	case "milestones":
		msg = fmt.Sprintf("added milestone: %s", task.Title)
	case "support":
		msg = fmt.Sprintf("added support: %s", task.Title)
	default:
		msg = fmt.Sprintf("created %s: %s", section, task.Title)
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: msg,
	}
	return nil
}

func (a *Application) cmdProjectUpdate(args []string) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: /project update <id|title> [--title title] [--owner owner] [--progress pct] [--status status] [--due date]")
	}
	if !a.ensureProjectService() {
		return fmt.Errorf("not logged in. Run /login <token> first")
	}
	id, target, err := a.resolveTaskTarget(args[0])
	if err != nil {
		return err
	}
	_ = target
	opts, err := parseProjectTaskOptions(args[1:], false)
	if err != nil {
		return err
	}
	if !opts.HasUpdates() {
		return fmt.Errorf("update requires at least one of --title, --owner, --status, --due, or --progress")
	}
	var titlePtr, ownerPtr, statusPtr, duePtr, tagsPtr *string
	if strings.TrimSpace(opts.Title) != "" {
		titlePtr = &opts.Title
	}
	if strings.TrimSpace(opts.Owner) != "" {
		ownerPtr = &opts.Owner
	}
	if strings.TrimSpace(opts.Status) != "" {
		statusPtr = &opts.Status
	}
	if strings.TrimSpace(opts.Due) != "" {
		duePtr = &opts.Due
	}
	if strings.TrimSpace(opts.Tags) != "" {
		tagsPtr = &opts.Tags
	}
	task, err := a.projectService.UpdateTask(id, titlePtr, ownerPtr, statusPtr, duePtr, tagsPtr, opts.Progress)
	if err != nil {
		return err
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("updated: %s", task.Title),
	}
	return nil
}

func (a *Application) cmdProjectRemove(args []string) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: /project rm <id|title>")
	}
	if !a.ensureProjectService() {
		return fmt.Errorf("not logged in. Run /login <token> first")
	}
	id, target, err := a.resolveTaskTarget(args[0])
	if err != nil {
		return err
	}
	if err := a.projectService.RemoveTask(id); err != nil {
		return err
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("removed: %s", target),
	}
	return nil
}

func (a *Application) resolveTaskTarget(arg string) (int, string, error) {
	// Try as numeric ID first (for tasks).
	if id, err := strconv.Atoi(arg); err == nil {
		return id, fmt.Sprintf("#%d", id), nil
	}
	// Otherwise match by title (for milestones/support).
	snap, err := a.projectService.Snapshot()
	if err != nil {
		return 0, "", fmt.Errorf("cannot resolve target: %v", err)
	}
	lower := strings.ToLower(strings.TrimSpace(arg))
	for _, t := range snap.Tasks {
		if strings.ToLower(t.Title) == lower {
			return t.ID, t.Title, nil
		}
	}
	return 0, "", fmt.Errorf("no item found matching %q", arg)
}

func (a *Application) cmdProjectOverview(args []string) error {
	if !a.ensureProjectService() {
		return fmt.Errorf("not logged in. Run /login <token> first")
	}
	opts, err := parseOverviewOptions(args)
	if err != nil {
		return err
	}
	if opts.Phase == "" && opts.Owner == "" && opts.Repo == "" && opts.Branch == "" {
		return fmt.Errorf("usage: /project overview --phase X --owner Y --repo URL --branch B")
	}
	ov, err := a.projectService.UpdateOverview(opts.Phase, opts.Owner, opts.Repo, opts.Branch)
	if err != nil {
		return err
	}
	lines := []string{"overview updated:"}
	if ov.Phase != "" {
		lines = append(lines, "  phase: "+ov.Phase)
	}
	if ov.Owner != "" {
		lines = append(lines, "  owner: "+ov.Owner)
	}
	if ov.Repo != "" {
		lines = append(lines, "  repo: "+ov.Repo)
	}
	if ov.Branch != "" {
		lines = append(lines, "  branch: "+ov.Branch)
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: strings.Join(lines, "\n"),
	}
	return nil
}

type overviewEdit struct {
	Phase  string
	Owner  string
	Repo   string
	Branch string
}

func parseOverviewOptions(args []string) (overviewEdit, error) {
	var edit overviewEdit
	for i := 0; i < len(args); i++ {
		switch strings.ToLower(strings.TrimSpace(args[i])) {
		case "--phase":
			if i+1 >= len(args) {
				return edit, fmt.Errorf("--phase requires a value")
			}
			i++
			edit.Phase = strings.TrimSpace(args[i])
		case "--owner":
			if i+1 >= len(args) {
				return edit, fmt.Errorf("--owner requires a value")
			}
			i++
			edit.Owner = strings.TrimSpace(args[i])
		case "--repo":
			if i+1 >= len(args) {
				return edit, fmt.Errorf("--repo requires a value")
			}
			i++
			edit.Repo = strings.TrimSpace(args[i])
		case "--branch":
			if i+1 >= len(args) {
				return edit, fmt.Errorf("--branch requires a value")
			}
			i++
			edit.Branch = strings.TrimSpace(args[i])
		default:
			return edit, fmt.Errorf("unknown flag %q", args[i])
		}
	}
	return edit, nil
}

type projectTaskEdit struct {
	Title    string
	Owner    string
	Status   string
	Due      string
	Tags     string
	Progress *int
}

func (e projectTaskEdit) HasUpdates() bool {
	return strings.TrimSpace(e.Title) != "" || strings.TrimSpace(e.Owner) != "" || strings.TrimSpace(e.Status) != "" || strings.TrimSpace(e.Due) != "" || strings.TrimSpace(e.Tags) != "" || e.Progress != nil
}

func canonicalSection(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "task", "tasks":
		return "tasks"
	case "milestone", "milestones":
		return "milestones"
	case "support":
		return "support"
	default:
		raw = strings.TrimSpace(raw)
		if raw == "" {
			return "tasks"
		}
		return raw
	}
}

func splitProjectArgs(raw string) ([]string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	var args []string
	var current strings.Builder
	var quote rune
	escaped := false

	flush := func() {
		if current.Len() == 0 {
			return
		}
		args = append(args, current.String())
		current.Reset()
	}

	for _, r := range raw {
		switch {
		case escaped:
			current.WriteRune(r)
			escaped = false
		case r == '\\' && quote != '\'':
			escaped = true
		case quote != 0:
			if r == quote {
				quote = 0
			} else {
				current.WriteRune(r)
			}
		case r == '"' || r == '\'':
			quote = r
		case r == ' ' || r == '\t' || r == '\n':
			flush()
		default:
			current.WriteRune(r)
		}
	}
	if escaped {
		current.WriteRune('\\')
	}
	if quote != 0 {
		return nil, fmt.Errorf("unterminated quote")
	}
	flush()
	return args, nil
}

func parseProjectTaskOptions(args []string, _ bool) (projectTaskEdit, error) {
	var edit projectTaskEdit
	for i := 0; i < len(args); i++ {
		switch strings.ToLower(strings.TrimSpace(args[i])) {
		case "--title":
			if i+1 >= len(args) {
				return edit, fmt.Errorf("--title requires a value")
			}
			i++
			edit.Title = strings.TrimSpace(args[i])
		case "--owner":
			if i+1 >= len(args) {
				return edit, fmt.Errorf("--owner requires a value")
			}
			i++
			edit.Owner = strings.TrimSpace(args[i])
		case "--status":
			if i+1 >= len(args) {
				return edit, fmt.Errorf("--status requires a value")
			}
			i++
			edit.Status = strings.TrimSpace(args[i])
		case "--due":
			if i+1 >= len(args) {
				return edit, fmt.Errorf("--due requires a value")
			}
			i++
			edit.Due = strings.TrimSpace(args[i])
		case "--tags":
			if i+1 >= len(args) {
				return edit, fmt.Errorf("--tags requires a value")
			}
			i++
			edit.Tags = strings.TrimSpace(args[i])
		case "--progress":
			if i+1 >= len(args) {
				return edit, fmt.Errorf("--progress requires a value")
			}
			i++
			pct, ok := parsePercent(args[i])
			if !ok {
				return edit, fmt.Errorf("invalid progress %q", args[i])
			}
			edit.Progress = &pct
		default:
			return edit, fmt.Errorf("unknown flag %q", args[i])
		}
	}
	return edit, nil
}

func collectProjectStatus(workDir string) (model.ProjectStatusView, error) {
	root, err := runProjectGit(workDir, "rev-parse", "--show-toplevel")
	if err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "not a git repository") {
			absRoot, absErr := filepath.Abs(workDir)
			if absErr == nil {
				workDir = absRoot
			}
			return model.ProjectStatusView{
				Name:    filepath.Base(workDir),
				Root:    workDir,
				Branch:  "-",
				Summary: "not a git repository",
			}, nil
		}
		return model.ProjectStatusView{}, err
	}

	branch, err := runProjectGit(root, "symbolic-ref", "--short", "HEAD")
	if err != nil {
		branch = "detached"
	}

	shortStatus, err := runProjectGit(root, "status", "--short")
	if err != nil {
		return model.ProjectStatusView{}, err
	}

	modified, staged, untracked := parseShortStatus(shortStatus)
	changed, docs, code, tests := classifyChangedFiles(shortStatus)
	ahead, behind := parseAheadBehind(root)
	summary, dirty := formatProjectSummary(modified, staged, untracked, ahead, behind)

	return model.ProjectStatusView{
		Name:      filepath.Base(root),
		Root:      root,
		Branch:    branch,
		Summary:   summary,
		Dirty:     dirty,
		Modified:  modified,
		Staged:    staged,
		Untracked: untracked,
		Ahead:     ahead,
		Behind:    behind,
		Changed:   changed,
		Docs:      docs,
		Code:      code,
		Tests:     tests,
	}, nil
}

func parseShortStatus(status string) (modified, staged, untracked int) {
	for _, line := range strings.Split(status, "\n") {
		line = strings.TrimRight(line, "\r")
		if strings.TrimSpace(line) == "" {
			continue
		}
		if strings.HasPrefix(line, "??") {
			untracked++
			continue
		}
		if len(line) >= 1 && line[0] != ' ' {
			staged++
		}
		if len(line) >= 2 && line[1] != ' ' {
			modified++
		}
	}
	return modified, staged, untracked
}

func classifyChangedFiles(status string) (changed, docs, code, tests int) {
	for _, line := range strings.Split(status, "\n") {
		line = strings.TrimRight(line, "\r")
		if strings.TrimSpace(line) == "" {
			continue
		}
		path := parseStatusPath(line)
		if path == "" {
			continue
		}
		changed++
		switch {
		case isTestPath(path):
			tests++
		case isDocPath(path):
			docs++
		default:
			code++
		}
	}
	return changed, docs, code, tests
}

func parseStatusPath(line string) string {
	if len(line) <= 3 {
		return ""
	}
	path := strings.TrimSpace(line[3:])
	if idx := strings.LastIndex(path, " -> "); idx >= 0 {
		path = strings.TrimSpace(path[idx+4:])
	}
	return path
}

func isDocPath(path string) bool {
	lower := strings.ToLower(path)
	return strings.HasPrefix(lower, "docs/") ||
		strings.HasSuffix(lower, ".md") ||
		strings.HasSuffix(lower, ".txt") ||
		strings.HasSuffix(lower, ".rst")
}

func isTestPath(path string) bool {
	lower := strings.ToLower(path)
	return strings.Contains(lower, "/testdata/") ||
		strings.HasSuffix(lower, "_test.go") ||
		strings.HasPrefix(lower, "test/") ||
		strings.HasPrefix(lower, "tests/")
}

func parseAheadBehind(workDir string) (ahead, behind int) {
	out, err := runProjectGit(workDir, "rev-list", "--left-right", "--count", "@{upstream}...HEAD")
	if err != nil {
		return 0, 0
	}
	fields := strings.Fields(out)
	if len(fields) != 2 {
		return 0, 0
	}
	fmt.Sscanf(fields[0], "%d", &behind)
	fmt.Sscanf(fields[1], "%d", &ahead)
	return ahead, behind
}

func formatProjectSummary(modified, staged, untracked, ahead, behind int) (string, bool) {
	parts := make([]string, 0, 5)
	dirty := modified > 0 || staged > 0 || untracked > 0
	if staged > 0 {
		parts = append(parts, fmt.Sprintf("%d staged", staged))
	}
	if modified > 0 {
		parts = append(parts, fmt.Sprintf("%d modified", modified))
	}
	if untracked > 0 {
		parts = append(parts, fmt.Sprintf("%d untracked", untracked))
	}
	if ahead > 0 {
		parts = append(parts, fmt.Sprintf("ahead %d", ahead))
	}
	if behind > 0 {
		parts = append(parts, fmt.Sprintf("behind %d", behind))
	}
	if len(parts) == 0 {
		return "clean working tree", false
	}
	return strings.Join(parts, " · "), dirty
}

func normalizeTaskStatus(status string) string {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "done":
		return "done"
	case "block", "blocked":
		return "block"
	case "todo", "pending", "not_started", "not-started":
		return "todo"
	case "doing", "in_progress", "in-progress":
		return "doing"
	default:
		return ""
	}
}

func fallbackOverview(status model.ProjectStatusView) []string {
	lines := []string{
		fmt.Sprintf("  repo: %s", status.Name),
		fmt.Sprintf("  root: %s", status.Root),
	}
	return lines
}

func progressBar(pct, width int) string {
	if width <= 0 {
		width = 10
	}
	if pct < 0 {
		pct = 0
	}
	if pct > 100 {
		pct = 100
	}
	filled := (pct * width) / 100
	if pct > 0 && filled == 0 {
		filled = 1
	}
	if filled > width {
		filled = width
	}
	cells := make([]string, 0, width)
	for i := 0; i < filled; i++ {
		cells = append(cells, "■")
	}
	for i := filled; i < width; i++ {
		cells = append(cells, "□")
	}
	return strings.Join(cells, "")
}

func renderProjectCard(card projectCard) string {
	sections := []string{
		applyProjectStyle("[ OVERVIEW ]", "252", true),
	}
	sections = append(sections, card.Overview...)
	if summary := strings.TrimSpace(card.Status.Summary); summary != "" {
		sections = append(sections, "  git: "+summary)
	}
	sections = append(sections, "", applyProjectStyle("[🚀 MILESTONE ]", "252", true))
	sections = append(sections, renderMilestoneLines(card.Milestones)...)
	sections = append(sections, "", applyProjectStyle("[ TASKS ]", "252", true))
	sections = append(sections, renderTaskLines(card.Tasks)...)
	sections = append(sections, "", applyProjectStyle("[ SUPPORT ]", "252", true))
	sections = append(sections, renderSupportLines(card.Support)...)

	return strings.Join(sections, "\n")
}

func renderMilestoneLines(milestones []projectTask) []string {
	if len(milestones) == 0 {
		return []string{"  (none)"}
	}
	tagsWidth := 0
	for _, m := range milestones {
		if l := len(m.Tags); l > tagsWidth {
			tagsWidth = l
		}
	}
	if tagsWidth > 20 {
		tagsWidth = 20
	}

	fixedWidth := 2 + 12 + 1 + 4
	if tagsWidth > 0 {
		fixedWidth += 2 + tagsWidth
	}
	titleWidth := projectColumnWidth - fixedWidth
	if titleWidth < 12 {
		titleWidth = 12
	}

	lines := make([]string, 0, len(milestones))
	for _, m := range milestones {
		bar := progressBar(m.Progress, 10)
		title := truncateStr(m.Title, titleWidth)
		tags := truncateStr(m.Tags, tagsWidth)
		var line string
		if tagsWidth > 0 {
			line = fmt.Sprintf("  [%s] %3d%%  %-*s  %s", bar, m.Progress, tagsWidth, tags, title)
		} else {
			line = fmt.Sprintf("  [%s] %3d%%  %s", bar, m.Progress, title)
		}
		lines = append(lines, applyProjectColor(line, "magenta"))
	}
	return lines
}

func renderTaskLines(tasks []projectTask) []string {
	if len(tasks) == 0 {
		return []string{"  (none)"}
	}
	// Sort by progress ascending — less progress on top, 100% at bottom.
	sort.SliceStable(tasks, func(i, j int) bool {
		return tasks[i].Progress < tasks[j].Progress
	})
	ownerWidth := 0
	tagsWidth := 0
	for _, t := range tasks {
		if l := len(t.Owner); l > ownerWidth {
			ownerWidth = l
		}
		if l := len(t.Tags); l > tagsWidth {
			tagsWidth = l
		}
	}
	if ownerWidth < 5 {
		ownerWidth = 5
	}
	if tagsWidth > 20 {
		tagsWidth = 20
	}

	fixedWidth := 2 + 5 + 1 + 1 + 12 + 1 + 4 + 3 + ownerWidth
	if tagsWidth > 0 {
		fixedWidth += 2 + tagsWidth
	}
	titleWidth := projectColumnWidth - fixedWidth
	if titleWidth < 12 {
		titleWidth = 12
	}
	prefixWidth := fixedWidth + 2

	lines := make([]string, 0, len(tasks))
	for _, task := range tasks {
		marker := taskStatusMarker(task.Status, task.Progress)
		bar := progressBar(task.Progress, 10)
		tags := truncateStr(task.Tags, tagsWidth)
		owner := fmt.Sprintf("%-*s", ownerWidth, task.Owner)
		var tagsCol string
		if tagsWidth > 0 {
			tagsCol = fmt.Sprintf("  %-*s", tagsWidth, tags)
		}

		coloredPart := fmt.Sprintf("#%-3s %s [%s] %3d%%", task.ID, marker, bar, task.Progress)
		titleLines := wrapProjectText(task.Title, titleWidth)
		if len(titleLines) == 0 {
			titleLines = []string{""}
		}
		dimPrefix := "  " + owner + tagsCol + "  "
		firstLine := "  " + coloredPart + dimPrefix + titleLines[0]
		switch {
		case task.Progress >= 100 || normalizeTaskStatus(task.Status) == "done":
			firstLine = "  " + applyProjectColor(coloredPart, "green") + dimPrefix + titleLines[0]
		case task.Progress > 0:
			firstLine = "  " + applyProjectColor(coloredPart, "yellow") + dimPrefix + titleLines[0]
		}
		lines = append(lines, firstLine)
		for _, continuation := range titleLines[1:] {
			lines = append(lines, strings.Repeat(" ", prefixWidth)+continuation)
		}
	}
	return lines
}

func renderSupportLines(items []string) []string {
	if len(items) == 0 {
		return []string{"  (none)"}
	}
	lines := make([]string, 0, len(items))
	for _, item := range items {
		lines = append(lines, "  ✓ "+item)
	}
	return lines
}

const projectColumnWidth = 100

func truncateStr(s string, maxWidth int) string {
	if maxWidth <= 0 {
		return ""
	}
	runes := []rune(s)
	if len(runes) <= maxWidth {
		return s
	}
	if maxWidth <= 3 {
		return string(runes[:maxWidth])
	}
	return string(runes[:maxWidth-3]) + "..."
}

func wrapProjectText(s string, width int) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	if width <= 0 {
		return []string{s}
	}

	words := strings.Fields(s)
	if len(words) == 0 {
		return []string{s}
	}

	lines := make([]string, 0, len(words))
	current := words[0]
	for _, word := range words[1:] {
		candidate := current + " " + word
		if len([]rune(candidate)) <= width {
			current = candidate
			continue
		}
		lines = append(lines, current)
		current = word
	}
	lines = append(lines, current)
	return lines
}

func applyProjectColor(text, color string) string {
	return applyProjectStyle(text, color, false)
}

func applyProjectStyle(text, color string, bold bool) string {
	code, ok := projectColorCode(color)
	if !ok && !bold {
		return text
	}
	switch {
	case ok && bold:
		return "\x1b[1;38;5;" + code + "m" + text + "\x1b[0m"
	case ok:
		return "\x1b[38;5;" + code + "m" + text + "\x1b[0m"
	case bold:
		return "\x1b[1m" + text + "\x1b[0m"
	default:
		return text
	}
}

func projectColorCode(color string) (string, bool) {
	switch strings.ToLower(strings.TrimSpace(color)) {
	case "green":
		return "34", true
	case "yellow":
		return "220", true
	case "magenta":
		return "201", true
	case "white":
		return "252", true
	case "gray", "grey":
		return "244", true
	}
	color = strings.TrimSpace(color)
	for _, r := range color {
		if r < '0' || r > '9' {
			return "", false
		}
	}
	if color == "" {
		return "", false
	}
	return color, true
}

const (
	projectSectionTasks = "tasks"
)

func parsePercent(value string) (int, bool) {
	value = strings.TrimSpace(strings.TrimSuffix(value, "%"))
	if value == "" {
		return 0, false
	}
	var pct int
	if _, err := fmt.Sscanf(value, "%d", &pct); err != nil {
		return 0, false
	}
	if pct < 0 {
		pct = 0
	}
	if pct > 100 {
		pct = 100
	}
	return pct, true
}

func taskStatusMarker(status string, progress int) string {
	normalized := normalizeTaskStatus(status)
	if normalized == "done" || progress >= 100 {
		return "✓"
	}
	if normalized == "block" {
		return "!"
	}
	if normalized == "todo" || progress <= 0 {
		return "○"
	}
	return "▶"
}

func stateLine(status model.ProjectStatusView) string {
	state := "clean"
	if status.Dirty {
		state = "dirty"
	}
	parts := []string{
		fmt.Sprintf("%s · branch %s", state, status.Branch),
		fmt.Sprintf("staged %d", status.Staged),
		fmt.Sprintf("modified %d", status.Modified),
		fmt.Sprintf("untracked %d", status.Untracked),
	}
	if status.Ahead > 0 || status.Behind > 0 {
		parts = append(parts, fmt.Sprintf("ahead %d", status.Ahead), fmt.Sprintf("behind %d", status.Behind))
	}
	return strings.Join(parts, "  ")
}
