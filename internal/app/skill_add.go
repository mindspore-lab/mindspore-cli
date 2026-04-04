package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/mindspore-lab/mindspore-cli/integrations/skills"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

const localSkillsDisplayDir = "~/.mscli/skills/"
const skillAddCloneTimeout = 2 * time.Minute
const skillAddUsage = "/skill-add <path|git-url|owner/repo>"

type skillAddSourceKind int

const (
	skillAddSourceLocal skillAddSourceKind = iota
	skillAddSourceGitURL
	skillAddSourceGitHub
)

type skillAddSource struct {
	kind     skillAddSourceKind
	source   string
	display  string
	localDir string
}

type discoveredSkill struct {
	dir string
}

func (a *Application) cmdSkillAddInput(raw string) {
	if a.skillLoader == nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Skills not available."}
		return
	}
	if strings.TrimSpace(raw) == "" {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Usage: " + skillAddUsage}
		return
	}

	source, err := classifySkillAddSource(strings.TrimSpace(raw), a.WorkDir, a.skillsHomeDir)
	if err != nil {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "skill-add",
			Message:  fmt.Sprintf("Failed to add local skill: %v", err),
		}
		return
	}
	a.emitSkillAddLog(source.display)
	sourceRoot, cleanup, err := prepareSkillAddSource(source)
	if err != nil {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "skill-add",
			Message:  fmt.Sprintf("Failed to add local skill: %v", err),
		}
		return
	}
	defer cleanup()

	foundSkills, err := discoverSkills(sourceRoot)
	if err != nil {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "skill-add",
			Message:  fmt.Sprintf("Failed to add local skill: %v", err),
		}
		return
	}
	if len(foundSkills) == 0 {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "skill-add",
			Message:  fmt.Sprintf("Failed to add local skill: no skill.md found under %s", sourceRoot),
		}
		return
	}

	destRoot, err := a.localSkillsDir()
	if err != nil {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "skill-add",
			Message:  fmt.Sprintf("Failed to add local skill: %v", err),
		}
		return
	}
	if err := os.MkdirAll(destRoot, 0o755); err != nil {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "skill-add",
			Message:  fmt.Sprintf("Failed to add local skill: create destination: %v", err),
		}
		return
	}

	for _, found := range foundSkills {
		destDir := filepath.Join(destRoot, filepath.Base(found.dir))
		if !samePath(found.dir, destDir) {
			if err := copySkillDir(found.dir, destDir); err != nil {
				a.EventCh <- model.Event{
					Type:     model.ToolError,
					ToolName: "skill-add",
					Message:  fmt.Sprintf("Failed to add local skill: %v", err),
				}
				return
			}
		}
	}

	a.refreshSkillCatalog()
}

func (a *Application) emitSkillAddLog(name string) {
	if a == nil || a.EventCh == nil {
		return
	}
	name = strings.TrimSpace(name)
	if name == "" {
		name = "skill"
	}
	a.EventCh <- model.Event{
		Type:     model.ToolSkill,
		ToolName: "Skill add",
		Summary:  fmt.Sprintf("adding %s to %s", name, localSkillsDisplayDir),
	}
}

func (a *Application) emitAvailableSkills(includeUsage bool) {
	if a == nil || a.skillLoader == nil || a.EventCh == nil {
		return
	}

	summaries := a.skillLoader.List()
	if len(summaries) == 0 {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "No skills available."}
		return
	}

	msg := "Available skills:\n\n" + skills.FormatSummaries(summaries)
	if includeUsage {
		msg += "\nUsage: /skill <name> [request...] (omit request to start the skill immediately)"
	}
	a.EventCh <- model.Event{Type: model.AgentReply, Message: msg}
}

func (a *Application) localSkillsDir() (string, error) {
	home := strings.TrimSpace(a.skillsHomeDir)
	if home == "" {
		var err error
		home, err = os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("resolve home directory: %w", err)
		}
	}
	if strings.TrimSpace(home) == "" {
		return "", fmt.Errorf("home directory is required")
	}
	return filepath.Join(home, ".mscli", "skills"), nil
}

func classifySkillAddSource(rawPath, workDir, homeDir string) (skillAddSource, error) {
	path := trimQuotedPath(rawPath)
	if path == "" {
		return skillAddSource{}, fmt.Errorf("usage: %s", skillAddUsage)
	}
	if strings.Contains(strings.ToLower(path), "http") {
		return skillAddSource{kind: skillAddSourceGitURL, source: path, display: path}, nil
	}

	localPath, err := resolveLocalSkillRoot(path, workDir, homeDir)
	if err == nil {
		return skillAddSource{
			kind:     skillAddSourceLocal,
			source:   path,
			display:  filepath.Base(localPath),
			localDir: localPath,
		}, nil
	}

	if looksLikeGitHubShorthand(path) && errors.Is(err, os.ErrNotExist) {
		return skillAddSource{
			kind:    skillAddSourceGitHub,
			source:  "https://github.com/" + strings.TrimSuffix(path, ".git") + ".git",
			display: path,
		}, nil
	}

	return skillAddSource{}, err
}

func prepareSkillAddSource(source skillAddSource) (string, func(), error) {
	switch source.kind {
	case skillAddSourceLocal:
		return source.localDir, func() {}, nil
	case skillAddSourceGitURL, skillAddSourceGitHub:
		return cloneSkillAddSource(source.source)
	default:
		return "", func() {}, fmt.Errorf("unsupported skill source")
	}
}

func resolveLocalSkillRoot(rawPath, workDir, homeDir string) (string, error) {
	path, err := expandSkillAddPath(rawPath, workDir, homeDir)
	if err != nil {
		return "", err
	}

	info, err := os.Stat(path)
	if err != nil {
		return "", err
	}

	if info.IsDir() {
		return path, nil
	}
	if !strings.EqualFold(info.Name(), "skill.md") {
		return "", fmt.Errorf("skill path must point to a directory or skill.md")
	}
	return filepath.Dir(path), nil
}

func expandSkillAddPath(rawPath, workDir, homeDir string) (string, error) {
	path := trimQuotedPath(rawPath)
	if path == "" {
		return "", fmt.Errorf("usage: %s", skillAddUsage)
	}

	if strings.HasPrefix(path, "~"+string(os.PathSeparator)) {
		home := strings.TrimSpace(homeDir)
		if home == "" {
			var err error
			home, err = os.UserHomeDir()
			if err != nil {
				return "", fmt.Errorf("resolve home directory: %w", err)
			}
		}
		path = filepath.Join(home, strings.TrimPrefix(path, "~"+string(os.PathSeparator)))
	}

	if !filepath.IsAbs(path) {
		base := strings.TrimSpace(workDir)
		if base == "" {
			base = "."
		}
		path = filepath.Join(base, path)
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("resolve skill path: %w", err)
	}
	return absPath, nil
}

func looksLikeGitHubShorthand(path string) bool {
	path = strings.TrimSpace(path)
	if strings.Count(path, "/") != 1 {
		return false
	}
	if strings.HasPrefix(path, "/") || strings.HasPrefix(path, "~") {
		return false
	}
	if strings.HasPrefix(path, "./") || strings.HasPrefix(path, "../") {
		return false
	}
	parts := strings.SplitN(path, "/", 2)
	return parts[0] != "" && parts[1] != ""
}

func cloneSkillAddSource(repoURL string) (string, func(), error) {
	if _, err := exec.LookPath("git"); err != nil {
		return "", func() {}, fmt.Errorf("git is required to add skills from remote repositories")
	}

	tempRoot, err := os.MkdirTemp("", "mscli-skill-add-*")
	if err != nil {
		return "", func() {}, fmt.Errorf("create temp dir: %w", err)
	}

	cleanup := func() { _ = os.RemoveAll(tempRoot) }
	cloneDir := filepath.Join(tempRoot, "repo")

	ctx, cancel := context.WithTimeout(context.Background(), skillAddCloneTimeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "git", "clone", "--depth", "1", repoURL, cloneDir)
	output, err := cmd.CombinedOutput()
	if err != nil {
		cleanup()
		msg := strings.TrimSpace(string(output))
		if msg == "" {
			msg = err.Error()
		}
		return "", func() {}, fmt.Errorf("git clone failed: %s", msg)
	}

	return cloneDir, cleanup, nil
}

func discoverSkills(root string) ([]discoveredSkill, error) {
	found := map[string]discoveredSkill{}
	if err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.EqualFold(d.Name(), "skill.md") {
			return nil
		}

		dir := filepath.Dir(path)
		if _, ok := found[dir]; !ok {
			found[dir] = discoveredSkill{dir: dir}
		}
		return nil
	}); err != nil {
		return nil, err
	}

	result := make([]discoveredSkill, 0, len(found))
	for _, item := range found {
		result = append(result, item)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].dir < result[j].dir })
	return result, nil
}

func trimQuotedPath(path string) string {
	path = strings.TrimSpace(path)
	if len(path) >= 2 {
		if (path[0] == '"' && path[len(path)-1] == '"') || (path[0] == '\'' && path[len(path)-1] == '\'') {
			return strings.TrimSpace(path[1 : len(path)-1])
		}
	}
	return path
}

func samePath(left, right string) bool {
	leftAbs, err := filepath.Abs(left)
	if err != nil {
		leftAbs = left
	}
	rightAbs, err := filepath.Abs(right)
	if err != nil {
		rightAbs = right
	}
	return filepath.Clean(leftAbs) == filepath.Clean(rightAbs)
}

func copySkillDir(sourceDir, destDir string) error {
	if err := os.RemoveAll(destDir); err != nil {
		return fmt.Errorf("reset destination %s: %w", destDir, err)
	}

	return filepath.WalkDir(sourceDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		rel, err := filepath.Rel(sourceDir, path)
		if err != nil {
			return err
		}
		if d.IsDir() {
			info, err := d.Info()
			if err != nil {
				return err
			}
			return os.MkdirAll(filepath.Join(destDir, rel), info.Mode().Perm())
		}
		if d.Type()&fs.ModeSymlink != 0 {
			return fmt.Errorf("symlinked files are not supported: %s", path)
		}
		if !d.Type().IsRegular() {
			return fmt.Errorf("unsupported file type: %s", path)
		}
		targetRel := rel
		if strings.EqualFold(filepath.Base(rel), "skill.md") {
			targetRel = filepath.Join(filepath.Dir(rel), "SKILL.md")
			if filepath.Dir(rel) == "." {
				targetRel = "SKILL.md"
			}
		}
		return copySkillFile(path, filepath.Join(destDir, targetRel), d)
	})
}

func copySkillFile(sourcePath, destPath string, d fs.DirEntry) error {
	info, err := d.Info()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(destPath), 0o755); err != nil {
		return err
	}

	sourceFile, err := os.Open(sourcePath)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	destFile, err := os.OpenFile(destPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, info.Mode().Perm())
	if err != nil {
		return err
	}
	defer destFile.Close()

	if _, err := io.Copy(destFile, sourceFile); err != nil {
		return err
	}
	return nil
}
