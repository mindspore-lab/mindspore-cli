package app

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vigo999/ms-cli/integrations/skills"
	"github.com/vigo999/ms-cli/ui/model"
)

func TestCmdSkillAddInputCopiesLocalSkillAndListsAvailableSkills(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	sourceRoot := t.TempDir()
	sourceDir := filepath.Join(sourceRoot, "demo-skill")
	if err := os.MkdirAll(sourceDir, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	skillBody := "---\nname: demo-skill\ndescription: Demo skill\n---\n\nbody"
	if err := os.WriteFile(filepath.Join(sourceDir, "SKILL.md"), []byte(skillBody), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	app := &Application{
		EventCh:       make(chan model.Event, 16),
		WorkDir:       sourceRoot,
		skillsHomeDir: home,
		skillLoader:   skills.NewLoader(filepath.Join(home, ".ms-cli", "skills")),
	}

	app.cmdSkillAddInput(sourceDir)

	destSkillFile := filepath.Join(home, ".ms-cli", "skills", "demo-skill", "SKILL.md")
	if _, err := os.Stat(destSkillFile); err != nil {
		t.Fatalf("expected copied skill at %s: %v", destSkillFile, err)
	}

	if _, err := app.skillLoader.Load("demo-skill"); err != nil {
		t.Fatalf("skillLoader.Load() error = %v", err)
	}

	adding := drainUntilEventType(t, app, model.ToolSkill)
	if got, want := adding.ToolName, "Skill add"; got != want {
		t.Fatalf("tool name = %q, want %q", got, want)
	}
	if got, want := adding.Summary, "adding demo-skill to ~/.ms-cli/skills/"; got != want {
		t.Fatalf("summary = %q, want %q", got, want)
	}

	ready := drainUntilEventType(t, app, model.ToolSkill)
	if got, want := ready.ToolName, "Skill ready: 1 available"; got != want {
		t.Fatalf("tool name = %q, want %q", got, want)
	}
	if !strings.Contains(ready.Summary, "demo-skill") {
		t.Fatalf("expected skill summary to include demo-skill, got %q", ready.Summary)
	}
}

func TestCmdSkillAddInputWithoutArgsPrintsUsage(t *testing.T) {
	app := &Application{
		EventCh:     make(chan model.Event, 4),
		skillLoader: skills.NewLoader(),
	}

	app.cmdSkillAddInput("")

	ev := drainUntilEventType(t, app, model.AgentReply)
	if got, want := ev.Message, "Usage: "+skillAddUsage; got != want {
		t.Fatalf("message = %q, want %q", got, want)
	}
}

func TestCmdSkillAddInputRecursivelyFindsNestedSkillMarkdownFiles(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	sourceRoot := t.TempDir()
	alphaDir := filepath.Join(sourceRoot, "skills", "alpha")
	betaDir := filepath.Join(sourceRoot, "skills", "beta")
	if err := os.MkdirAll(alphaDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(alpha) error = %v", err)
	}
	if err := os.MkdirAll(betaDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(beta) error = %v", err)
	}
	if err := os.WriteFile(filepath.Join(alphaDir, "skill.md"), []byte("---\nname: alpha\n---\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(alpha) error = %v", err)
	}
	if err := os.WriteFile(filepath.Join(betaDir, "SKILL.md"), []byte("---\nname: beta\n---\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(beta) error = %v", err)
	}

	app := &Application{
		EventCh:       make(chan model.Event, 16),
		WorkDir:       sourceRoot,
		skillsHomeDir: home,
		skillLoader:   skills.NewLoader(filepath.Join(home, ".ms-cli", "skills")),
	}

	app.cmdSkillAddInput(sourceRoot)

	if _, err := os.Stat(filepath.Join(home, ".ms-cli", "skills", "alpha", "SKILL.md")); err != nil {
		t.Fatalf("expected copied alpha skill: %v", err)
	}
	if _, err := os.Stat(filepath.Join(home, ".ms-cli", "skills", "beta", "SKILL.md")); err != nil {
		t.Fatalf("expected copied beta skill: %v", err)
	}
	if _, err := app.skillLoader.Load("alpha"); err != nil {
		t.Fatalf("skillLoader.Load(alpha) error = %v", err)
	}
	if _, err := app.skillLoader.Load("beta"); err != nil {
		t.Fatalf("skillLoader.Load(beta) error = %v", err)
	}

	adding := drainUntilEventType(t, app, model.ToolSkill)
	if got, want := adding.ToolName, "Skill add"; got != want {
		t.Fatalf("tool name = %q, want %q", got, want)
	}
	if got, want := adding.Summary, "adding "+filepath.Base(sourceRoot)+" to ~/.ms-cli/skills/"; got != want {
		t.Fatalf("summary = %q, want %q", got, want)
	}

	ready := drainUntilEventType(t, app, model.ToolSkill)
	if got, want := ready.ToolName, "Skill ready: 2 available"; got != want {
		t.Fatalf("tool name = %q, want %q", got, want)
	}
	if !strings.Contains(ready.Summary, "alpha") || !strings.Contains(ready.Summary, "beta") {
		t.Fatalf("expected ready summary to include alpha and beta, got %q", ready.Summary)
	}
}

func TestClassifySkillAddSource(t *testing.T) {
	workDir := t.TempDir()
	home := t.TempDir()

	localDir := filepath.Join(workDir, "local-skill")
	if err := os.MkdirAll(localDir, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	if err := os.WriteFile(filepath.Join(localDir, "SKILL.md"), []byte("body"), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	local, err := classifySkillAddSource(localDir, workDir, home)
	if err != nil {
		t.Fatalf("classify local error = %v", err)
	}
	if local.kind != skillAddSourceLocal {
		t.Fatalf("expected local kind, got %v", local.kind)
	}
	if got, want := local.display, "local-skill"; got != want {
		t.Fatalf("local display = %q, want %q", got, want)
	}

	github, err := classifySkillAddSource("openai/codex", workDir, home)
	if err != nil {
		t.Fatalf("classify github error = %v", err)
	}
	if github.kind != skillAddSourceGitHub {
		t.Fatalf("expected github kind, got %v", github.kind)
	}
	if got, want := github.source, "https://github.com/openai/codex.git"; got != want {
		t.Fatalf("github source = %q, want %q", got, want)
	}
	if got, want := github.display, "openai/codex"; got != want {
		t.Fatalf("github display = %q, want %q", got, want)
	}

	gitURL, err := classifySkillAddSource("https://example.com/repo.git", workDir, home)
	if err != nil {
		t.Fatalf("classify git url error = %v", err)
	}
	if gitURL.kind != skillAddSourceGitURL {
		t.Fatalf("expected git url kind, got %v", gitURL.kind)
	}
	if got, want := gitURL.display, "https://example.com/repo.git"; got != want {
		t.Fatalf("git url display = %q, want %q", got, want)
	}
}
