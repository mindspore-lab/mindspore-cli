package fs

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResolveSafePathAllowsRelativePathInsideWorkDir(t *testing.T) {
	workDir := t.TempDir()

	got, err := resolveSafePath(workDir, "nested/file.txt")
	if err != nil {
		t.Fatalf("resolveSafePath returned error: %v", err)
	}

	want := filepath.Join(workDir, "nested", "file.txt")
	if got != want {
		t.Fatalf("resolveSafePath = %q, want %q", got, want)
	}
}

func TestResolveSafePathRejectsEscapeFromWorkDir(t *testing.T) {
	workDir := t.TempDir()

	_, err := resolveSafePath(workDir, "../outside.txt")
	if err == nil {
		t.Fatal("resolveSafePath returned nil error, want path escape error")
	}
	if !strings.Contains(err.Error(), "path escapes working directory") {
		t.Fatalf("resolveSafePath error = %q, want path escape error", err)
	}
}

func TestResolveSafePathAllowsAbsolutePathInsideWorkDir(t *testing.T) {
	workDir := t.TempDir()
	insidePath := filepath.Join(workDir, "nested", "inside.txt")

	got, err := resolveSafePath(workDir, insidePath)
	if err != nil {
		t.Fatalf("resolveSafePath returned error: %v", err)
	}
	if got != insidePath {
		t.Fatalf("resolveSafePath = %q, want %q", got, insidePath)
	}
}

func TestResolveSafePathRejectsAbsolutePathOutsideAllowedRoots(t *testing.T) {
	workDir := t.TempDir()
	outsidePath := filepath.Join(filepath.Dir(workDir), "outside.txt")

	_, err := resolveSafePath(workDir, outsidePath)
	if err == nil {
		t.Fatal("resolveSafePath returned nil error, want absolute path rejection")
	}
	if !strings.Contains(err.Error(), "absolute paths are not allowed") {
		t.Fatalf("resolveSafePath error = %q, want absolute path rejection", err)
	}
}

func TestResolveSafePathAllowsConfiguredHomeDirectories(t *testing.T) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("UserHomeDir returned error: %v", err)
	}

	workDir := t.TempDir()
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "tilde skills root",
			input: "~/.mscli/skills",
			want:  filepath.Join(homeDir, ".mscli", "skills"),
		},
		{
			name:  "tilde skills descendant",
			input: "~/.mscli/skills/demo/SKILL.md",
			want:  filepath.Join(homeDir, ".mscli", "skills", "demo", "SKILL.md"),
		},
		{
			name:  "absolute shared skills root",
			input: filepath.Join(homeDir, ".mscli", "mindspore-skills"),
			want:  filepath.Join(homeDir, ".mscli", "mindspore-skills"),
		},
		{
			name:  "absolute shared skills descendant",
			input: filepath.Join(homeDir, ".mscli", "mindspore-skills", "skills", "demo", "SKILL.md"),
			want:  filepath.Join(homeDir, ".mscli", "mindspore-skills", "skills", "demo", "SKILL.md"),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := resolveSafePath(workDir, tc.input)
			if err != nil {
				t.Fatalf("resolveSafePath returned error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("resolveSafePath = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestResolveSafePathRejectsGitMetadataPaths(t *testing.T) {
	workDir := t.TempDir()

	cases := []string{
		".git",
		filepath.Join(".git", "config"),
		filepath.Join("nested", ".git", "config"),
	}

	for _, input := range cases {
		t.Run(input, func(t *testing.T) {
			_, err := resolveSafePath(workDir, input)
			if err == nil {
				t.Fatalf("resolveSafePath(%q) returned nil error, want ignored path error", input)
			}
			if !strings.Contains(err.Error(), "path is ignored") {
				t.Fatalf("resolveSafePath(%q) error = %q, want ignored path error", input, err)
			}
		})
	}
}
