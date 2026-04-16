package shell

import (
	"strings"
	"testing"
)

func TestShellToolDescriptionIncludesDedicatedToolGuidance(t *testing.T) {
	t.Parallel()

	got := NewShellTool(nil).Description()

	for _, want := range []string{
		"Reserve using the shell exclusively for system commands and terminal operations that require shell execution",
		"Do not use shell redirection, heredoc, tee, sed -i, or similar patterns to create or edit files",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("shell description missing %q: %q", want, got)
		}
	}
}
