package loop

import (
	"strings"
	"testing"
)

func TestDefaultSystemPromptIncludesWriteArgumentValidationRules(t *testing.T) {
	prompt := DefaultSystemPrompt()

	if !strings.Contains(prompt, `verify arguments contain BOTH "path" and "content"`) {
		t.Fatalf("DefaultSystemPrompt() missing write arg validation rule: %q", prompt)
	}
	if !strings.Contains(prompt, "Never call write with empty JSON arguments ({})") {
		t.Fatalf("DefaultSystemPrompt() missing empty-args guard: %q", prompt)
	}
	if !strings.Contains(prompt, "Prefer dedicated tools over shell whenever a dedicated tool can do the job") {
		t.Fatalf("DefaultSystemPrompt() missing dedicated-tool preference: %q", prompt)
	}
	if !strings.Contains(prompt, "If the user rejects a tool permission request, do not try to bypass that denial with another tool or shell pattern") {
		t.Fatalf("DefaultSystemPrompt() missing rejection bypass guardrail: %q", prompt)
	}
	if !strings.Contains(prompt, "ask the user how to proceed") {
		t.Fatalf("DefaultSystemPrompt() missing ask-user follow-up guidance after rejection: %q", prompt)
	}

	for _, phrase := range []string{
		"To read files use read instead of cat, head, tail, or sed",
		"To edit files use edit instead of sed or awk",
		"To create files use write instead of cat with heredoc or echo redirection",
		"To search for files use glob instead of find or ls",
		"To search the content of files, use grep instead of grep or rg",
		"Reserve using the shell exclusively for system commands and terminal operations that require shell execution",
		"Do not use shell redirection, heredoc, tee, or similar shell patterns to create or edit files",
	} {
		if strings.Contains(prompt, phrase) {
			t.Fatalf("DefaultSystemPrompt() should keep tool-selection guidance in tool descriptions, found %q in prompt: %q", phrase, prompt)
		}
	}
}
