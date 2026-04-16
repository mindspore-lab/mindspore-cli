package app

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mindspore-lab/mindspore-cli/agent/loop"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/permission"
)

func TestToolSystemE2E_GlobFindsRootTrainPyInSingleToolCall(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("MSCLI_PROVIDER", "openai-completion")
	t.Setenv("MSCLI_API_KEY", "token")
	t.Setenv("MSCLI_MODEL", "gpt-4o-mini")

	workDir := t.TempDir()
	t.Chdir(workDir)
	if err := os.WriteFile(filepath.Join(workDir, "train.py"), []byte("print('root')\n"), 0o644); err != nil {
		t.Fatalf("write root train.py: %v", err)
	}

	provider := &scriptedAppStreamProvider{
		responses: []*llm.CompletionResponse{
			{
				ToolCalls: []llm.ToolCall{{
					ID:   "call-glob-train",
					Type: "function",
					Function: llm.ToolCallFunc{
						Name:      "glob",
						Arguments: json.RawMessage(`{"pattern":"**/train.py","path":"."}`),
					},
				}},
				FinishReason: llm.FinishToolCalls,
			},
			{
				Content:      "yes, train.py exists in the current directory",
				FinishReason: llm.FinishStop,
			},
		},
	}

	origBuildProvider := buildProvider
	buildProvider = func(resolved llm.ResolvedConfig) (llm.Provider, error) {
		return provider, nil
	}
	defer func() { buildProvider = origBuildProvider }()

	app, err := Wire(BootstrapConfig{})
	if err != nil {
		t.Fatalf("Wire() error = %v", err)
	}

	events, err := app.Engine.Run(loop.Task{
		ID:          "tool-e2e-train-py",
		Description: "当前目录是否有train.py文件",
	})
	if err != nil {
		t.Fatalf("Engine.Run() error = %v", err)
	}

	var sawGlob bool
	for _, ev := range events {
		if ev.Type != loop.EventToolGlob {
			continue
		}
		sawGlob = true
		if !strings.Contains(ev.Message, "train.py") {
			t.Fatalf("glob event message = %q, want train.py", ev.Message)
		}
	}
	if !sawGlob {
		t.Fatal("expected glob tool event, got none")
	}
}

func TestToolSystemE2E_ExposesListDirAndShellGuardrailsToModel(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("MSCLI_PROVIDER", "openai-completion")
	t.Setenv("MSCLI_API_KEY", "token")
	t.Setenv("MSCLI_MODEL", "gpt-4o-mini")

	workDir := t.TempDir()
	t.Chdir(workDir)

	provider := &captureStreamProvider{}
	origBuildProvider := buildProvider
	buildProvider = func(resolved llm.ResolvedConfig) (llm.Provider, error) {
		return provider, nil
	}
	defer func() { buildProvider = origBuildProvider }()

	app, err := Wire(BootstrapConfig{})
	if err != nil {
		t.Fatalf("Wire() error = %v", err)
	}

	_, err = app.Engine.Run(loop.Task{
		ID:          "tool-e2e-structure-summary",
		Description: "为我总结当前目录的代码结构",
	})
	if err != nil {
		t.Fatalf("Engine.Run() error = %v", err)
	}

	if provider.lastReq == nil {
		t.Fatal("expected provider to receive completion request")
	}

	var sawListDir bool
	var shellDescription string
	for _, tool := range provider.lastReq.Tools {
		switch tool.Function.Name {
		case "list_dir":
			sawListDir = true
		case "shell":
			shellDescription = tool.Function.Description
		}
	}
	if !sawListDir {
		t.Fatal("expected list_dir to be exposed to the model")
	}
	if !strings.Contains(shellDescription, "Do not use shell redirection, heredoc, tee, sed -i, or similar patterns to create or edit files") {
		t.Fatalf("shell description = %q, want file-authoring guardrail", shellDescription)
	}
}

func TestToolSystemE2E_DenyWriteBlocksShellFileAuthoringFallback(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("MSCLI_PROVIDER", "openai-completion")
	t.Setenv("MSCLI_API_KEY", "token")
	t.Setenv("MSCLI_MODEL", "gpt-4o-mini")

	workDir := t.TempDir()
	t.Chdir(workDir)

	provider := &scriptedAppStreamProvider{
		responses: []*llm.CompletionResponse{
			{
				ToolCalls: []llm.ToolCall{{
					ID:   "call-shell-write",
					Type: "function",
					Function: llm.ToolCallFunc{
						Name:      "shell",
						Arguments: json.RawMessage("{\"command\":\"cat <<'EOF' > note.md\\nhello\\nEOF\"}"),
					},
				}},
				FinishReason: llm.FinishToolCalls,
			},
			{
				Content:      "unable to proceed without file-authoring permission",
				FinishReason: llm.FinishStop,
			},
		},
	}

	origBuildProvider := buildProvider
	buildProvider = func(resolved llm.ResolvedConfig) (llm.Provider, error) {
		return provider, nil
	}
	defer func() { buildProvider = origBuildProvider }()

	app, err := Wire(BootstrapConfig{})
	if err != nil {
		t.Fatalf("Wire() error = %v", err)
	}

	if svc, ok := app.permService.(*permission.DefaultPermissionService); ok {
		svc.Grant("write", permission.PermissionDeny)
	}

	events, err := app.Engine.Run(loop.Task{
		ID:          "tool-e2e-write-denied",
		Description: "帮我写一个markdown文档",
	})
	if err == nil {
		t.Fatal("Engine.Run() error = nil, want shell file authoring denial")
	}
	if !strings.Contains(err.Error(), "shell file authoring is blocked") {
		t.Fatalf("Engine.Run() error = %v, want shell file authoring blocked", err)
	}

	if _, statErr := os.Stat(filepath.Join(workDir, "note.md")); !os.IsNotExist(statErr) {
		t.Fatalf("note.md should not be created after shell fallback denial, stat err = %v", statErr)
	}

	for _, ev := range events {
		if ev.Type == loop.EventToolError && !strings.Contains(ev.Message, "shell file authoring is blocked") {
			t.Fatalf("tool error = %q, want shell file authoring blocked", ev.Message)
		}
	}
}
