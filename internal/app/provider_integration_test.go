package app

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/vigo999/ms-cli/configs"
	"github.com/vigo999/ms-cli/integrations/llm"
)

func TestWire_OpenAICompatibleDefaultRouting(t *testing.T) {
	t.Setenv("MSCLI_PROVIDER", "")
	t.Setenv("MSCLI_API_KEY", "mscli-token")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")
	t.Setenv("ANTHROPIC_API_KEY", "")

	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "cmpl-test",
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"role":    "assistant",
						"content": "ok",
					},
					"finish_reason": "stop",
				},
			},
		})
	}))
	defer server.Close()
	t.Setenv("MSCLI_BASE_URL", server.URL+"/v1")

	tempDir := t.TempDir()
	t.Chdir(tempDir)

	defaultCfg := configs.DefaultConfig()
	defaultCfg.Model.Provider = "openai-compatible"
	defaultCfg.Model.Model = "gpt-4o-mini"
	defaultCfg.Model.Key = ""
	configPath := filepath.Join(tempDir, "mscli.yaml")
	if err := configs.SaveToFile(defaultCfg, configPath); err != nil {
		t.Fatalf("SaveToFile() error = %v", err)
	}

	app, err := Wire(BootstrapConfig{ConfigPath: configPath})
	if err != nil {
		t.Fatalf("Wire() error = %v", err)
	}

	if got, want := app.provider.Name(), "openai-compatible"; got != want {
		t.Fatalf("provider.Name() = %q, want %q", got, want)
	}

	_, err = app.provider.Complete(context.Background(), &llm.CompletionRequest{
		Messages: []llm.Message{
			{Role: "user", Content: "ping"},
		},
	})
	if err != nil {
		t.Fatalf("provider.Complete() error = %v", err)
	}

	if got, want := gotPath, "/v1/chat/completions"; got != want {
		t.Fatalf("request path = %q, want %q", got, want)
	}
}
