package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vigo999/mindspore-cli/configs"
)

func TestModelPresetCredentialEndpoint(t *testing.T) {
	mux := NewMux(nil, []configs.TokenEntry{
		{Token: "t1", User: "alice", Role: "user"},
	}, []configs.ModelPresetCredential{
		{ID: "kimi-k2.5-free", APIKey: "server-key"},
	})

	req := httptest.NewRequest(http.MethodGet, "/model-presets/kimi-k2.5-free/credential", nil)
	req.Header.Set("Authorization", "Bearer t1")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if got, want := rec.Code, http.StatusOK; got != want {
		t.Fatalf("status code = %d, want %d", got, want)
	}
	var body struct {
		ID     string `json:"id"`
		APIKey string `json:"api_key"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if got, want := body.ID, "kimi-k2.5-free"; got != want {
		t.Fatalf("id = %q, want %q", got, want)
	}
	if got, want := body.APIKey, "server-key"; got != want {
		t.Fatalf("api key = %q, want %q", got, want)
	}
}

func TestModelPresetCredentialEndpointUnknownPreset(t *testing.T) {
	mux := NewMux(nil, []configs.TokenEntry{
		{Token: "t1", User: "alice", Role: "user"},
	}, nil)

	req := httptest.NewRequest(http.MethodGet, "/model-presets/kimi-k2.5-free/credential", nil)
	req.Header.Set("Authorization", "Bearer t1")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if got, want := rec.Code, http.StatusNotFound; got != want {
		t.Fatalf("status code = %d, want %d", got, want)
	}
}
