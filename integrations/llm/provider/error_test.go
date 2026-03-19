package provider

import (
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestParseOpenAIError_ParsesPayload(t *testing.T) {
	resp := &http.Response{
		StatusCode: http.StatusBadRequest,
		Body: io.NopCloser(strings.NewReader(`{
			"error": {
				"type": "invalid_request_error",
				"message": "bad request"
			}
		}`)),
	}

	err := parseOpenAIError(resp)
	if err == nil {
		t.Fatal("parseOpenAIError() error = nil, want error")
	}

	got := err.Error()
	if !strings.Contains(got, "invalid_request_error") || !strings.Contains(got, "bad request") {
		t.Fatalf("parseOpenAIError() = %q, want type and message", got)
	}
}

func TestParseAnthropicError_ParsesPayload(t *testing.T) {
	resp := &http.Response{
		StatusCode: http.StatusUnauthorized,
		Body: io.NopCloser(strings.NewReader(`{
			"error": {
				"type": "authentication_error",
				"message": "invalid x-api-key"
			}
		}`)),
	}

	err := parseAnthropicError(resp)
	if err == nil {
		t.Fatal("parseAnthropicError() error = nil, want error")
	}

	got := err.Error()
	if !strings.Contains(got, "authentication_error") || !strings.Contains(got, "invalid x-api-key") {
		t.Fatalf("parseAnthropicError() = %q, want type and message", got)
	}
}
