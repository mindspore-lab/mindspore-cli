package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// NewJSONRequest builds an HTTP request with a JSON body and headers.
func NewJSONRequest(ctx context.Context, method, url string, headers map[string]string, body any) (*http.Request, error) {
	var payload io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("marshal json: %w", err)
		}
		payload = bytes.NewReader(data)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, payload)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	for key, value := range headers {
		if key == "" {
			continue
		}
		req.Header.Set(key, value)
	}

	return req, nil
}

// DoJSON sends an HTTP request using the provided client.
func DoJSON(client *http.Client, req *http.Request) (*http.Response, error) {
	if client == nil {
		client = http.DefaultClient
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}

	return resp, nil
}
