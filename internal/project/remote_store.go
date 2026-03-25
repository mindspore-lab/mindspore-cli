package project

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type RemoteStore struct {
	baseURL string
	token   string
	client  *http.Client
}

func NewRemoteStore(baseURL, token string) *RemoteStore {
	return &RemoteStore{
		baseURL: baseURL,
		token:   token,
		client:  &http.Client{Timeout: 10 * time.Second},
	}
}

func (r *RemoteStore) do(method, path string, body any) ([]byte, int, error) {
	var bodyReader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, 0, err
		}
		bodyReader = bytes.NewReader(data)
	}
	req, err := http.NewRequest(method, r.baseURL+path, bodyReader)
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Authorization", "Bearer "+r.token)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := r.client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, resp.StatusCode, err
	}
	return respBody, resp.StatusCode, nil
}

func (r *RemoteStore) GetSnapshot() (*Snapshot, error) {
	body, status, err := r.do("GET", "/project", nil)
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		return nil, fmt.Errorf("get project: server returned %d: %s", status, body)
	}
	var snap Snapshot
	if err := json.Unmarshal(body, &snap); err != nil {
		return nil, err
	}
	return &snap, nil
}

func (r *RemoteStore) CreateTask(section, title, owner, createdBy, due, tags string, progress *int) (*Task, error) {
	payload := map[string]any{
		"section": section,
		"title":   title,
	}
	if owner != "" {
		payload["owner"] = owner
	}
	if due != "" {
		payload["due"] = due
	}
	if tags != "" {
		payload["tags"] = tags
	}
	if progress != nil {
		payload["progress"] = *progress
	}
	body, status, err := r.do("POST", "/project/tasks", payload)
	if err != nil {
		return nil, err
	}
	if status != http.StatusCreated {
		return nil, fmt.Errorf("create task: server returned %d: %s", status, body)
	}
	var task Task
	if err := json.Unmarshal(body, &task); err != nil {
		return nil, err
	}
	return &task, nil
}

func (r *RemoteStore) UpdateTask(id int, title, owner, status, due, tags *string, progress *int) (*Task, error) {
	payload := map[string]any{}
	if title != nil {
		payload["title"] = *title
	}
	if owner != nil {
		payload["owner"] = *owner
	}
	if status != nil {
		payload["status"] = *status
	}
	if due != nil {
		payload["due"] = *due
	}
	if tags != nil {
		payload["tags"] = *tags
	}
	if progress != nil {
		payload["progress"] = *progress
	}
	body, code, err := r.do("PATCH", fmt.Sprintf("/project/tasks/%d", id), payload)
	if err != nil {
		return nil, err
	}
	if code != http.StatusOK {
		return nil, fmt.Errorf("update task: server returned %d: %s", code, body)
	}
	var task Task
	if err := json.Unmarshal(body, &task); err != nil {
		return nil, err
	}
	return &task, nil
}

func (r *RemoteStore) DeleteTask(id int) error {
	body, status, err := r.do("DELETE", fmt.Sprintf("/project/tasks/%d", id), nil)
	if err != nil {
		return err
	}
	if status != http.StatusOK {
		return fmt.Errorf("delete task: server returned %d: %s", status, body)
	}
	return nil
}

func (r *RemoteStore) UpdateOverview(phase, owner, repo, branch string) (*Overview, error) {
	payload := map[string]any{}
	if phase != "" {
		payload["phase"] = phase
	}
	if owner != "" {
		payload["owner"] = owner
	}
	if repo != "" {
		payload["repo"] = repo
	}
	if branch != "" {
		payload["branch"] = branch
	}
	body, status, err := r.do("PATCH", "/project/overview", payload)
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		return nil, fmt.Errorf("update overview: server returned %d: %s", status, body)
	}
	var ov Overview
	if err := json.Unmarshal(body, &ov); err != nil {
		return nil, err
	}
	return &ov, nil
}
