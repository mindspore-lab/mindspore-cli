package issues

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type createIssuePayload struct {
	Title string `json:"title"`
	Kind  Kind   `json:"kind"`
}

type updateIssueStatusPayload struct {
	Status string `json:"status"`
}

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

func (r *RemoteStore) CreateIssue(title string, kind Kind, reporter string) (*Issue, error) {
	_ = reporter
	body, status, err := r.do("POST", "/issues", createIssuePayload{Title: title, Kind: kind})
	if err != nil {
		return nil, err
	}
	if status != http.StatusCreated {
		return nil, fmt.Errorf("create issue: server returned %d: %s", status, body)
	}
	var issue Issue
	if err := json.Unmarshal(body, &issue); err != nil {
		return nil, err
	}
	return &issue, nil
}

func (r *RemoteStore) ListIssues(filterStatus string) ([]Issue, error) {
	path := "/issues"
	if filterStatus != "" {
		path += "?status=" + filterStatus
	}
	body, status, err := r.do("GET", path, nil)
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		return nil, fmt.Errorf("list issues: server returned %d: %s", status, body)
	}
	var issueList []Issue
	if err := json.Unmarshal(body, &issueList); err != nil {
		return nil, err
	}
	return issueList, nil
}

func (r *RemoteStore) GetIssue(id int) (*Issue, error) {
	body, status, err := r.do("GET", fmt.Sprintf("/issues/%d", id), nil)
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		return nil, fmt.Errorf("get issue: server returned %d: %s", status, body)
	}
	var issue Issue
	if err := json.Unmarshal(body, &issue); err != nil {
		return nil, err
	}
	return &issue, nil
}

func (r *RemoteStore) AddNote(issueID int, author, content string) (*Note, error) {
	body, status, err := r.do("POST", fmt.Sprintf("/issues/%d/notes", issueID), map[string]string{"content": content})
	if err != nil {
		return nil, err
	}
	if status != http.StatusCreated {
		return nil, fmt.Errorf("add issue note: server returned %d: %s", status, body)
	}
	var note Note
	if err := json.Unmarshal(body, &note); err != nil {
		return nil, err
	}
	return &note, nil
}

func (r *RemoteStore) ListNotes(issueID int) ([]Note, error) {
	body, status, err := r.do("GET", fmt.Sprintf("/issues/%d/notes", issueID), nil)
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		return nil, fmt.Errorf("list issue notes: server returned %d: %s", status, body)
	}
	var notes []Note
	if err := json.Unmarshal(body, &notes); err != nil {
		return nil, err
	}
	return notes, nil
}

func (r *RemoteStore) ListActivity(issueID int) ([]Activity, error) {
	body, status, err := r.do("GET", fmt.Sprintf("/issues/%d/activity", issueID), nil)
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		return nil, fmt.Errorf("list issue activity: server returned %d: %s", status, body)
	}
	var acts []Activity
	if err := json.Unmarshal(body, &acts); err != nil {
		return nil, err
	}
	return acts, nil
}

func (r *RemoteStore) ClaimIssue(id int, lead string) (*Issue, error) {
	body, status, err := r.do("POST", fmt.Sprintf("/issues/%d/claim", id), nil)
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		return nil, fmt.Errorf("claim issue: server returned %d: %s", status, body)
	}
	var issue Issue
	if err := json.Unmarshal(body, &issue); err != nil {
		return nil, err
	}
	return &issue, nil
}

func (r *RemoteStore) UpdateStatus(id int, statusValue string, actor string) (*Issue, error) {
	_ = actor
	body, status, err := r.do("PATCH", fmt.Sprintf("/issues/%d/status", id), updateIssueStatusPayload{Status: statusValue})
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		return nil, fmt.Errorf("update issue status: server returned %d: %s", status, body)
	}
	var issue Issue
	if err := json.Unmarshal(body, &issue); err != nil {
		return nil, err
	}
	return &issue, nil
}
