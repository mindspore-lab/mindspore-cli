package issues

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

func (r *RemoteStore) CreateBug(title, reporter string) (*Bug, error) {
	body, status, err := r.do("POST", "/bugs", map[string]string{"title": title})
	if err != nil {
		return nil, err
	}
	if status != http.StatusCreated {
		return nil, fmt.Errorf("create bug: server returned %d: %s", status, body)
	}
	var bug Bug
	if err := json.Unmarshal(body, &bug); err != nil {
		return nil, err
	}
	return &bug, nil
}

func (r *RemoteStore) ListBugs(filterStatus string) ([]Bug, error) {
	path := "/bugs"
	if filterStatus != "" {
		path += "?status=" + filterStatus
	}
	body, status, err := r.do("GET", path, nil)
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		return nil, fmt.Errorf("list bugs: server returned %d: %s", status, body)
	}
	var bugs []Bug
	if err := json.Unmarshal(body, &bugs); err != nil {
		return nil, err
	}
	return bugs, nil
}

func (r *RemoteStore) GetBug(id int) (*Bug, error) {
	body, status, err := r.do("GET", fmt.Sprintf("/bugs/%d", id), nil)
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		return nil, fmt.Errorf("get bug: server returned %d: %s", status, body)
	}
	var bug Bug
	if err := json.Unmarshal(body, &bug); err != nil {
		return nil, err
	}
	return &bug, nil
}

func (r *RemoteStore) ClaimBug(id int, lead string) error {
	body, status, err := r.do("POST", fmt.Sprintf("/bugs/%d/claim", id), nil)
	if err != nil {
		return err
	}
	if status != http.StatusOK {
		return fmt.Errorf("claim bug: server returned %d: %s", status, body)
	}
	return nil
}

func (r *RemoteStore) AddNote(bugID int, author, content string) (*Note, error) {
	body, status, err := r.do("POST", fmt.Sprintf("/bugs/%d/notes", bugID), map[string]string{"content": content})
	if err != nil {
		return nil, err
	}
	if status != http.StatusCreated {
		return nil, fmt.Errorf("add note: server returned %d: %s", status, body)
	}
	var note Note
	if err := json.Unmarshal(body, &note); err != nil {
		return nil, err
	}
	return &note, nil
}

func (r *RemoteStore) ListActivity(bugID int) ([]Activity, error) {
	body, status, err := r.do("GET", fmt.Sprintf("/bugs/%d/activity", bugID), nil)
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		return nil, fmt.Errorf("list activity: server returned %d: %s", status, body)
	}
	var acts []Activity
	if err := json.Unmarshal(body, &acts); err != nil {
		return nil, err
	}
	return acts, nil
}

func (r *RemoteStore) DockSummary() (*DockData, error) {
	body, status, err := r.do("GET", "/dock", nil)
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		return nil, fmt.Errorf("dock summary: server returned %d: %s", status, body)
	}
	var data DockData
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, err
	}
	return &data, nil
}
