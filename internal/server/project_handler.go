package server

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
)

func HandleGetProjectSnapshot(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		snap, err := store.GetProjectSnapshot()
		if err != nil {
			http.Error(w, `{"error":"failed to get project snapshot"}`, http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(snap)
	}
}

type createProjectTaskRequest struct {
	Section  string `json:"section"`
	Title    string `json:"title"`
	Owner    string `json:"owner"`
	Due      string `json:"due"`
	Tags     string `json:"tags"`
	Progress *int   `json:"progress,omitempty"`
}

func HandleCreateProjectTask(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req createProjectTaskRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
			return
		}
		if strings.TrimSpace(req.Title) == "" {
			http.Error(w, `{"error":"title is required"}`, http.StatusBadRequest)
			return
		}
		if strings.TrimSpace(req.Section) == "" {
			req.Section = "tasks"
		}
		createdBy := UserFromContext(r.Context())
		task, err := store.CreateProjectTask(req.Section, req.Title, req.Owner, createdBy, req.Due, req.Tags, req.Progress)
		if err != nil {
			http.Error(w, `{"error":"failed to create task"}`, http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(task)
	}
}

type updateProjectTaskRequest struct {
	Title    *string `json:"title,omitempty"`
	Owner    *string `json:"owner,omitempty"`
	Status   *string `json:"status,omitempty"`
	Due      *string `json:"due,omitempty"`
	Tags     *string `json:"tags,omitempty"`
	Progress *int    `json:"progress,omitempty"`
}

func HandleUpdateProjectTask(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id, err := strconv.Atoi(r.PathValue("id"))
		if err != nil {
			http.Error(w, `{"error":"invalid task id"}`, http.StatusBadRequest)
			return
		}
		var req updateProjectTaskRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
			return
		}
		task, err := store.UpdateProjectTask(id, req.Title, req.Owner, req.Status, req.Due, req.Tags, req.Progress)
		if err != nil {
			http.Error(w, `{"error":"task not found"}`, http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(task)
	}
}

func HandleDeleteProjectTask(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id, err := strconv.Atoi(r.PathValue("id"))
		if err != nil {
			http.Error(w, `{"error":"invalid task id"}`, http.StatusBadRequest)
			return
		}
		if err := store.DeleteProjectTask(id); err != nil {
			http.Error(w, `{"error":"task not found"}`, http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"ok":true}`))
	}
}

type updateProjectOverviewRequest struct {
	Phase  string `json:"phase"`
	Owner  string `json:"owner"`
	Repo   string `json:"repo"`
	Branch string `json:"branch"`
}

func HandleUpdateProjectOverview(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req updateProjectOverviewRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
			return
		}
		ov, err := store.UpdateProjectOverview(req.Phase, req.Owner, req.Repo, req.Branch)
		if err != nil {
			http.Error(w, `{"error":"failed to update overview"}`, http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ov)
	}
}
