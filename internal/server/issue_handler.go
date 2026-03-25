package server

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"

	issuepkg "github.com/vigo999/ms-cli/internal/issues"
)

type createIssueRequest struct {
	Title string `json:"title"`
	Kind  string `json:"kind"`
}

func HandleCreateIssue(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req createIssueRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
			return
		}
		if strings.TrimSpace(req.Title) == "" {
			http.Error(w, `{"error":"title is required"}`, http.StatusBadRequest)
			return
		}
		kind, err := issuepkg.NormalizeKind(req.Kind)
		if err != nil {
			http.Error(w, `{"error":"invalid issue kind"}`, http.StatusBadRequest)
			return
		}
		reporter := UserFromContext(r.Context())
		issue, err := store.CreateIssue(req.Title, kind, reporter)
		if err != nil {
			http.Error(w, `{"error":"failed to create issue"}`, http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(issue)
	}
}

func HandleListIssues(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		status := strings.TrimSpace(r.URL.Query().Get("status"))
		issueList, err := store.ListIssues(status)
		if err != nil {
			http.Error(w, `{"error":"failed to list issues"}`, http.StatusInternalServerError)
			return
		}
		if issueList == nil {
			issueList = []issuepkg.Issue{}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(issueList)
	}
}

func HandleGetIssue(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id, err := strconv.Atoi(r.PathValue("id"))
		if err != nil {
			http.Error(w, `{"error":"invalid issue id"}`, http.StatusBadRequest)
			return
		}
		issue, err := store.GetIssue(id)
		if err != nil {
			http.Error(w, `{"error":"issue not found"}`, http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(issue)
	}
}
