package server

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
)

type updateIssueStatusRequest struct {
	Status string `json:"status"`
}

func HandleUpdateIssueStatus(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		issueID, err := strconv.Atoi(r.PathValue("id"))
		if err != nil {
			http.Error(w, `{"error":"invalid issue id"}`, http.StatusBadRequest)
			return
		}
		var req updateIssueStatusRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
			return
		}
		if strings.TrimSpace(req.Status) == "" {
			http.Error(w, `{"error":"status is required"}`, http.StatusBadRequest)
			return
		}
		actor := UserFromContext(r.Context())
		issue, err := store.UpdateIssueStatus(issueID, req.Status, actor)
		if err != nil {
			http.Error(w, `{"error":"failed to update issue status"}`, http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(issue)
	}
}
