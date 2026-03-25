package server

import (
	"encoding/json"
	"net/http"
	"strconv"

	issuepkg "github.com/vigo999/ms-cli/internal/issues"
)

func HandleListIssueActivity(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		issueID, err := strconv.Atoi(r.PathValue("id"))
		if err != nil {
			http.Error(w, `{"error":"invalid issue id"}`, http.StatusBadRequest)
			return
		}
		acts, err := store.ListIssueActivity(issueID)
		if err != nil {
			http.Error(w, `{"error":"failed to list issue activity"}`, http.StatusInternalServerError)
			return
		}
		if acts == nil {
			acts = []issuepkg.Activity{}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(acts)
	}
}
