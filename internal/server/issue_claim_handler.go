package server

import (
	"encoding/json"
	"net/http"
	"strconv"
)

func HandleClaimIssue(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		issueID, err := strconv.Atoi(r.PathValue("id"))
		if err != nil {
			http.Error(w, `{"error":"invalid issue id"}`, http.StatusBadRequest)
			return
		}
		lead := UserFromContext(r.Context())
		issue, err := store.ClaimIssue(issueID, lead)
		if err != nil {
			http.Error(w, `{"error":"failed to claim issue"}`, http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(issue)
	}
}
