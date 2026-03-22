package server

import (
	"encoding/json"
	"net/http"
	"strconv"
)

func HandleClaimBug(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		bugID, err := strconv.Atoi(r.PathValue("id"))
		if err != nil {
			http.Error(w, `{"error":"invalid bug id"}`, http.StatusBadRequest)
			return
		}
		lead := UserFromContext(r.Context())
		if err := store.ClaimBug(bugID, lead); err != nil {
			http.Error(w, `{"error":"failed to claim bug"}`, http.StatusInternalServerError)
			return
		}
		bug, err := store.GetBug(bugID)
		if err != nil {
			http.Error(w, `{"error":"bug not found"}`, http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(bug)
	}
}
