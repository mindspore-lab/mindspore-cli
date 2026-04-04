package server

import (
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/mindspore-lab/mindspore-cli/internal/bugs"
)

func HandleListActivity(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		bugID, err := strconv.Atoi(r.PathValue("id"))
		if err != nil {
			http.Error(w, `{"error":"invalid bug id"}`, http.StatusBadRequest)
			return
		}
		acts, err := store.ListActivity(bugID)
		if err != nil {
			http.Error(w, `{"error":"failed to list activity"}`, http.StatusInternalServerError)
			return
		}
		if acts == nil {
			acts = []bugs.Activity{}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(acts)
	}
}
