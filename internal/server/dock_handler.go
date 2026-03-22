package server

import (
	"encoding/json"
	"net/http"
)

func HandleDock(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		data, err := store.DockSummary()
		if err != nil {
			http.Error(w, `{"error":"failed to get dock summary"}`, http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(data)
	}
}
