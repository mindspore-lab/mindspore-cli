package server

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
)

type addNoteRequest struct {
	Content string `json:"content"`
}

func HandleAddNote(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		bugID, err := strconv.Atoi(r.PathValue("id"))
		if err != nil {
			http.Error(w, `{"error":"invalid bug id"}`, http.StatusBadRequest)
			return
		}
		var req addNoteRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
			return
		}
		if strings.TrimSpace(req.Content) == "" {
			http.Error(w, `{"error":"content is required"}`, http.StatusBadRequest)
			return
		}
		author := UserFromContext(r.Context())
		note, err := store.AddNote(bugID, author, req.Content)
		if err != nil {
			http.Error(w, `{"error":"failed to add note"}`, http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(note)
	}
}
