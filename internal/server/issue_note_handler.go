package server

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"

	issuepkg "github.com/mindspore-lab/mindspore-cli/internal/issues"
)

type addIssueNoteRequest struct {
	Content string `json:"content"`
}

func HandleAddIssueNote(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		issueID, err := strconv.Atoi(r.PathValue("id"))
		if err != nil {
			http.Error(w, `{"error":"invalid issue id"}`, http.StatusBadRequest)
			return
		}
		var req addIssueNoteRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
			return
		}
		if strings.TrimSpace(req.Content) == "" {
			http.Error(w, `{"error":"content is required"}`, http.StatusBadRequest)
			return
		}
		author := UserFromContext(r.Context())
		note, err := store.AddIssueNote(issueID, author, req.Content)
		if err != nil {
			http.Error(w, `{"error":"failed to add issue note"}`, http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(note)
	}
}

func HandleListIssueNotes(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		issueID, err := strconv.Atoi(r.PathValue("id"))
		if err != nil {
			http.Error(w, `{"error":"invalid issue id"}`, http.StatusBadRequest)
			return
		}
		notes, err := store.ListIssueNotes(issueID)
		if err != nil {
			http.Error(w, `{"error":"failed to list issue notes"}`, http.StatusInternalServerError)
			return
		}
		if notes == nil {
			notes = []issuepkg.Note{}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(notes)
	}
}
