package server

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"

	"github.com/vigo999/ms-cli/internal/issues"
)

type createBugRequest struct {
	Title string `json:"title"`
}

func HandleCreateBug(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req createBugRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
			return
		}
		if strings.TrimSpace(req.Title) == "" {
			http.Error(w, `{"error":"title is required"}`, http.StatusBadRequest)
			return
		}
		reporter := UserFromContext(r.Context())
		bug, err := store.CreateBug(req.Title, reporter)
		if err != nil {
			http.Error(w, `{"error":"failed to create bug"}`, http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(bug)
	}
}

func HandleListBugs(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		status := r.URL.Query().Get("status")
		bugs, err := store.ListBugs(status)
		if err != nil {
			http.Error(w, `{"error":"failed to list bugs"}`, http.StatusInternalServerError)
			return
		}
		if bugs == nil {
			bugs = []issues.Bug{}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(bugs)
	}
}

func HandleGetBug(store *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id, err := strconv.Atoi(r.PathValue("id"))
		if err != nil {
			http.Error(w, `{"error":"invalid bug id"}`, http.StatusBadRequest)
			return
		}
		bug, err := store.GetBug(id)
		if err != nil {
			http.Error(w, `{"error":"bug not found"}`, http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(bug)
	}
}
