package server

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"

	bugspkg "github.com/mindspore-lab/mindspore-cli/internal/bugs"
)

type createBugRequest struct {
	Title string   `json:"title"`
	Tags  []string `json:"tags,omitempty"`
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
		bug, err := store.CreateBug(req.Title, reporter, req.Tags)
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
		bugList, err := store.ListBugs(status)
		if err != nil {
			http.Error(w, `{"error":"failed to list bugs"}`, http.StatusInternalServerError)
			return
		}
		if bugList == nil {
			bugList = []bugspkg.Bug{}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(bugList)
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
