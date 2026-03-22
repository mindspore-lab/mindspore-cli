package server

import (
	"encoding/json"
	"net/http"
)

type meResponse struct {
	User string `json:"user"`
	Role string `json:"role"`
}

func HandleMe(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(meResponse{
		User: UserFromContext(r.Context()),
		Role: RoleFromContext(r.Context()),
	})
}
