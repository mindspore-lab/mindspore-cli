package server

import (
	"net/http"

	"github.com/vigo999/ms-cli/configs"
)

func NewMux(store *Store, tokens []configs.TokenEntry) *http.ServeMux {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

	auth := func(h http.Handler) http.Handler {
		return AuthMiddleware(tokens, h)
	}

	mux.Handle("GET /me", auth(http.HandlerFunc(HandleMe)))
	mux.Handle("POST /bugs", auth(http.HandlerFunc(HandleCreateBug(store))))
	mux.Handle("GET /bugs", auth(http.HandlerFunc(HandleListBugs(store))))
	mux.Handle("GET /bugs/{id}", auth(http.HandlerFunc(HandleGetBug(store))))
	mux.Handle("POST /bugs/{id}/notes", auth(http.HandlerFunc(HandleAddNote(store))))
	mux.Handle("POST /bugs/{id}/claim", auth(http.HandlerFunc(HandleClaimBug(store))))
	mux.Handle("GET /bugs/{id}/activity", auth(http.HandlerFunc(HandleListActivity(store))))
	mux.Handle("GET /dock", auth(http.HandlerFunc(HandleDock(store))))

	return mux
}
