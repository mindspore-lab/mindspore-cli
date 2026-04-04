package server

import (
	"net/http"

	"github.com/mindspore-lab/mindspore-cli/configs"
)

func NewMux(store *Store, tokens []configs.TokenEntry, modelPresets []configs.ModelPresetCredential) *http.ServeMux {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

	auth := func(h http.Handler) http.Handler {
		return AuthMiddleware(tokens, h)
	}

	mux.Handle("GET /me", auth(http.HandlerFunc(HandleMe(store))))
	mux.Handle("GET /model-presets/{id}/credential", auth(http.HandlerFunc(HandleGetModelPresetCredential(modelPresets))))
	mux.Handle("POST /issues", auth(http.HandlerFunc(HandleCreateIssue(store))))
	mux.Handle("GET /issues", auth(http.HandlerFunc(HandleListIssues(store))))
	mux.Handle("GET /issues/{id}", auth(http.HandlerFunc(HandleGetIssue(store))))
	mux.Handle("GET /issues/{id}/notes", auth(http.HandlerFunc(HandleListIssueNotes(store))))
	mux.Handle("POST /issues/{id}/notes", auth(http.HandlerFunc(HandleAddIssueNote(store))))
	mux.Handle("GET /issues/{id}/activity", auth(http.HandlerFunc(HandleListIssueActivity(store))))
	mux.Handle("POST /issues/{id}/claim", auth(http.HandlerFunc(HandleClaimIssue(store))))
	mux.Handle("PATCH /issues/{id}/status", auth(http.HandlerFunc(HandleUpdateIssueStatus(store))))
	mux.Handle("POST /bugs", auth(http.HandlerFunc(HandleCreateBug(store))))
	mux.Handle("GET /bugs", auth(http.HandlerFunc(HandleListBugs(store))))
	mux.Handle("GET /bugs/{id}", auth(http.HandlerFunc(HandleGetBug(store))))
	mux.Handle("POST /bugs/{id}/notes", auth(http.HandlerFunc(HandleAddNote(store))))
	mux.Handle("POST /bugs/{id}/claim", auth(http.HandlerFunc(HandleClaimBug(store))))
	mux.Handle("POST /bugs/{id}/close", auth(http.HandlerFunc(HandleCloseBug(store))))
	mux.Handle("GET /bugs/{id}/activity", auth(http.HandlerFunc(HandleListActivity(store))))
	mux.Handle("GET /dock", auth(http.HandlerFunc(HandleDock(store))))

	// Project routes
	mux.Handle("GET /project", auth(http.HandlerFunc(HandleGetProjectSnapshot(store))))
	mux.Handle("POST /project/tasks", auth(AdminOnly(http.HandlerFunc(HandleCreateProjectTask(store)))))
	mux.Handle("PATCH /project/tasks/{id}", auth(AdminOnly(http.HandlerFunc(HandleUpdateProjectTask(store)))))
	mux.Handle("DELETE /project/tasks/{id}", auth(AdminOnly(http.HandlerFunc(HandleDeleteProjectTask(store)))))
	mux.Handle("PATCH /project/overview", auth(AdminOnly(http.HandlerFunc(HandleUpdateProjectOverview(store)))))

	return mux
}
