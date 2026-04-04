package server

import (
	"context"
	"net/http"
	"strings"

	"github.com/vigo999/mindspore-cli/configs"
)

type ctxKey string

const (
	ctxUser ctxKey = "user"
	ctxRole ctxKey = "role"
)

func UserFromContext(ctx context.Context) string {
	v, _ := ctx.Value(ctxUser).(string)
	return v
}

func RoleFromContext(ctx context.Context) string {
	v, _ := ctx.Value(ctxRole).(string)
	return v
}

func AdminOnly(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if RoleFromContext(r.Context()) != "admin" {
			http.Error(w, `{"error":"admin role required"}`, http.StatusForbidden)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func AuthMiddleware(tokens []configs.TokenEntry, next http.Handler) http.Handler {
	lookup := make(map[string]configs.TokenEntry, len(tokens))
	for _, t := range tokens {
		lookup[t.Token] = t
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		header := r.Header.Get("Authorization")
		if !strings.HasPrefix(header, "Bearer ") {
			http.Error(w, `{"error":"missing or invalid authorization header"}`, http.StatusUnauthorized)
			return
		}
		token := strings.TrimPrefix(header, "Bearer ")
		entry, ok := lookup[token]
		if !ok {
			http.Error(w, `{"error":"invalid token"}`, http.StatusUnauthorized)
			return
		}
		ctx := context.WithValue(r.Context(), ctxUser, entry.User)
		ctx = context.WithValue(ctx, ctxRole, entry.Role)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}
