package server

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/vigo999/mindspore-cli/configs"
)

func HandleGetModelPresetCredential(presets []configs.ModelPresetCredential) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimSpace(r.PathValue("id"))
		if id == "" {
			http.Error(w, `{"error":"model preset id is required"}`, http.StatusBadRequest)
			return
		}

		for _, preset := range presets {
			if strings.EqualFold(strings.TrimSpace(preset.ID), id) {
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]string{
					"id":      strings.TrimSpace(preset.ID),
					"api_key": strings.TrimSpace(preset.APIKey),
				})
				return
			}
		}

		http.Error(w, `{"error":"model preset not found"}`, http.StatusNotFound)
	}
}
