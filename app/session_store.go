package main

import (
	"os"
	"strings"
)

func (a *Application) persistSessionState() error {
	if a == nil {
		return nil
	}
	st := a.SessionState
	st.Model = a.SessionModel

	if a.Config.Session.PersistAPIKeys {
		if key := strings.TrimSpace(os.Getenv(a.Config.Providers.OpenAI.APIKeyEnv)); key != "" {
			st.APIKeys.OpenAI = key
		}
		if key := strings.TrimSpace(os.Getenv(a.Config.Providers.OpenRouter.APIKeyEnv)); key != "" {
			st.APIKeys.OpenRouter = key
		}
	} else {
		st.APIKeys = PersistedAPIKeys{}
	}

	if err := SavePersistentState(a.SessionPath, st); err != nil {
		return err
	}
	a.SessionState = st
	return nil
}
