package app

import (
	"strings"

	agentctx "github.com/mindspore-lab/mindspore-cli/agent/context"
	"github.com/mindspore-lab/mindspore-cli/agent/session"
)

func providerUsageSnapshotFromDetails(details agentctx.TokenUsageDetails) *session.UsageSnapshot {
	if details.Source != agentctx.TokenUsageSourceProvider || details.ProviderSnapshotTokens <= 0 {
		return nil
	}

	return &session.UsageSnapshot{
		Provider:   details.Provider,
		TokenScope: string(details.ProviderTokenScope),
		Tokens:     details.ProviderSnapshotTokens,
		LocalDelta: details.LocalDelta,
	}
}

func restoreProviderUsageSnapshot(cm *agentctx.Manager, usage *session.UsageSnapshot) {
	if cm == nil || usage == nil || usage.Tokens <= 0 {
		return
	}

	scope := agentctx.ProviderTokenScope(strings.TrimSpace(usage.TokenScope))
	if scope != agentctx.ProviderTokenScopeTotal {
		scope = agentctx.ProviderTokenScopePrompt
	}

	cm.RestoreProviderUsageSnapshot(agentctx.ProviderUsageSnapshot{
		Provider:   usage.Provider,
		TokenScope: scope,
		Tokens:     usage.Tokens,
		LocalDelta: usage.LocalDelta,
	})
}
