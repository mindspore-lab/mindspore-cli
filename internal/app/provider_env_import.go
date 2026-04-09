package app

import (
	"fmt"
	"net/url"
	"os"
	"path"
	"regexp"
	"sort"
	"strings"

	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

const (
	providerImportSourceClaudeCode    = "claude-code-env"
	providerImportSourceClaudeCodeUI  = "Claude Code env"
	providerProtocolAnthropicMessages = "anthropic-messages"
)

var providerVersionSuffixPattern = regexp.MustCompile(`^v[0-9][a-z0-9.-]*$`)

type providerImportSuggestion struct {
	ProviderID    string
	ProviderLabel string
	Source        string
	SourceLabel   string
	Protocol      string
	BaseURL       string
	APIKey        string
	APIKeyEnvVar  string
	BaseURLEnvVar string
}

type providerEnvCandidate struct {
	Source        string
	SourceLabel   string
	Protocol      string
	BaseURL       string
	APIKey        string
	APIKeyEnvVar  string
	BaseURLEnvVar string
}

func detectProviderImportSuggestions(catalog *providerCatalog, persistedAuth *providerAuthState) []providerImportSuggestion {
	if catalog == nil {
		return nil
	}

	connected := map[string]struct{}{}
	if persistedAuth != nil {
		for providerID, entry := range persistedAuth.Providers {
			if strings.TrimSpace(entry.APIKey) == "" {
				continue
			}
			connected[normalizedProviderID(providerID)] = struct{}{}
		}
	}

	seen := map[string]struct{}{}
	out := make([]providerImportSuggestion, 0)
	for _, candidate := range detectClaudeCodeEnvCandidates() {
		provider, ok := matchProviderCatalogForEnvCandidate(catalog, candidate)
		if !ok {
			continue
		}
		if _, ok := connected[provider.ID]; ok {
			continue
		}
		if _, ok := seen[provider.ID]; ok {
			continue
		}
		seen[provider.ID] = struct{}{}
		out = append(out, providerImportSuggestion{
			ProviderID:    provider.ID,
			ProviderLabel: provider.Label,
			Source:        candidate.Source,
			SourceLabel:   candidate.SourceLabel,
			Protocol:      provider.Protocol,
			BaseURL:       provider.BaseURL,
			APIKey:        candidate.APIKey,
			APIKeyEnvVar:  candidate.APIKeyEnvVar,
			BaseURLEnvVar: candidate.BaseURLEnvVar,
		})
	}

	sort.Slice(out, func(i, j int) bool {
		if out[i].ProviderLabel != out[j].ProviderLabel {
			return out[i].ProviderLabel < out[j].ProviderLabel
		}
		return out[i].ProviderID < out[j].ProviderID
	})
	return out
}

func detectClaudeCodeEnvCandidates() []providerEnvCandidate {
	baseURL := strings.TrimSpace(os.Getenv("ANTHROPIC_BASE_URL"))
	apiKey := strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY"))
	if baseURL == "" {
		return nil
	}
	return []providerEnvCandidate{
		{
			Source:        providerImportSourceClaudeCode,
			SourceLabel:   providerImportSourceClaudeCodeUI,
			Protocol:      providerProtocolAnthropicMessages,
			BaseURL:       baseURL,
			APIKey:        apiKey,
			APIKeyEnvVar:  "ANTHROPIC_API_KEY",
			BaseURLEnvVar: "ANTHROPIC_BASE_URL",
		},
	}
}

func hasProviderEnvCandidates() bool {
	return len(detectClaudeCodeEnvCandidates()) > 0
}

func providerImportBaseURLLine(suggestion providerImportSuggestion) string {
	return "- " + suggestion.BaseURLEnvVar + "=" + strings.TrimSpace(os.Getenv(suggestion.BaseURLEnvVar))
}

func providerImportAPIKeyLine(suggestion providerImportSuggestion) string {
	if strings.TrimSpace(suggestion.APIKey) == "" {
		return fmt.Sprintf("- %s is not set; you'll enter it next", suggestion.APIKeyEnvVar)
	}
	return "- " + suggestion.APIKeyEnvVar + "=" + maskImportedAPIKey(suggestion.APIKey)
}

func maskImportedAPIKey(raw string) string {
	token := strings.TrimSpace(raw)
	if token == "" {
		return ""
	}
	runes := []rune(token)
	if len(runes) <= 8 {
		prefix := 4
		if len(runes) < prefix {
			prefix = len(runes)
		}
		return string(runes[:prefix]) + "****"
	}
	if len(runes) <= 16 {
		return string(runes[:8]) + "****" + string(runes[len(runes)-4:])
	}
	prefix := 12
	if prefix > len(runes)-4 {
		prefix = len(runes) - 4
	}
	if prefix < 4 {
		prefix = 4
	}
	return string(runes[:prefix]) + "****" + string(runes[len(runes)-4:])
}

func matchProviderCatalogForEnvCandidate(catalog *providerCatalog, candidate providerEnvCandidate) (providerCatalogEntry, bool) {
	if catalog == nil {
		return providerCatalogEntry{}, false
	}

	bestScore := 0
	best := providerCatalogEntry{}
	for _, provider := range catalog.Providers {
		if normalizedProviderProtocol(provider.Protocol) != normalizedProviderProtocol(candidate.Protocol) {
			continue
		}
		score := providerBaseURLMatchScore(provider.BaseURL, candidate.BaseURL)
		if score > bestScore {
			bestScore = score
			best = provider
		}
	}
	if bestScore == 0 {
		return providerCatalogEntry{}, false
	}
	return best, true
}

func normalizedProviderProtocol(v string) string {
	return strings.ToLower(strings.TrimSpace(v))
}

func providerBaseURLMatchScore(catalogBaseURL, envBaseURL string) int {
	catalogVariants := comparableProviderBaseURLVariants(catalogBaseURL)
	envVariants := comparableProviderBaseURLVariants(envBaseURL)
	if len(catalogVariants) == 0 || len(envVariants) == 0 {
		return 0
	}

	for i, left := range catalogVariants {
		for j, right := range envVariants {
			if left != right {
				continue
			}
			if i == 0 && j == 0 {
				return 2
			}
			return 1
		}
	}
	return 0
}

func comparableProviderBaseURLVariants(raw string) []string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return nil
	}

	parsed, err := url.Parse(trimmed)
	if err != nil || parsed.Host == "" {
		return []string{strings.ToLower(strings.TrimRight(trimmed, "/"))}
	}

	scheme := strings.ToLower(strings.TrimSpace(parsed.Scheme))
	host := strings.ToLower(strings.TrimSpace(parsed.Host))
	basePath := normalizeComparableURLPath(parsed.Path)

	seen := map[string]struct{}{}
	add := func(pathValue string, out *[]string) {
		value := scheme + "://" + host + pathValue
		if _, ok := seen[value]; ok {
			return
		}
		seen[value] = struct{}{}
		*out = append(*out, value)
	}

	out := make([]string, 0, 3)
	add(basePath, &out)

	if trimmedVersion := trimVersionSuffixFromPath(basePath); trimmedVersion != basePath {
		add(trimmedVersion, &out)
	}

	return out
}

func normalizeComparableURLPath(rawPath string) string {
	if strings.TrimSpace(rawPath) == "" {
		return ""
	}
	cleaned := path.Clean("/" + strings.TrimSpace(rawPath))
	if cleaned == "/" {
		return ""
	}
	return strings.TrimRight(cleaned, "/")
}

func trimVersionSuffixFromPath(rawPath string) string {
	trimmed := strings.Trim(strings.TrimSpace(rawPath), "/")
	if trimmed == "" {
		return ""
	}
	segments := strings.Split(trimmed, "/")
	last := strings.ToLower(strings.TrimSpace(segments[len(segments)-1]))
	if !providerVersionSuffixPattern.MatchString(last) {
		return rawPath
	}
	if len(segments) == 1 {
		return ""
	}
	return "/" + strings.Join(segments[:len(segments)-1], "/")
}

func mergeProviderAuthStateWithImports(persistedAuth *providerAuthState, suggestions []providerImportSuggestion) *providerAuthState {
	merged := cloneProviderAuthState(persistedAuth)
	if merged == nil {
		merged = emptyProviderAuthState()
	}
	for _, suggestion := range suggestions {
		if strings.TrimSpace(suggestion.ProviderID) == "" || strings.TrimSpace(suggestion.APIKey) == "" {
			continue
		}
		if entry, ok := merged.Providers[suggestion.ProviderID]; ok && strings.TrimSpace(entry.APIKey) != "" {
			continue
		}
		merged.Providers[suggestion.ProviderID] = providerAuthEntry{
			ProviderID: suggestion.ProviderID,
			APIKey:     suggestion.APIKey,
		}
	}
	merged.normalize()
	return merged
}

func cloneProviderAuthState(in *providerAuthState) *providerAuthState {
	if in == nil {
		return nil
	}
	out := &providerAuthState{
		Providers: make(map[string]providerAuthEntry, len(in.Providers)),
	}
	for key, entry := range in.Providers {
		out.Providers[key] = entry
	}
	out.normalize()
	return out
}

func providerImportSuggestionByID(suggestions []providerImportSuggestion) map[string]providerImportSuggestion {
	if len(suggestions) == 0 {
		return nil
	}
	out := make(map[string]providerImportSuggestion, len(suggestions))
	for _, suggestion := range suggestions {
		out[normalizedProviderID(suggestion.ProviderID)] = suggestion
	}
	return out
}

func toModelProviderImportSuggestions(in []providerImportSuggestion) []model.ProviderImportSuggestion {
	if len(in) == 0 {
		return nil
	}
	out := make([]model.ProviderImportSuggestion, 0, len(in))
	for _, suggestion := range in {
		out = append(out, model.ProviderImportSuggestion{
			ProviderID:    suggestion.ProviderID,
			ProviderLabel: suggestion.ProviderLabel,
			Source:        suggestion.Source,
			SourceLabel:   suggestion.SourceLabel,
			Protocol:      suggestion.Protocol,
			BaseURL:       suggestion.BaseURL,
			APIKeyEnvVar:  suggestion.APIKeyEnvVar,
			BaseURLEnvVar: suggestion.BaseURLEnvVar,
		})
	}
	return out
}
