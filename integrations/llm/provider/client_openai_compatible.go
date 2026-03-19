package provider

import "github.com/vigo999/ms-cli/integrations/llm"

// NewOpenAICompatibleProvider builds an OpenAI-compatible provider using the OpenAI protocol codec and client.
func NewOpenAICompatibleProvider(cfg ResolvedConfig) (llm.Provider, error) {
	return newOpenAIClient(cfg, string(ProviderOpenAICompatible), nil)
}
