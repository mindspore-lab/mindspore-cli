package provider

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
)

type cache struct {
	mu    sync.RWMutex
	items map[string]any
}

func newCache() *cache {
	return &cache{items: make(map[string]any)}
}

func (c *cache) get(key string) (any, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	value, ok := c.items[key]
	return value, ok
}

func (c *cache) set(key string, value any) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.items[key] = value
}

func cacheKey(cfg ResolvedConfig) string {
	fingerprint := struct {
		Kind           string      `json:"kind"`
		BaseURL        string      `json:"base_url"`
		Model          string      `json:"model"`
		Timeout        string      `json:"timeout"`
		AuthHeaderName string      `json:"auth_header_name"`
		APIKeyHash     string      `json:"api_key_hash"`
		Headers        []headerKey `json:"headers"`
	}{
		Kind:           strings.TrimSpace(string(cfg.Kind)),
		BaseURL:        strings.TrimSpace(cfg.BaseURL),
		Model:          strings.TrimSpace(cfg.Model),
		Timeout:        cfg.Timeout.String(),
		AuthHeaderName: strings.TrimSpace(cfg.AuthHeaderName),
		APIKeyHash:     hashString(strings.TrimSpace(cfg.APIKey)),
		Headers:        canonicalHeaders(cfg.Headers),
	}

	payload, err := json.Marshal(fingerprint)
	if err != nil {
		return fmt.Sprintf("marshal-error:%x", sha256.Sum256([]byte(fmt.Sprintf("%#v", fingerprint))))
	}

	sum := sha256.Sum256(payload)
	return hex.EncodeToString(sum[:])
}

type headerKey struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

func canonicalHeaders(headers map[string]string) []headerKey {
	if len(headers) == 0 {
		return nil
	}

	keys := make([]string, 0, len(headers))
	lookup := make(map[string]string, len(headers))
	for key, value := range headers {
		name := strings.ToLower(strings.TrimSpace(key))
		if name == "" {
			continue
		}
		keys = append(keys, name)
		lookup[name] = value
	}

	sort.Strings(keys)

	result := make([]headerKey, 0, len(keys))
	for _, key := range keys {
		result = append(result, headerKey{Name: key, Value: lookup[key]})
	}
	return result
}

func hashString(v string) string {
	sum := sha256.Sum256([]byte(v))
	return hex.EncodeToString(sum[:])
}
