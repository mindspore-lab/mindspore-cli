package context

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
)

// ManagerConfig holds the manager configuration.
type ManagerConfig struct {
	ContextWindow       int
	ReserveTokens       int
	CompactionThreshold float64

	// 新增配置
	EnableSmartCompact bool            // 启用智能压缩
	CompactStrategy    CompactStrategy // 压缩策略
	EnablePriority     bool            // 启用优先级系统
}

// DefaultManagerConfig 返回默认配置
func DefaultManagerConfig() ManagerConfig {
	return ManagerConfig{
		ContextWindow:       200000,
		ReserveTokens:       4000,
		CompactionThreshold: 0.9,
		EnableSmartCompact:  true,
		CompactStrategy:     CompactStrategyHybrid,
		EnablePriority:      true,
	}
}

const compactionTargetRatio = 0.5

// Manager manages conversation context.
type Manager struct {
	config   ManagerConfig
	mu       sync.RWMutex
	messages []llm.Message
	system   *llm.Message
	usage    TokenUsage

	exactSnapshotTokens   int
	exactSnapshotEstimate int
	exactSnapshotProvider string
	exactSnapshotScope    ProviderTokenScope
	hasExactSnapshot      bool

	// 增强组件
	tokenizer *Tokenizer
	compactor *Compactor
	scorer    *PriorityScorer

	// 统计
	stats Stats
}

// TokenUsage represents token usage statistics.
type TokenUsage struct {
	Current       int
	ContextWindow int
	Reserved      int
	Available     int
}

type TokenUsageSource string
type ProviderTokenScope string

const (
	TokenUsageSourceLocalEstimate TokenUsageSource = "local_estimate"
	TokenUsageSourceProvider      TokenUsageSource = "provider_snapshot"

	ProviderTokenScopePrompt ProviderTokenScope = "prompt"
	ProviderTokenScopeTotal  ProviderTokenScope = "total"
)

type TokenUsageDetails struct {
	TokenUsage
	Source                 TokenUsageSource
	Provider               string
	ProviderSnapshotTokens int
	ProviderTokenScope     ProviderTokenScope
	LocalEstimatedTotal    int
	LocalDelta             int
}

type ProviderUsageSnapshot struct {
	Provider   string
	TokenScope ProviderTokenScope
	Tokens     int
	LocalDelta int
}

// Stats 上下文统计
type Stats struct {
	MessageCount    int
	ToolCallCount   int
	CompactCount    int
	LastCompactAt   *time.Time
	TotalTokensUsed int
}

// NewManager creates a new context manager.
func NewManager(cfg ManagerConfig) *Manager {
	if cfg.ContextWindow == 0 {
		cfg.ContextWindow = 200000
	}
	if cfg.ReserveTokens == 0 {
		cfg.ReserveTokens = 4000
	}
	if cfg.CompactionThreshold == 0 {
		cfg.CompactionThreshold = 0.9
	}

	// 创建压缩器
	compactor := NewCompactor(CompactorConfig{
		Strategy: cfg.CompactStrategy,
	})

	m := &Manager{
		config:    cfg,
		messages:  make([]llm.Message, 0),
		tokenizer: NewTokenizer(),
		compactor: compactor,
		scorer:    NewPriorityScorer(),
		usage: TokenUsage{
			ContextWindow: cfg.ContextWindow,
			Reserved:      cfg.ReserveTokens,
			Available:     cfg.ContextWindow - cfg.ReserveTokens,
		},
	}

	return m
}

// SetSystemPrompt sets the system prompt.
func (m *Manager) SetSystemPrompt(content string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	msg := llm.NewSystemMessage(content)
	m.system = &msg
	m.clearProviderTokenUsageLocked()

	m.recalculateUsage()
}

// GetSystemPrompt returns the system prompt.
func (m *Manager) GetSystemPrompt() *llm.Message {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.system == nil {
		return nil
	}
	msg := *m.system
	return &msg
}

// AddMessage adds a message to the context.
func (m *Manager) AddMessage(msg llm.Message) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// 估算新消息的 Token
	msgTokens := m.tokenizer.EstimateMessage(msg)
	maxUsable := m.config.ContextWindow - m.config.ReserveTokens
	if msgTokens > maxUsable {
		return fmt.Errorf("single message too large for context budget: %d tokens > %d", msgTokens, maxUsable)
	}

	// 先追加，再按真实占用触发后置压缩
	m.messages = append(m.messages, msg)

	// 后置压缩：基于最新上下文做决策，避免仅靠预估触发
	if m.shouldCompactLocked(0) {
		if err := m.compactLocked(); err != nil {
			return fmt.Errorf("compact context: %w", err)
		}
	}

	m.recalculateUsage()
	m.stats.MessageCount++
	if msg.Role == "tool" {
		m.stats.ToolCallCount++
	}

	return nil
}

// AddToolResult adds a tool result to the context.
func (m *Manager) AddToolResult(callID, content string) error {
	return m.AddMessage(llm.NewToolMessage(callID, content))
}

// GetMessages returns all messages including system prompt.
func (m *Manager) GetMessages() []llm.Message {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]llm.Message, 0, len(m.messages)+1)
	if m.system != nil {
		result = append(result, *m.system)
	}
	result = append(result, m.messages...)
	return result
}

// GetNonSystemMessages returns only non-system messages.
func (m *Manager) GetNonSystemMessages() []llm.Message {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]llm.Message, len(m.messages))
	copy(result, m.messages)
	return result
}

// SetNonSystemMessages replaces all non-system messages.
func (m *Manager) SetNonSystemMessages(msgs []llm.Message) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.messages = make([]llm.Message, len(msgs))
	copy(m.messages, msgs)
	m.clearProviderTokenUsageLocked()

	m.stats.MessageCount = len(m.messages)
	m.stats.ToolCallCount = 0
	for _, msg := range m.messages {
		if msg.Role == "tool" {
			m.stats.ToolCallCount++
		}
	}

	m.recalculateUsage()
}

// Clear clears all messages except system prompt.
func (m *Manager) Clear() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.messages = make([]llm.Message, 0)
	m.clearProviderTokenUsageLocked()
	m.recalculateUsage()
}

// Compact manually triggers context compaction.
func (m *Manager) Compact() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	currentTokens := m.currentTokensLocked()
	if currentTokens == 0 {
		return nil
	}
	return m.compactToTargetLocked(currentTokens / 2)
}

// TokenUsage returns current token usage.
func (m *Manager) TokenUsage() TokenUsage {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.usage
}

// CompactCount returns how many times compaction has run.
func (m *Manager) CompactCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.stats.CompactCount
}

// SetContextWindowLimits updates the runtime context window limits.
func (m *Manager) SetContextWindowLimits(contextWindow, reserveTokens int) error {
	if contextWindow <= 0 {
		return fmt.Errorf("context window must be positive")
	}
	if reserveTokens < 0 {
		return fmt.Errorf("reserve tokens must be non-negative")
	}
	if reserveTokens >= contextWindow {
		return fmt.Errorf("reserve tokens must be less than context window")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.config.ContextWindow = contextWindow
	m.config.ReserveTokens = reserveTokens
	m.recalculateUsage()

	return nil
}

// SetPromptTokenUsage records provider-reported prompt tokens for the current context.
// Values <= 0 clear the provider usage and fall back to local estimation.
func (m *Manager) SetPromptTokenUsage(provider string, promptTokens int) {
	m.setProviderTokenUsage(provider, promptTokens, ProviderTokenScopePrompt)
}

// SetTotalTokenUsage records provider-reported total tokens for the current context.
// Values <= 0 clear the provider usage and fall back to local estimation.
func (m *Manager) SetTotalTokenUsage(provider string, totalTokens int) {
	m.setProviderTokenUsage(provider, totalTokens, ProviderTokenScopeTotal)
}

// SetProviderTokenUsage records the best provider-reported usage snapshot for the current context.
// When total tokens are available they are preferred over prompt-only snapshots.
func (m *Manager) SetProviderTokenUsage(provider string, usage llm.Usage) {
	switch {
	case usage.TotalTokens > 0:
		m.setProviderTokenUsage(provider, usage.TotalTokens, ProviderTokenScopeTotal)
	case usage.PromptTokens > 0:
		m.setProviderTokenUsage(provider, usage.PromptTokens, ProviderTokenScopePrompt)
	default:
		m.setProviderTokenUsage(provider, 0, "")
	}
}

func (m *Manager) setProviderTokenUsage(provider string, tokens int, scope ProviderTokenScope) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if tokens <= 0 {
		m.clearProviderTokenUsageLocked()
	} else {
		m.exactSnapshotTokens = tokens
		m.exactSnapshotEstimate = m.totalTokensLocked()
		m.exactSnapshotProvider = strings.TrimSpace(provider)
		m.exactSnapshotScope = scope
		m.hasExactSnapshot = true
	}

	m.recalculateUsage()
}

// RestoreProviderUsageSnapshot restores a persisted provider snapshot for the current context.
// The caller must restore the matching message set before invoking this method.
func (m *Manager) RestoreProviderUsageSnapshot(snapshot ProviderUsageSnapshot) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if snapshot.Tokens <= 0 {
		m.clearProviderTokenUsageLocked()
		m.recalculateUsage()
		return
	}

	localTotal := m.totalTokensLocked()
	localDelta := snapshot.LocalDelta
	if localDelta < 0 {
		localDelta = 0
	}
	exactEstimate := localTotal - localDelta
	if exactEstimate < 0 {
		exactEstimate = 0
	}

	scope := snapshot.TokenScope
	if scope != ProviderTokenScopeTotal {
		scope = ProviderTokenScopePrompt
	}

	m.exactSnapshotTokens = snapshot.Tokens
	m.exactSnapshotEstimate = exactEstimate
	m.exactSnapshotProvider = strings.TrimSpace(snapshot.Provider)
	m.exactSnapshotScope = scope
	m.hasExactSnapshot = true
	m.recalculateUsage()
}

// TokenUsageDetails returns current token usage together with its source metadata.
func (m *Manager) TokenUsageDetails() TokenUsageDetails {
	m.mu.RLock()
	defer m.mu.RUnlock()

	localEstimatedTotal := m.totalTokensLocked()
	details := TokenUsageDetails{
		TokenUsage:          m.usage,
		Source:              TokenUsageSourceLocalEstimate,
		LocalEstimatedTotal: localEstimatedTotal,
	}
	if !m.hasExactSnapshot {
		return details
	}

	localDelta := localEstimatedTotal - m.exactSnapshotEstimate
	if localDelta < 0 {
		localDelta = 0
	}
	details.Source = TokenUsageSourceProvider
	details.Provider = m.exactSnapshotProvider
	details.ProviderSnapshotTokens = m.exactSnapshotTokens
	details.ProviderTokenScope = m.exactSnapshotScope
	details.LocalDelta = localDelta
	return details
}

// EstimateTokens estimates token count for messages.
func (m *Manager) EstimateTokens(msgs []llm.Message) int {
	return m.tokenizer.EstimateMessages(msgs)
}

// IsWithinBudget checks if adding a message would exceed the context window.
func (m *Manager) IsWithinBudget(msg llm.Message) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	estimated := m.currentTokensLocked() + m.tokenizer.EstimateMessage(msg)
	return estimated <= m.maxUsableTokensLocked()
}

// GetStats returns context statistics.
func (m *Manager) GetStats() map[string]any {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return map[string]any{
		"total_messages":    len(m.messages),
		"has_system_prompt": m.system != nil,
		"token_usage":       m.usage,
		"context_window":    m.config.ContextWindow,
		"compact_count":     m.stats.CompactCount,
		"tool_call_count":   m.stats.ToolCallCount,
		"last_compact_at":   m.stats.LastCompactAt,
	}
}

// GetDetailedStats returns detailed statistics.
func (m *Manager) GetDetailedStats() map[string]any {
	m.mu.RLock()
	defer m.mu.RUnlock()

	stats := map[string]any{
		"messages": map[string]any{
			"total":     len(m.messages),
			"user":      m.countByRole("user"),
			"assistant": m.countByRole("assistant"),
			"tool":      m.countByRole("tool"),
		},
		"tokens": map[string]any{
			"current":        m.usage.Current,
			"context_window": m.usage.ContextWindow,
			"reserved":       m.usage.Reserved,
			"available":      m.usage.Available,
		},
		"stats": m.stats,
	}

	return stats
}

// shouldCompactLocked checks if compaction is needed (must hold lock).
func (m *Manager) shouldCompactLocked(additionalTokens int) bool {
	threshold := m.compactionThresholdPercentLocked()
	estimatedTokens := m.currentTokensLocked() + additionalTokens
	return float64(estimatedTokens) >= float64(m.config.ContextWindow)*(threshold/100.0)
}

// compactLocked compacts the context (must hold lock).
func (m *Manager) compactLocked() error {
	currentTokens := m.currentTokensLocked()
	if currentTokens == 0 || !m.shouldCompactLocked(0) {
		return nil
	}
	return m.compactToTargetLocked(m.compactionTargetTokensLocked())
}

func (m *Manager) compactToTargetLocked(targetTokens int) error {
	currentTokens := m.currentTokensLocked()
	if currentTokens == 0 {
		return nil
	}
	if targetTokens <= 0 {
		targetTokens = 1
	}
	if currentTokens <= targetTokens {
		return nil
	}

	compacted := m.messages
	if m.config.EnableSmartCompact && m.compactor != nil {
		next, result := m.compactor.Compact(m.messages, m.system, targetTokens)
		compacted = next
		_ = result // 可以在日志中记录
	} else {
		compacted = keepRecentMessagesWithinTotalBudget(m.messages, m.system, targetTokens, m.tokenizer)
	}

	compacted = keepRecentMessagesWithinTotalBudget(compacted, m.system, targetTokens, m.tokenizer)
	maxUsableTokens := m.maxUsableTokensLocked()
	if estimateMessagesWithSystem(compacted, m.system, m.tokenizer) > maxUsableTokens {
		compacted = keepRecentMessagesWithinTotalBudget(compacted, m.system, maxUsableTokens, m.tokenizer)
	}

	newTokens := estimateMessagesWithSystem(compacted, m.system, m.tokenizer)
	if newTokens >= currentTokens {
		return nil
	}

	m.messages = compacted
	m.clearProviderTokenUsageLocked()
	m.stats.CompactCount++
	now := time.Now()
	m.stats.LastCompactAt = &now
	m.recalculateUsage()
	return nil
}

func (m *Manager) compactionThresholdPercentLocked() float64 {
	threshold := m.config.CompactionThreshold
	switch {
	case threshold <= 0:
		return 90.0
	case threshold <= 1:
		return threshold * 100
	default:
		// 兼容旧配置：允许直接填写百分比（0-100）
		if threshold > 100 {
			return 100
		}
		return threshold
	}
}

func (m *Manager) compactionTargetTokensLocked() int {
	targetTokens := int(float64(m.config.ContextWindow) * compactionTargetRatio)
	maxUsableTokens := m.maxUsableTokensLocked()
	if targetTokens <= 0 || targetTokens > maxUsableTokens {
		targetTokens = maxUsableTokens
	}
	if targetTokens < 0 {
		return 0
	}
	return targetTokens
}

// recalculateUsage recalculates token usage (must hold lock).
func (m *Manager) recalculateUsage() {
	total := m.currentTokensLocked()

	m.usage = TokenUsage{
		Current:       total,
		ContextWindow: m.config.ContextWindow,
		Reserved:      m.config.ReserveTokens,
		Available:     m.config.ContextWindow - total - m.config.ReserveTokens,
	}

	m.stats.TotalTokensUsed = total
}

func (m *Manager) totalTokensLocked() int {
	total := m.tokenizer.EstimateMessages(m.messages)
	if m.system != nil {
		total += m.tokenizer.EstimateMessage(*m.system)
	}
	return total
}

func (m *Manager) currentTokensLocked() int {
	total := m.totalTokensLocked()
	if !m.hasExactSnapshot {
		return total
	}

	delta := total - m.exactSnapshotEstimate
	if delta < 0 {
		return total
	}
	return m.exactSnapshotTokens + delta
}

func (m *Manager) clearProviderTokenUsageLocked() {
	m.exactSnapshotTokens = 0
	m.exactSnapshotEstimate = 0
	m.exactSnapshotProvider = ""
	m.exactSnapshotScope = ""
	m.hasExactSnapshot = false
}

func (m *Manager) maxUsableTokensLocked() int {
	return m.config.ContextWindow - m.config.ReserveTokens
}

// countByRole counts messages by role (must hold lock).
func (m *Manager) countByRole(role string) int {
	count := 0
	for _, msg := range m.messages {
		if msg.Role == role {
			count++
		}
	}
	return count
}

// SetCompactStrategy sets the compaction strategy.
func (m *Manager) SetCompactStrategy(s CompactStrategy) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.config.CompactStrategy = s
	if m.compactor != nil {
		m.compactor.SetStrategy(s)
	}
}

// GetMessagePriority returns the priority of a message.
func (m *Manager) GetMessagePriority(index int) Priority {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if index < 0 || index >= len(m.messages) {
		return PriorityLow
	}

	return m.scorer.ScoreMessage(m.messages[index], index, len(m.messages))
}

// TruncateTo truncates messages to the specified count (keeping the most recent).
func (m *Manager) TruncateTo(count int) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if count < 0 {
		count = 0
	}
	if count >= len(m.messages) {
		return
	}

	m.messages = keepRecentMessages(m.messages, count)
	m.clearProviderTokenUsageLocked()
	m.recalculateUsage()
}

func estimateMessagesWithSystem(messages []llm.Message, systemMsg *llm.Message, tokenizer *Tokenizer) int {
	total := tokenizer.EstimateMessages(messages)
	if systemMsg != nil {
		total += tokenizer.EstimateMessage(*systemMsg)
	}
	return total
}

func keepRecentMessagesWithinTotalBudget(messages []llm.Message, systemMsg *llm.Message, targetTokens int, tokenizer *Tokenizer) []llm.Message {
	messageBudget := targetTokens
	if systemMsg != nil {
		messageBudget -= tokenizer.EstimateMessage(*systemMsg)
	}
	if messageBudget <= 0 {
		return nil
	}
	return keepRecentMessagesByTokens(messages, messageBudget, tokenizer)
}
