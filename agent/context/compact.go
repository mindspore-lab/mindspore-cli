package context

import (
	"fmt"
	"sort"
	"strings"

	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
)

// CompactStrategy 压缩策略类型
type CompactStrategy int

const (
	// CompactStrategySimple 简单策略：直接丢弃旧消息
	CompactStrategySimple CompactStrategy = iota
	// CompactStrategySummarize 摘要策略：将旧消息摘要为一句话
	CompactStrategySummarize
	// CompactStrategyPriority 优先级策略：基于优先级保留消息
	CompactStrategyPriority
	// CompactStrategyHybrid 混合策略：结合多种策略
	CompactStrategyHybrid
)

// String 返回策略名称
func (s CompactStrategy) String() string {
	switch s {
	case CompactStrategySimple:
		return "simple"
	case CompactStrategySummarize:
		return "summarize"
	case CompactStrategyPriority:
		return "priority"
	case CompactStrategyHybrid:
		return "hybrid"
	default:
		return "unknown"
	}
}

// ParseCompactStrategy 解析策略字符串
func ParseCompactStrategy(s string) CompactStrategy {
	switch strings.ToLower(s) {
	case "simple":
		return CompactStrategySimple
	case "summarize":
		return CompactStrategySummarize
	case "priority":
		return CompactStrategyPriority
	case "hybrid":
		return CompactStrategyHybrid
	default:
		return CompactStrategySimple
	}
}

// Compactor 上下文压缩器
type Compactor struct {
	strategy  CompactStrategy
	scorer    *PriorityScorer
	tokenizer *Tokenizer
}

type messageGroup struct {
	Messages []llm.Message
	Start    int
}

type prioritizedGroup struct {
	Group    messageGroup
	Priority Priority
}

// CompactorConfig 压缩器配置
type CompactorConfig struct {
	Strategy CompactStrategy
}

// NewCompactor 创建新的压缩器
func NewCompactor(cfg CompactorConfig) *Compactor {
	return &Compactor{
		strategy:  cfg.Strategy,
		scorer:    NewPriorityScorer(),
		tokenizer: NewTokenizer(),
	}
}

// SetStrategy 设置压缩策略
func (c *Compactor) SetStrategy(s CompactStrategy) {
	c.strategy = s
}

// Compact 执行压缩，目标是将总 token 占用降到 targetTokens 以内。
func (c *Compactor) Compact(messages []llm.Message, systemMsg *llm.Message, targetTokens int) ([]llm.Message, CompactResult) {
	if targetTokens <= 0 {
		return nil, CompactResult{Kept: 0, Removed: len(messages)}
	}
	if c.totalTokens(messages, systemMsg) <= targetTokens {
		return messages, CompactResult{Kept: len(messages), Removed: 0}
	}

	switch c.strategy {
	case CompactStrategySimple:
		return c.compactSimple(messages, systemMsg, targetTokens)
	case CompactStrategySummarize:
		return c.compactSummarize(messages, systemMsg, targetTokens)
	case CompactStrategyPriority:
		return c.compactPriority(messages, systemMsg, targetTokens)
	case CompactStrategyHybrid:
		return c.compactHybrid(messages, systemMsg, targetTokens)
	default:
		return c.compactSimple(messages, systemMsg, targetTokens)
	}
}

// compactSimple 简单压缩策略
func (c *Compactor) compactSimple(messages []llm.Message, systemMsg *llm.Message, targetTokens int) ([]llm.Message, CompactResult) {
	messageBudget := c.messageTokenBudget(targetTokens, systemMsg)
	result := keepRecentMessagesByTokens(messages, messageBudget, c.tokenizer)
	removed := len(messages) - len(result)

	return result, CompactResult{
		Kept:     len(result),
		Removed:  removed,
		Strategy: CompactStrategySimple,
		Summary:  fmt.Sprintf("Removed %d old messages", removed),
	}
}

// compactSummarize 摘要压缩策略
func (c *Compactor) compactSummarize(messages []llm.Message, systemMsg *llm.Message, targetTokens int) ([]llm.Message, CompactResult) {
	messageBudget := c.messageTokenBudget(targetTokens, systemMsg)
	if messageBudget <= 0 {
		return nil, CompactResult{Kept: 0, Removed: len(messages), Strategy: CompactStrategySummarize}
	}
	groups := groupMessages(messages)
	keptGroups := keepRecentMessageGroupsByTokens(groups, messageBudget, c.tokenizer)
	toSummarize := flattenMessageGroups(excludeMessageGroups(groups, keptGroups))
	if len(toSummarize) == 0 {
		kept := flattenMessageGroups(keptGroups)
		return kept, CompactResult{
			Kept:     len(kept),
			Removed:  0,
			Strategy: CompactStrategySummarize,
			Summary:  "No messages summarized",
		}
	}

	summary := c.generateSummary(toSummarize)
	summaryMsg := llm.NewSystemMessage(summary)
	keepBudget := messageBudget - c.tokenizer.EstimateMessage(summaryMsg)
	if keepBudget < 0 {
		keepBudget = 0
	}

	keptGroups = keepRecentMessageGroupsByTokens(groups, keepBudget, c.tokenizer)
	toSummarize = flattenMessageGroups(excludeMessageGroups(groups, keptGroups))
	result := flattenMessageGroups(keptGroups)
	if len(toSummarize) > 0 {
		summary = c.generateSummary(toSummarize)
		summaryMsg = llm.NewSystemMessage(summary)
		if c.tokenizer.EstimateMessage(summaryMsg) <= messageBudget {
			result = append([]llm.Message{summaryMsg}, result...)
		}
	}
	if len(result) == 0 {
		result = keepRecentMessagesByTokens(messages, messageBudget, c.tokenizer)
	}

	return result, CompactResult{
		Kept:     len(result),
		Removed:  len(toSummarize),
		Strategy: CompactStrategySummarize,
		Summary:  summary,
	}
}

// compactPriority 优先级压缩策略
func (c *Compactor) compactPriority(messages []llm.Message, systemMsg *llm.Message, targetTokens int) ([]llm.Message, CompactResult) {
	messageBudget := c.messageTokenBudget(targetTokens, systemMsg)
	prioritized := c.prioritizeGroups(groupMessages(messages), len(messages))
	result := flattenMessageGroups(selectPrioritizedGroupsByTokens(prioritized, messageBudget, c.tokenizer))

	return result, CompactResult{
		Kept:     len(result),
		Removed:  len(messages) - len(result),
		Strategy: CompactStrategyPriority,
		Summary:  fmt.Sprintf("Kept %d high-priority messages, removed %d", len(result), len(messages)-len(result)),
	}
}

// compactHybrid 混合压缩策略
func (c *Compactor) compactHybrid(messages []llm.Message, systemMsg *llm.Message, targetTokens int) ([]llm.Message, CompactResult) {
	// 策略：
	// 1. 保留最近的几条消息（高优先级）
	// 2. 基于优先级选择保留的较旧消息
	// 3. 将其他旧消息摘要

	messageBudget := c.messageTokenBudget(targetTokens, systemMsg)
	if messageBudget <= 0 {
		return nil, CompactResult{Kept: 0, Removed: len(messages), Strategy: CompactStrategyHybrid}
	}

	groups := groupMessages(messages)
	recentBudget := messageBudget / 2
	if recentBudget <= 0 {
		recentBudget = messageBudget
	}
	recentGroups := keepRecentMessageGroupsByTokens(groups, recentBudget, c.tokenizer)
	oldGroups := excludeMessageGroups(groups, recentGroups)
	prioritized := c.prioritizeGroups(oldGroups, len(messages))

	var result []llm.Message
	remainingBudget := messageBudget - countTokensInGroups(recentGroups, c.tokenizer)
	if remainingBudget < 0 {
		remainingBudget = 0
	}
	highPriorityOld := selectPrioritizedGroupsByTokens(prioritized, remainingBudget, c.tokenizer)
	remainingBudget -= countTokensInGroups(highPriorityOld, c.tokenizer)

	// 如果有需要摘要的旧消息，添加摘要
	if len(oldGroups) > len(highPriorityOld) && remainingBudget > 0 {
		toSummarize := flattenMessageGroups(excludeMessageGroups(oldGroups, highPriorityOld))
		if len(toSummarize) > 0 {
			summary := c.generateSummary(toSummarize)
			summaryMsg := llm.NewSystemMessage(summary)
			if c.tokenizer.EstimateMessage(summaryMsg) <= remainingBudget {
				result = append(result, summaryMsg)
			}
		}
	}

	// 添加保留的高优先级旧消息
	if len(highPriorityOld) > 0 {
		result = append(result, flattenMessageGroups(highPriorityOld)...)
	}

	// 添加最近的消息
	result = append(result, flattenMessageGroups(recentGroups)...)

	removed := len(messages) - len(result)

	return result, CompactResult{
		Kept:     len(result),
		Removed:  removed,
		Strategy: CompactStrategyHybrid,
		Summary:  fmt.Sprintf("Hybrid compact: kept %d messages", len(result)),
	}
}

func (c *Compactor) totalTokens(messages []llm.Message, systemMsg *llm.Message) int {
	total := c.tokenizer.EstimateMessages(messages)
	if systemMsg != nil {
		total += c.tokenizer.EstimateMessage(*systemMsg)
	}
	return total
}

func (c *Compactor) messageTokenBudget(targetTokens int, systemMsg *llm.Message) int {
	budget := targetTokens
	if systemMsg != nil {
		budget -= c.tokenizer.EstimateMessage(*systemMsg)
	}
	if budget < 0 {
		return 0
	}
	return budget
}

// generateSummary 生成消息摘要
func (c *Compactor) generateSummary(messages []llm.Message) string {
	userCount := 0
	assistantCount := 0
	toolCount := 0

	for _, msg := range messages {
		switch msg.Role {
		case "user":
			userCount++
		case "assistant":
			assistantCount++
		case "tool":
			toolCount++
		}
	}

	parts := []string{"[Context Summary]"}
	parts = append(parts, fmt.Sprintf("Earlier conversation: %d messages", len(messages)))

	if userCount > 0 {
		parts = append(parts, fmt.Sprintf("%d user messages", userCount))
	}
	if assistantCount > 0 {
		parts = append(parts, fmt.Sprintf("%d assistant responses", assistantCount))
	}
	if toolCount > 0 {
		parts = append(parts, fmt.Sprintf("%d tool calls", toolCount))
	}

	return strings.Join(parts, ", ")
}

// CompactResult 压缩结果
type CompactResult struct {
	Kept     int
	Removed  int
	Strategy CompactStrategy
	Summary  string
}

// String 返回压缩结果的字符串表示
func (r CompactResult) String() string {
	return fmt.Sprintf("Compact [%s]: kept %d, removed %d - %s",
		r.Strategy, r.Kept, r.Removed, r.Summary)
}

// SimpleCompact 简单压缩函数（保持向后兼容）
func SimpleCompact(messages []llm.Message, maxKeep int) []llm.Message {
	if len(messages) <= maxKeep {
		return messages
	}
	return keepRecentMessages(messages, maxKeep)
}

func groupMessages(messages []llm.Message) []messageGroup {
	if len(messages) == 0 {
		return nil
	}

	groups := make([]messageGroup, 0, len(messages))
	for i := 0; i < len(messages); {
		start := i
		i++
		if messages[start].Role == "assistant" && len(messages[start].ToolCalls) > 0 {
			for i < len(messages) && messages[i].Role == "tool" {
				i++
			}
		}
		groups = append(groups, messageGroup{
			Messages: messages[start:i],
			Start:    start,
		})
	}

	return groups
}

func flattenMessageGroups(groups []messageGroup) []llm.Message {
	if len(groups) == 0 {
		return nil
	}

	total := 0
	for _, group := range groups {
		total += len(group.Messages)
	}

	flattened := make([]llm.Message, 0, total)
	for _, group := range groups {
		flattened = append(flattened, group.Messages...)
	}
	return flattened
}

func keepRecentMessages(messages []llm.Message, maxKeep int) []llm.Message {
	if maxKeep <= 0 {
		return nil
	}
	return flattenMessageGroups(keepRecentMessageGroups(groupMessages(messages), maxKeep))
}

func keepRecentMessagesByTokens(messages []llm.Message, maxTokens int, tokenizer *Tokenizer) []llm.Message {
	if maxTokens <= 0 {
		return nil
	}
	return flattenMessageGroups(keepRecentMessageGroupsByTokens(groupMessages(messages), maxTokens, tokenizer))
}

func keepRecentMessageGroups(groups []messageGroup, maxKeep int) []messageGroup {
	if len(groups) == 0 || maxKeep <= 0 {
		return nil
	}

	pinned := pinnedMessageGroups(groups)
	pinnedSet := messageGroupSet(pinned)
	kept := append([]messageGroup{}, pinned...)
	remainingBudget := maxKeep - countMessagesInGroups(pinned)
	if remainingBudget <= 0 {
		sortMessageGroupsByStart(kept)
		return kept
	}

	recent := make([]messageGroup, 0, len(groups))
	keptMessages := 0
	for i := len(groups) - 1; i >= 0; i-- {
		if _, ok := pinnedSet[groups[i].Start]; ok {
			continue
		}
		groupSize := len(groups[i].Messages)
		if keptMessages+groupSize > remainingBudget && len(recent) > 0 {
			break
		}
		recent = append(recent, groups[i])
		keptMessages += groupSize
		if keptMessages >= remainingBudget {
			break
		}
	}

	kept = append(kept, recent...)
	sortMessageGroupsByStart(kept)
	return kept
}

func keepRecentMessageGroupsByTokens(groups []messageGroup, maxTokens int, tokenizer *Tokenizer) []messageGroup {
	if len(groups) == 0 {
		return nil
	}

	pinned := pinnedMessageGroups(groups)
	pinnedSet := messageGroupSet(pinned)
	kept := append([]messageGroup{}, pinned...)
	remainingBudget := maxTokens - countTokensInGroups(pinned, tokenizer)
	if remainingBudget <= 0 {
		sortMessageGroupsByStart(kept)
		return kept
	}

	recent := make([]messageGroup, 0, len(groups))
	usedTokens := 0
	for i := len(groups) - 1; i >= 0; i-- {
		if _, ok := pinnedSet[groups[i].Start]; ok {
			continue
		}
		groupTokens := estimateGroupTokens(groups[i], tokenizer)
		if groupTokens > remainingBudget-usedTokens {
			if len(recent) > 0 {
				break
			}
			recent = append(recent, groups[i])
			break
		}
		recent = append(recent, groups[i])
		usedTokens += groupTokens
		if usedTokens >= remainingBudget {
			break
		}
	}

	kept = append(kept, recent...)
	sortMessageGroupsByStart(kept)
	return kept
}

func (c *Compactor) prioritizeGroups(groups []messageGroup, totalMessages int) []prioritizedGroup {
	prioritized := make([]prioritizedGroup, 0, len(groups))
	for _, group := range groups {
		priority := PriorityLow
		for offset, msg := range group.Messages {
			score := c.scorer.ScoreMessage(msg, group.Start+offset, totalMessages)
			if score > priority {
				priority = score
			}
		}
		prioritized = append(prioritized, prioritizedGroup{
			Group:    group,
			Priority: priority,
		})
	}

	sort.Slice(prioritized, func(i, j int) bool {
		if prioritized[i].Priority == prioritized[j].Priority {
			return prioritized[i].Group.Start > prioritized[j].Group.Start
		}
		return prioritized[i].Priority > prioritized[j].Priority
	})
	return prioritized
}

func selectPrioritizedGroups(groups []prioritizedGroup, maxKeep int) []messageGroup {
	if len(groups) == 0 {
		return nil
	}

	pinned := make([]messageGroup, 0)
	pinnedSet := make(map[int]struct{})
	for _, group := range groups {
		if !isPinnedMessageGroup(group.Group) {
			continue
		}
		pinned = append(pinned, group.Group)
		pinnedSet[group.Group.Start] = struct{}{}
	}

	selected := append([]messageGroup{}, pinned...)
	remainingBudget := maxKeep - countMessagesInGroups(pinned)
	if remainingBudget <= 0 {
		sortMessageGroupsByStart(selected)
		return selected
	}

	keptMessages := 0
	for _, group := range groups {
		if _, ok := pinnedSet[group.Group.Start]; ok {
			continue
		}
		groupSize := len(group.Group.Messages)
		if keptMessages+groupSize > remainingBudget && len(selected) > len(pinned) {
			continue
		}
		selected = append(selected, group.Group)
		keptMessages += groupSize
		if keptMessages >= remainingBudget {
			break
		}
	}

	sortMessageGroupsByStart(selected)
	return selected
}

func selectPrioritizedGroupsByTokens(groups []prioritizedGroup, maxTokens int, tokenizer *Tokenizer) []messageGroup {
	if len(groups) == 0 {
		return nil
	}

	pinned := make([]messageGroup, 0)
	pinnedSet := make(map[int]struct{})
	for _, group := range groups {
		if !isPinnedMessageGroup(group.Group) {
			continue
		}
		pinned = append(pinned, group.Group)
		pinnedSet[group.Group.Start] = struct{}{}
	}

	selected := append([]messageGroup{}, pinned...)
	remainingBudget := maxTokens - countTokensInGroups(pinned, tokenizer)
	if remainingBudget <= 0 {
		sortMessageGroupsByStart(selected)
		return selected
	}

	usedTokens := 0
	for _, group := range groups {
		if _, ok := pinnedSet[group.Group.Start]; ok {
			continue
		}
		groupTokens := estimateGroupTokens(group.Group, tokenizer)
		if groupTokens > remainingBudget-usedTokens {
			if len(selected) > len(pinned) {
				continue
			}
			selected = append(selected, group.Group)
			break
		}
		selected = append(selected, group.Group)
		usedTokens += groupTokens
		if usedTokens >= remainingBudget {
			break
		}
	}

	sortMessageGroupsByStart(selected)
	return selected
}

func excludeMessageGroups(groups, excluded []messageGroup) []messageGroup {
	if len(groups) == 0 {
		return nil
	}
	if len(excluded) == 0 {
		result := make([]messageGroup, len(groups))
		copy(result, groups)
		return result
	}

	excludedSet := messageGroupSet(excluded)
	result := make([]messageGroup, 0, len(groups))
	for _, group := range groups {
		if _, ok := excludedSet[group.Start]; ok {
			continue
		}
		result = append(result, group)
	}
	return result
}

func pinnedMessageGroups(groups []messageGroup) []messageGroup {
	result := make([]messageGroup, 0)
	for _, group := range groups {
		if isPinnedMessageGroup(group) {
			result = append(result, group)
		}
	}
	return result
}

func isPinnedMessageGroup(group messageGroup) bool {
	for _, msg := range group.Messages {
		for _, tc := range msg.ToolCalls {
			if tc.Function.Name == "load_skill" {
				return true
			}
		}
	}
	return false
}

func countMessagesInGroups(groups []messageGroup) int {
	total := 0
	for _, group := range groups {
		total += len(group.Messages)
	}
	return total
}

func estimateGroupTokens(group messageGroup, tokenizer *Tokenizer) int {
	return tokenizer.EstimateMessages(group.Messages)
}

func countTokensInGroups(groups []messageGroup, tokenizer *Tokenizer) int {
	total := 0
	for _, group := range groups {
		total += estimateGroupTokens(group, tokenizer)
	}
	return total
}

func messageGroupSet(groups []messageGroup) map[int]struct{} {
	result := make(map[int]struct{}, len(groups))
	for _, group := range groups {
		result[group.Start] = struct{}{}
	}
	return result
}

func sortMessageGroupsByStart(groups []messageGroup) {
	sort.Slice(groups, func(i, j int) bool {
		return groups[i].Start < groups[j].Start
	})
}
