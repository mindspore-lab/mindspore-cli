package context

import (
	"strings"
	"testing"

	"github.com/vigo999/mindspore-cli/integrations/llm"
)

func TestNewManager(t *testing.T) {
	cfg := DefaultManagerConfig()
	mgr := NewManager(cfg)

	if mgr == nil {
		t.Fatal("NewManager returned nil")
	}

	if mgr.config.ContextWindow != 24000 {
		t.Errorf("Expected ContextWindow to be 24000, got %d", mgr.config.ContextWindow)
	}

	if mgr.tokenizer == nil {
		t.Error("Tokenizer should be initialized")
	}

	if mgr.compactor == nil {
		t.Error("Compactor should be initialized")
	}
}

func TestSetSystemPrompt(t *testing.T) {
	mgr := NewManager(DefaultManagerConfig())

	prompt := "You are a helpful assistant."
	mgr.SetSystemPrompt(prompt)

	systemMsg := mgr.GetSystemPrompt()
	if systemMsg == nil {
		t.Fatal("System prompt should not be nil")
	}

	if systemMsg.Content != prompt {
		t.Errorf("Expected system prompt '%s', got '%s'", prompt, systemMsg.Content)
	}

	if systemMsg.Role != "system" {
		t.Errorf("Expected role 'system', got '%s'", systemMsg.Role)
	}
}

func TestAddMessage(t *testing.T) {
	mgr := NewManager(DefaultManagerConfig())

	msg := llm.NewUserMessage("Hello")
	err := mgr.AddMessage(msg)
	if err != nil {
		t.Fatalf("AddMessage failed: %v", err)
	}

	messages := mgr.GetNonSystemMessages()
	if len(messages) != 1 {
		t.Errorf("Expected 1 message, got %d", len(messages))
	}

	if messages[0].Content != "Hello" {
		t.Errorf("Expected message content 'Hello', got '%s'", messages[0].Content)
	}
}

func TestAddToolResult(t *testing.T) {
	mgr := NewManager(DefaultManagerConfig())

	err := mgr.AddToolResult("call_123", "Result content")
	if err != nil {
		t.Fatalf("AddToolResult failed: %v", err)
	}

	messages := mgr.GetNonSystemMessages()
	if len(messages) != 1 {
		t.Errorf("Expected 1 message, got %d", len(messages))
	}

	if messages[0].Role != "tool" {
		t.Errorf("Expected role 'tool', got '%s'", messages[0].Role)
	}

	if messages[0].ToolCallID != "call_123" {
		t.Errorf("Expected ToolCallID 'call_123', got '%s'", messages[0].ToolCallID)
	}
}

func TestGetMessages(t *testing.T) {
	mgr := NewManager(DefaultManagerConfig())

	// Set system prompt
	mgr.SetSystemPrompt("System prompt")

	// Add user message
	mgr.AddMessage(llm.NewUserMessage("Hello"))

	// Get all messages
	messages := mgr.GetMessages()

	if len(messages) != 2 {
		t.Errorf("Expected 2 messages (system + user), got %d", len(messages))
	}

	if messages[0].Role != "system" {
		t.Errorf("First message should be system, got %s", messages[0].Role)
	}

	if messages[1].Role != "user" {
		t.Errorf("Second message should be user, got %s", messages[1].Role)
	}
}

func TestClear(t *testing.T) {
	mgr := NewManager(DefaultManagerConfig())

	mgr.SetSystemPrompt("System")
	mgr.AddMessage(llm.NewUserMessage("Hello"))
	mgr.AddMessage(llm.NewAssistantMessage("Hi"))

	mgr.Clear()

	nonSystem := mgr.GetNonSystemMessages()
	if len(nonSystem) != 0 {
		t.Errorf("Expected 0 non-system messages after clear, got %d", len(nonSystem))
	}

	// System prompt should still exist
	system := mgr.GetSystemPrompt()
	if system == nil {
		t.Error("System prompt should persist after Clear()")
	}
}

func TestTokenUsage(t *testing.T) {
	mgr := NewManager(DefaultManagerConfig())

	initialUsage := mgr.TokenUsage()
	if initialUsage.Current != 0 {
		t.Errorf("Expected initial usage to be 0, got %d", initialUsage.Current)
	}

	// Add messages
	mgr.AddMessage(llm.NewUserMessage("Hello world"))

	usage := mgr.TokenUsage()
	if usage.Current == 0 {
		t.Error("Token usage should increase after adding message")
	}

	if usage.ContextWindow != 24000 {
		t.Errorf("Expected ContextWindow to be 24000, got %d", usage.ContextWindow)
	}
}

func TestSetContextWindowLimits(t *testing.T) {
	mgr := NewManager(DefaultManagerConfig())
	mgr.SetSystemPrompt("system prompt")
	if err := mgr.AddMessage(llm.NewUserMessage("hello world")); err != nil {
		t.Fatalf("AddMessage failed: %v", err)
	}

	if err := mgr.SetContextWindowLimits(200000, 4000); err != nil {
		t.Fatalf("SetContextWindowLimits failed: %v", err)
	}

	usage := mgr.TokenUsage()
	if got, want := usage.ContextWindow, 200000; got != want {
		t.Fatalf("usage.ContextWindow = %d, want %d", got, want)
	}
	if got, want := usage.Reserved, 4000; got != want {
		t.Fatalf("usage.Reserved = %d, want %d", got, want)
	}
}

func TestIsWithinBudget(t *testing.T) {
	cfg := DefaultManagerConfig()
	cfg.ContextWindow = 100
	cfg.ReserveTokens = 20

	mgr := NewManager(cfg)

	// Small message should be within budget
	smallMsg := llm.NewUserMessage("Hi")
	if !mgr.IsWithinBudget(smallMsg) {
		t.Error("Small message should be within budget")
	}
}

func TestCompactionThresholdSupportsRatioAndPercent(t *testing.T) {
	cfgRatio := DefaultManagerConfig()
	cfgRatio.CompactionThreshold = 0.85
	mgrRatio := NewManager(cfgRatio)
	mgrRatio.mu.Lock()
	if got := mgrRatio.compactionThresholdPercentLocked(); got != 85 {
		mgrRatio.mu.Unlock()
		t.Fatalf("expected 85%% threshold for ratio config, got %.2f", got)
	}
	mgrRatio.mu.Unlock()

	cfgPercent := DefaultManagerConfig()
	cfgPercent.CompactionThreshold = 85
	mgrPercent := NewManager(cfgPercent)
	mgrPercent.mu.Lock()
	if got := mgrPercent.compactionThresholdPercentLocked(); got != 85 {
		mgrPercent.mu.Unlock()
		t.Fatalf("expected 85%% threshold for percent config, got %.2f", got)
	}
	mgrPercent.mu.Unlock()
}

func TestAddMessageRejectsSingleOversizedMessage(t *testing.T) {
	cfg := DefaultManagerConfig()
	cfg.ContextWindow = 100
	cfg.ReserveTokens = 20
	mgr := NewManager(cfg)

	oversized := llm.NewToolMessage("call_1", strings.Repeat("x", 1000)) // ~250 tokens
	if err := mgr.AddMessage(oversized); err == nil {
		t.Fatal("expected oversized message to be rejected")
	}
}

func TestEstimateTokens(t *testing.T) {
	mgr := NewManager(DefaultManagerConfig())

	msgs := []llm.Message{
		llm.NewUserMessage("Hello"),
		llm.NewAssistantMessage("World"),
	}

	tokens := mgr.EstimateTokens(msgs)
	if tokens <= 0 {
		t.Error("Estimated tokens should be positive")
	}
}

func TestGetStats(t *testing.T) {
	mgr := NewManager(DefaultManagerConfig())
	mgr.SetSystemPrompt("System")
	mgr.AddMessage(llm.NewUserMessage("Hello"))

	stats := mgr.GetStats()

	if stats["total_messages"] != 1 {
		t.Errorf("Expected 1 message in stats, got %v", stats["total_messages"])
	}

	if stats["has_system_prompt"] != true {
		t.Error("Expected has_system_prompt to be true")
	}
}

func TestCompactManuallyHalvesCurrentUsage(t *testing.T) {
	cfg := DefaultManagerConfig()
	cfg.ContextWindow = 100
	cfg.ReserveTokens = 10
	cfg.EnableSmartCompact = false

	mgr := NewManager(cfg)

	for i := 0; i < 3; i++ {
		if err := mgr.AddMessage(llm.NewUserMessage(strings.Repeat("x", 80))); err != nil {
			t.Fatalf("AddMessage failed: %v", err)
		}
	}

	beforeUsage := mgr.TokenUsage().Current
	beforeCount := len(mgr.GetNonSystemMessages())
	err := mgr.Compact()
	if err != nil {
		t.Fatalf("Compact failed: %v", err)
	}

	if got := mgr.TokenUsage().Current; got > beforeUsage/2 {
		t.Fatalf("token usage after manual Compact = %d, want <= %d", got, beforeUsage/2)
	}
	if got := len(mgr.GetNonSystemMessages()); got >= beforeCount {
		t.Fatalf("message count after manual Compact = %d, want less than %d", got, beforeCount)
	}
}

func TestAddMessageCompactsToTargetAfterThresholdExceeded(t *testing.T) {
	cfg := DefaultManagerConfig()
	cfg.ContextWindow = 100
	cfg.ReserveTokens = 10
	cfg.EnableSmartCompact = false

	mgr := NewManager(cfg)

	for i := 0; i < 3; i++ {
		if err := mgr.AddMessage(llm.NewUserMessage(strings.Repeat("x", 80))); err != nil {
			t.Fatalf("AddMessage #%d failed: %v", i+1, err)
		}
	}

	if got := mgr.TokenUsage().Current; got >= 90 {
		t.Fatalf("token usage before threshold test = %d, want below 90", got)
	}

	if err := mgr.AddMessage(llm.NewUserMessage(strings.Repeat("x", 80))); err != nil {
		t.Fatalf("AddMessage triggering compaction failed: %v", err)
	}

	if got := mgr.TokenUsage().Current; got > 50 {
		t.Fatalf("token usage after compaction = %d, want <= 50", got)
	}
	if got := len(mgr.GetNonSystemMessages()); got >= 4 {
		t.Fatalf("message count after compaction = %d, want fewer than 4", got)
	}
}

func TestSetCompactStrategy(t *testing.T) {
	mgr := NewManager(DefaultManagerConfig())

	mgr.SetCompactStrategy(CompactStrategySummarize)

	if mgr.config.CompactStrategy != CompactStrategySummarize {
		t.Errorf("Expected strategy to be CompactStrategySummarize, got %v", mgr.config.CompactStrategy)
	}
}

func TestGetMessagePriority(t *testing.T) {
	mgr := NewManager(DefaultManagerConfig())
	mgr.SetSystemPrompt("System")
	mgr.AddMessage(llm.NewUserMessage("Hello"))

	priority := mgr.GetMessagePriority(0)
	if priority <= 0 {
		t.Error("Priority should be positive")
	}
}

func TestTruncateTo(t *testing.T) {
	mgr := NewManager(DefaultManagerConfig())

	// Add messages
	for i := 0; i < 5; i++ {
		mgr.AddMessage(llm.NewUserMessage("Message"))
	}

	mgr.TruncateTo(2)

	messages := mgr.GetNonSystemMessages()
	if len(messages) != 2 {
		t.Errorf("Expected 2 messages after truncate, got %d", len(messages))
	}
}

func TestPriorityScorer(t *testing.T) {
	scorer := NewPriorityScorer()

	// System message should have high priority
	systemMsg := llm.NewSystemMessage("System prompt")
	priority := scorer.ScoreMessage(systemMsg, 0, 1)

	if priority < PriorityHigh {
		t.Errorf("System message should have high priority, got %d", priority)
	}

	// User message
	userMsg := llm.NewUserMessage("Hello")
	priority = scorer.ScoreMessage(userMsg, 0, 1)

	if priority < PriorityMedium {
		t.Errorf("User message should have at least medium priority, got %d", priority)
	}
}

func TestCompactResult(t *testing.T) {
	result := CompactResult{
		Kept:     5,
		Removed:  3,
		Strategy: CompactStrategySimple,
		Summary:  "Test summary",
	}

	str := result.String()
	if str == "" {
		t.Error("CompactResult.String() should not be empty")
	}
}
