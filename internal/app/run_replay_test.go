package app

import (
	"context"
	"testing"
	"time"

	agentctx "github.com/mindspore-lab/mindspore-cli/agent/context"
	"github.com/mindspore-lab/mindspore-cli/agent/session"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestReplayHistoryEmitsUsageSnapshotAfterBacklog(t *testing.T) {
	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 4096,
		ReserveTokens: 512,
	})
	ctxManager.SetSystemPrompt("system prompt")
	if err := ctxManager.AddMessage(llm.NewUserMessage("hello")); err != nil {
		t.Fatalf("add context message: %v", err)
	}

	expected := ctxManager.TokenUsage()
	eventCh := make(chan model.Event, 2)
	app := &Application{
		EventCh:       eventCh,
		ctxManager:    ctxManager,
		replayBacklog: []model.Event{{Type: model.UserInput, Message: "hello"}},
	}

	app.replayHistory()

	first := <-eventCh
	if first.Type != model.UserInput {
		t.Fatalf("first replay event type = %q, want %q", first.Type, model.UserInput)
	}

	second := <-eventCh
	if second.Type != model.TokenUpdate {
		t.Fatalf("second replay event type = %q, want %q", second.Type, model.TokenUpdate)
	}
	if second.CtxUsed != expected.Current {
		t.Fatalf("token update ctx used = %d, want %d", second.CtxUsed, expected.Current)
	}
	if second.CtxMax != expected.ContextWindow {
		t.Fatalf("token update ctx max = %d, want %d", second.CtxMax, expected.ContextWindow)
	}
}

func TestReplayHistoryTimelinePreservesRecordedDelays(t *testing.T) {
	previousWait := waitReplayDelay
	t.Cleanup(func() {
		waitReplayDelay = previousWait
	})

	var waits []time.Duration
	waitReplayDelay = func(ctx context.Context, d time.Duration) error {
		waits = append(waits, d)
		return nil
	}

	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 4096,
		ReserveTokens: 512,
	})
	ctxManager.SetSystemPrompt("system prompt")
	if err := ctxManager.AddMessage(llm.NewUserMessage("hello")); err != nil {
		t.Fatalf("add user context: %v", err)
	}
	if err := ctxManager.AddMessage(llm.NewAssistantMessage("hi")); err != nil {
		t.Fatalf("add assistant context: %v", err)
	}

	t0 := time.Date(2026, time.March, 27, 12, 0, 0, 0, time.UTC)
	eventCh := make(chan model.Event, 3)
	app := &Application{
		EventCh:    eventCh,
		ctxManager: ctxManager,
		replayOnly: true,
		replayTimeline: []session.ReplayFrame{
			{
				Timestamp: t0,
				Event:     model.Event{Type: model.UserInput, Message: "hello"},
			},
			{
				Timestamp: t0.Add(200 * time.Millisecond),
				Event:     model.Event{Type: model.AgentReply, Message: "hi"},
			},
		},
	}

	app.replayHistory()

	first := <-eventCh
	if first.Type != model.UserInput {
		t.Fatalf("first event type = %q, want %q", first.Type, model.UserInput)
	}
	second := <-eventCh
	if second.Type != model.AgentReply {
		t.Fatalf("second event type = %q, want %q", second.Type, model.AgentReply)
	}
	third := <-eventCh
	if third.Type != model.TokenUpdate {
		t.Fatalf("third event type = %q, want %q", third.Type, model.TokenUpdate)
	}
	if len(waits) != 1 {
		t.Fatalf("wait count = %d, want 1", len(waits))
	}
	if waits[0] != 200*time.Millisecond {
		t.Fatalf("wait duration = %v, want %v", waits[0], 200*time.Millisecond)
	}
}

func TestReplayHistoryTimelineShowsThinkingDuringLLMWait(t *testing.T) {
	previousWait := waitReplayDelay
	t.Cleanup(func() {
		waitReplayDelay = previousWait
	})

	var waits []time.Duration
	waitReplayDelay = func(ctx context.Context, d time.Duration) error {
		waits = append(waits, d)
		return nil
	}

	eventCh := make(chan model.Event, 3)
	t0 := time.Date(2026, time.March, 27, 12, 0, 0, 0, time.UTC)
	app := &Application{
		EventCh:    eventCh,
		replayOnly: true,
		replayTimeline: []session.ReplayFrame{
			{
				Timestamp: t0,
				Event:     model.Event{Type: model.UserInput, Message: "hello"},
			},
			{
				Timestamp: t0,
				Event:     model.Event{Type: model.AgentThinking},
			},
			{
				Timestamp: t0.Add(200 * time.Millisecond),
				Event:     model.Event{Type: model.AgentReply, Message: "hi"},
			},
		},
	}

	app.replayHistory()

	first := <-eventCh
	if first.Type != model.UserInput {
		t.Fatalf("first event type = %q, want %q", first.Type, model.UserInput)
	}
	second := <-eventCh
	if second.Type != model.AgentThinking {
		t.Fatalf("second event type = %q, want %q", second.Type, model.AgentThinking)
	}
	third := <-eventCh
	if third.Type != model.AgentReply {
		t.Fatalf("third event type = %q, want %q", third.Type, model.AgentReply)
	}
	if len(waits) != 1 {
		t.Fatalf("wait count = %d, want 1", len(waits))
	}
	if waits[0] != 200*time.Millisecond {
		t.Fatalf("wait duration = %v, want %v", waits[0], 200*time.Millisecond)
	}
}

func TestReplayHistoryTimelineSkipsDelayBetweenAssistantAndNextUser(t *testing.T) {
	previousWait := waitReplayDelay
	t.Cleanup(func() {
		waitReplayDelay = previousWait
	})

	var waits []time.Duration
	waitReplayDelay = func(ctx context.Context, d time.Duration) error {
		waits = append(waits, d)
		return nil
	}

	eventCh := make(chan model.Event, 4)
	t0 := time.Date(2026, time.March, 27, 12, 0, 0, 0, time.UTC)
	app := &Application{
		EventCh:    eventCh,
		replayOnly: true,
		replayTimeline: []session.ReplayFrame{
			{
				Timestamp: t0,
				Event:     model.Event{Type: model.AgentReplyDelta, Message: "he"},
			},
			{
				Timestamp: t0.Add(100 * time.Millisecond),
				Event:     model.Event{Type: model.AgentReply, Message: "hello"},
			},
			{
				Timestamp: t0.Add(5 * time.Second),
				Event:     model.Event{Type: model.UserInput, Message: "next"},
			},
		},
	}

	app.replayHistory()

	first := <-eventCh
	if first.Type != model.AgentReplyDelta {
		t.Fatalf("first event type = %q, want %q", first.Type, model.AgentReplyDelta)
	}
	second := <-eventCh
	if second.Type != model.AgentReply {
		t.Fatalf("second event type = %q, want %q", second.Type, model.AgentReply)
	}
	third := <-eventCh
	if third.Type != model.UserInput {
		t.Fatalf("third event type = %q, want %q", third.Type, model.UserInput)
	}
	if len(waits) != 1 {
		t.Fatalf("wait count = %d, want 1", len(waits))
	}
	if waits[0] != 100*time.Millisecond {
		t.Fatalf("wait duration = %v, want %v", waits[0], 100*time.Millisecond)
	}
}

func TestParseBootstrapConfigReplay(t *testing.T) {
	cfg, err := parseBootstrapConfig([]string{"replay", "sess_123"})
	if err != nil {
		t.Fatalf("parse replay config: %v", err)
	}
	if !cfg.Replay {
		t.Fatal("expected replay mode")
	}
	if cfg.ReplaySessionID != "sess_123" {
		t.Fatalf("replay session id = %q, want %q", cfg.ReplaySessionID, "sess_123")
	}
	if cfg.ReplaySpeed != 1 {
		t.Fatalf("replay speed = %v, want 1", cfg.ReplaySpeed)
	}
}

func TestParseBootstrapConfigReplayWithSpeed(t *testing.T) {
	cfg, err := parseBootstrapConfig([]string{"replay", "trajectory.json", "2x"})
	if err != nil {
		t.Fatalf("parse replay config with speed: %v", err)
	}
	if cfg.ReplaySessionID != "trajectory.json" {
		t.Fatalf("replay target = %q, want %q", cfg.ReplaySessionID, "trajectory.json")
	}
	if cfg.ReplaySpeed != 2 {
		t.Fatalf("replay speed = %v, want %v", cfg.ReplaySpeed, 2.0)
	}
}

func TestParseBootstrapConfigReplaySpeedOnly(t *testing.T) {
	cfg, err := parseBootstrapConfig([]string{"replay", "1.5"})
	if err != nil {
		t.Fatalf("parse replay speed only: %v", err)
	}
	if cfg.ReplaySessionID != "" {
		t.Fatalf("replay target = %q, want empty", cfg.ReplaySessionID)
	}
	if cfg.ReplaySpeed != 1.5 {
		t.Fatalf("replay speed = %v, want %v", cfg.ReplaySpeed, 1.5)
	}
}

func TestParseBootstrapConfigDebug(t *testing.T) {
	cfg, err := parseBootstrapConfig([]string{"--debug"})
	if err != nil {
		t.Fatalf("parse debug config: %v", err)
	}
	if !cfg.Debug {
		t.Fatal("expected debug mode to be enabled")
	}
}

func TestParseBootstrapConfigResumeDebug(t *testing.T) {
	cfg, err := parseBootstrapConfig([]string{"resume", "--debug", "sess_123"})
	if err != nil {
		t.Fatalf("parse resume debug config: %v", err)
	}
	if !cfg.Resume {
		t.Fatal("expected resume mode")
	}
	if !cfg.Debug {
		t.Fatal("expected debug mode to be enabled")
	}
	if cfg.ResumeSessionID != "sess_123" {
		t.Fatalf("resume session id = %q, want %q", cfg.ResumeSessionID, "sess_123")
	}
}

func TestLooksLikeTrajectoryPath(t *testing.T) {
	tests := []struct {
		target string
		want   bool
	}{
		{target: "sess_123", want: false},
		{target: "trajectory.json", want: true},
		{target: "trajectory.jsonl", want: true},
		{target: "./trajectory.json", want: true},
		{target: "logs/trajectory.json", want: true},
	}

	for _, tt := range tests {
		if got := looksLikeTrajectoryPath(tt.target); got != tt.want {
			t.Fatalf("looksLikeTrajectoryPath(%q) = %v, want %v", tt.target, got, tt.want)
		}
	}
}

func TestScaledReplayDelayUsesSpeedMultiplier(t *testing.T) {
	app := &Application{replaySpeed: 2}
	got := app.scaledReplayDelay(300 * time.Millisecond)
	if got != 150*time.Millisecond {
		t.Fatalf("scaled replay delay = %v, want %v", got, 150*time.Millisecond)
	}
}

func TestShouldSkipReplayDelay(t *testing.T) {
	if !shouldSkipReplayDelay(model.AgentReply, model.UserInput) {
		t.Fatal("expected agent reply to next user input to skip delay")
	}
	if !shouldSkipReplayDelay(model.AgentReplyDelta, model.UserInput) {
		t.Fatal("expected agent reply delta to next user input to skip delay")
	}
	if shouldSkipReplayDelay(model.ToolReplay, model.UserInput) {
		t.Fatal("did not expect tool replay to next user input to skip delay")
	}
}

func TestProcessInputHistoryReplayReadyStartsDeferredReplayOnce(t *testing.T) {
	eventCh := make(chan model.Event, 2)
	app := &Application{
		EventCh:            eventCh,
		replayBacklog:      []model.Event{{Type: model.UserInput, Message: "hello"}},
		deferHistoryReplay: true,
	}

	app.processInput(historyReplayReadyToken)

	select {
	case ev := <-eventCh:
		if ev.Type != model.UserInput || ev.Message != "hello" {
			t.Fatalf("replayed event = %#v, want user input hello", ev)
		}
	case <-time.After(200 * time.Millisecond):
		t.Fatal("expected deferred replay to start")
	}

	app.processInput(historyReplayReadyToken)

	select {
	case ev := <-eventCh:
		t.Fatalf("unexpected second replay event: %#v", ev)
	case <-time.After(50 * time.Millisecond):
	}
}

func TestProcessInputHistoryReplayReadyIgnoredWhenReplayNotDeferred(t *testing.T) {
	eventCh := make(chan model.Event, 1)
	app := &Application{
		EventCh:       eventCh,
		replayBacklog: []model.Event{{Type: model.UserInput, Message: "hello"}},
	}

	app.processInput(historyReplayReadyToken)

	select {
	case ev := <-eventCh:
		t.Fatalf("unexpected replay event without deferred replay: %#v", ev)
	case <-time.After(50 * time.Millisecond):
	}
}
