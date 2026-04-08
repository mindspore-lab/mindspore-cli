package session

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

const (
	trajectoryFilename   = "trajectory.jsonl"
	snapshotFilename     = "snapshot.json"
	recordTypeMeta       = "session_meta"
	recordTypeUser       = "user"
	recordTypeAssistant  = "assistant"
	recordTypeToolCall   = "tool_call"
	recordTypeToolResult = "tool_result"
	recordTypeSkill      = "skill_activation"
	formatVersion        = 1
	defaultSessionSubdir = ".mscli/sessions"
	replayWaitCap        = 5 * time.Second
)

// Meta is the first JSONL record describing the session.
type Meta struct {
	Type         string    `json:"type"`
	Version      int       `json:"version"`
	SessionID    string    `json:"session_id"`
	WorkDir      string    `json:"workdir"`
	WorkDirKey   string    `json:"workdir_key"`
	SystemPrompt string    `json:"system_prompt"`
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"updated_at"`
}

// MessageRecord is one persisted conversation event.
type MessageRecord struct {
	Type       string          `json:"type"`
	Timestamp  time.Time       `json:"timestamp"`
	Content    string          `json:"content,omitempty"`
	ToolName   string          `json:"tool_name,omitempty"`
	Arguments  json.RawMessage `json:"arguments,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	SkillName  string          `json:"skill_name,omitempty"`
}

// Snapshot stores the latest restorable context state.
type Snapshot struct {
	SessionID     string         `json:"session_id"`
	WorkDir       string         `json:"workdir"`
	SystemPrompt  string         `json:"system_prompt"`
	UpdatedAt     time.Time      `json:"updated_at"`
	Messages      []llm.Message  `json:"messages,omitempty"`
	ProviderUsage *UsageSnapshot `json:"provider_usage,omitempty"`
}

// UsageSnapshot stores the latest provider-backed token snapshot for resume.
type UsageSnapshot struct {
	Provider   string `json:"provider,omitempty"`
	TokenScope string `json:"token_scope,omitempty"`
	Tokens     int    `json:"tokens,omitempty"`
	LocalDelta int    `json:"local_delta,omitempty"`
}

// ReplayFrame is one UI replay event paired with its original timestamp.
type ReplayFrame struct {
	Timestamp time.Time
	Event     model.Event
}

// Summary is a lightweight description of one persisted session for UI pickers.
type Summary struct {
	SessionID      string
	CreatedAt      time.Time
	UpdatedAt      time.Time
	FirstUserInput string
	HasDialogue    bool
}

// Session owns trajectory persistence for one workspace conversation.
type Session struct {
	mu           sync.RWMutex
	meta         Meta
	records      []MessageRecord
	snapshot     Snapshot
	path         string
	snapshotPath string
	persisted    bool
	file         *os.File
	enc          *json.Encoder
}

// Create allocates a new session under ~/.mscli/sessions.
// Persistence begins only after Activate is called.
func Create(workDir, systemPrompt string) (*Session, error) {
	absWorkDir, err := normalizeWorkDir(workDir)
	if err != nil {
		return nil, err
	}

	key := workDirKey(absWorkDir)
	now := time.Now()
	id, path, err := nextSessionLocation(key, now)
	if err != nil {
		return nil, err
	}

	s := &Session{
		meta: Meta{
			Type:         recordTypeMeta,
			Version:      formatVersion,
			SessionID:    id,
			WorkDir:      absWorkDir,
			WorkDirKey:   key,
			SystemPrompt: systemPrompt,
			CreatedAt:    now,
			UpdatedAt:    now,
		},
		records: make([]MessageRecord, 0),
		snapshot: Snapshot{
			SessionID:    id,
			WorkDir:      absWorkDir,
			SystemPrompt: systemPrompt,
			UpdatedAt:    now,
		},
		path:         path,
		snapshotPath: snapshotPath(path),
	}
	return s, nil
}

// LoadLatest loads the latest session for the workdir, ordered by trajectory mtime.
func LoadLatest(workDir string) (*Session, error) {
	absWorkDir, err := normalizeWorkDir(workDir)
	if err != nil {
		return nil, err
	}

	key := workDirKey(absWorkDir)
	bucket := sessionBucketDir(key)
	entries, err := os.ReadDir(bucket)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("no resumable session found for workdir: %s", absWorkDir)
		}
		return nil, fmt.Errorf("read session bucket: %w", err)
	}

	type candidate struct {
		id      string
		modTime time.Time
	}

	candidates := make([]candidate, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		path := filepath.Join(bucket, entry.Name(), trajectoryFilename)
		info, err := os.Stat(path)
		if err != nil || info.IsDir() {
			continue
		}
		candidates = append(candidates, candidate{id: entry.Name(), modTime: info.ModTime()})
	}

	if len(candidates) == 0 {
		return nil, fmt.Errorf("no resumable session found for workdir: %s", absWorkDir)
	}

	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].modTime.Equal(candidates[j].modTime) {
			return candidates[i].id > candidates[j].id
		}
		return candidates[i].modTime.After(candidates[j].modTime)
	})

	return LoadByID(absWorkDir, candidates[0].id)
}

// ListForWorkDir returns persisted session summaries for the given workdir.
func ListForWorkDir(workDir string) ([]Summary, error) {
	absWorkDir, err := normalizeWorkDir(workDir)
	if err != nil {
		return nil, err
	}

	key := workDirKey(absWorkDir)
	bucket := sessionBucketDir(key)
	entries, err := os.ReadDir(bucket)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read session bucket: %w", err)
	}

	summaries := make([]Summary, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		path := filepath.Join(bucket, entry.Name(), trajectoryFilename)
		loaded, err := loadFromPath(path, false)
		if err != nil {
			continue
		}

		summary := loaded.summary()
		if !summary.HasDialogue {
			continue
		}
		summaries = append(summaries, summary)
	}

	sort.Slice(summaries, func(i, j int) bool {
		if summaries[i].UpdatedAt.Equal(summaries[j].UpdatedAt) {
			return summaries[i].SessionID > summaries[j].SessionID
		}
		return summaries[i].UpdatedAt.After(summaries[j].UpdatedAt)
	})

	return summaries, nil
}

// CleanupExpired removes persisted sessions whose latest on-disk update is older than maxAge.
func CleanupExpired(maxAge time.Duration) (int, error) {
	if maxAge <= 0 {
		return 0, fmt.Errorf("max age must be positive")
	}

	root, err := sessionRootDir()
	if err != nil {
		return 0, err
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, fmt.Errorf("read session root: %w", err)
	}

	cutoff := time.Now().Add(-maxAge)
	removed := 0
	for _, bucket := range entries {
		if !bucket.IsDir() {
			continue
		}
		bucketPath := filepath.Join(root, bucket.Name())
		bucketRemoved, err := cleanupExpiredBucket(bucketPath, cutoff)
		removed += bucketRemoved
		if err != nil {
			return removed, err
		}
		_ = os.Remove(bucketPath)
	}

	return removed, nil
}

// LoadByID loads a specific session for the given workdir.
func LoadByID(workDir, sessionID string) (*Session, error) {
	absWorkDir, err := normalizeWorkDir(workDir)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(sessionID) == "" {
		return nil, fmt.Errorf("session id cannot be empty")
	}

	key := workDirKey(absWorkDir)
	path := trajectoryPath(key, sessionID)
	return loadFromPath(path, true)
}

// LoadReplayPath loads a trajectory file directly for read-only replay.
func LoadReplayPath(path string) (*Session, error) {
	if strings.TrimSpace(path) == "" {
		return nil, fmt.Errorf("trajectory path cannot be empty")
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("resolve trajectory path: %w", err)
	}
	return loadFromPath(absPath, false)
}

// AppendUserInput appends one user input record and syncs it immediately.
func (s *Session) AppendUserInput(content string) error {
	if s == nil {
		return fmt.Errorf("session is nil")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	record := MessageRecord{
		Type:      recordTypeUser,
		Timestamp: time.Now(),
		Content:   content,
	}
	s.records = append(s.records, record)
	s.meta.UpdatedAt = record.Timestamp
	if !s.persisted {
		return nil
	}
	return s.writeRecordLocked(record)
}

// AppendAssistant appends one assistant reply line and syncs it immediately.
func (s *Session) AppendAssistant(content string) error {
	if s == nil {
		return fmt.Errorf("session is nil")
	}
	if strings.TrimSpace(content) == "" {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	record := MessageRecord{
		Type:      recordTypeAssistant,
		Timestamp: time.Now(),
		Content:   content,
	}
	s.records = append(s.records, record)
	s.meta.UpdatedAt = record.Timestamp
	if !s.persisted {
		return nil
	}
	return s.writeRecordLocked(record)
}

// AppendToolCall appends one tool call record and syncs it immediately.
func (s *Session) AppendToolCall(tc llm.ToolCall) error {
	if s == nil {
		return fmt.Errorf("session is nil")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	record := MessageRecord{
		Type:       recordTypeToolCall,
		Timestamp:  time.Now(),
		ToolName:   tc.Function.Name,
		Arguments:  append([]byte(nil), tc.Function.Arguments...),
		ToolCallID: tc.ID,
	}
	s.records = append(s.records, record)
	s.meta.UpdatedAt = record.Timestamp
	if !s.persisted {
		return nil
	}
	return s.writeRecordLocked(record)
}

// AppendToolResult appends one tool result record and syncs it immediately.
func (s *Session) AppendToolResult(toolCallID, toolName, content string) error {
	if s == nil {
		return fmt.Errorf("session is nil")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	record := MessageRecord{
		Type:       recordTypeToolResult,
		Timestamp:  time.Now(),
		Content:    content,
		ToolName:   toolName,
		ToolCallID: toolCallID,
	}
	s.records = append(s.records, record)
	s.meta.UpdatedAt = record.Timestamp
	if !s.persisted {
		return nil
	}
	return s.writeRecordLocked(record)
}

// AppendSkillActivation appends one skill activation record and syncs it immediately.
func (s *Session) AppendSkillActivation(skillName string) error {
	if s == nil {
		return fmt.Errorf("session is nil")
	}
	if strings.TrimSpace(skillName) == "" {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	record := MessageRecord{
		Type:      recordTypeSkill,
		Timestamp: time.Now(),
		SkillName: strings.TrimSpace(skillName),
	}
	s.records = append(s.records, record)
	s.meta.UpdatedAt = record.Timestamp
	if !s.persisted {
		return nil
	}
	return s.writeRecordLocked(record)
}

// ReplayEvents synthesizes UI replay events from persisted conversation records.
func (s *Session) ReplayEvents() []model.Event {
	frames := s.ReplayTimeline()
	if len(frames) == 0 {
		return nil
	}

	events := make([]model.Event, 0, len(frames))
	for _, frame := range frames {
		events = append(events, frame.Event)
	}
	return events
}

// ReplayTimeline synthesizes timestamped UI replay events from persisted conversation records.
func (s *Session) ReplayTimeline() []ReplayFrame {
	if s == nil {
		return nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	frames := make([]ReplayFrame, 0, len(s.records))
	pendingLoadSkillCallIDs := make([]string, 0)
	for _, record := range s.records {
		ev, ok := replayEvent(record)
		if record.Type == recordTypeToolCall && strings.TrimSpace(record.ToolName) == "load_skill" {
			pendingLoadSkillCallIDs = append(pendingLoadSkillCallIDs, strings.TrimSpace(record.ToolCallID))
		}
		if record.Type == recordTypeSkill {
			toolCallID := strings.TrimSpace(record.ToolCallID)
			if toolCallID == "" && len(pendingLoadSkillCallIDs) > 0 {
				toolCallID = pendingLoadSkillCallIDs[0]
				pendingLoadSkillCallIDs = pendingLoadSkillCallIDs[1:]
			}
			ev = replaySkillEvent(record.SkillName, toolCallID)
			ok = true
		}
		if !ok {
			continue
		}
		frames = append(frames, ReplayFrame{
			Timestamp: record.Timestamp,
			Event:     ev,
		})
	}
	return frames
}

// PlaybackTimeline synthesizes timestamped UI replay events for real-time playback.
// It inserts AgentThinking between turns when the next visible event comes from LLM reasoning.
func (s *Session) PlaybackTimeline() []ReplayFrame {
	frames := s.ReplayTimeline()
	if len(frames) < 2 {
		return frames
	}

	playback := make([]ReplayFrame, 0, len(frames)*2)
	for i, frame := range frames {
		var previous *ReplayFrame
		if i > 0 {
			prev := frames[i-1]
			previous = &prev
			if shouldInsertThinking(prev, frame) {
				playback = append(playback, ReplayFrame{
					Timestamp: prev.Timestamp,
					Event: model.Event{
						Type: model.AgentThinking,
					},
				})
			}
		}
		playback = append(playback, expandAssistantReplay(previous, frame)...)
	}
	playback = compressReplaySegments(playback, replayAssistantCompressionSegments(playback))
	return compressReplaySegments(playback, replayToolCompressionSegments(playback))
}

type replayCompressionSegment struct {
	start   int
	end     int
	waitEnd int
}

type toolReplayInterval struct {
	start int
	end   int
}

func compressReplaySegments(frames []ReplayFrame, segments []replayCompressionSegment) []ReplayFrame {
	if len(frames) == 0 || len(segments) == 0 {
		return frames
	}

	original := make([]ReplayFrame, len(frames))
	copy(original, frames)

	var saved time.Duration
	segmentIdx := 0
	for i := 0; i < len(frames); i++ {
		frames[i].Timestamp = original[i].Timestamp.Add(-saved)

		if segmentIdx >= len(segments) {
			continue
		}
		segment := segments[segmentIdx]
		if i != segment.start {
			continue
		}
		segmentIdx++

		originalDuration := original[segment.end].Timestamp.Sub(original[segment.start].Timestamp)
		if originalDuration <= replayWaitCap {
			continue
		}

		start := original[segment.start].Timestamp.Add(-saved)
		for j := segment.start; j <= segment.end; j++ {
			offset := original[j].Timestamp.Sub(original[segment.start].Timestamp)
			frames[j].Timestamp = start.Add(scaleReplayDuration(offset, originalDuration, replayWaitCap))
		}

		waitOriginal := original[segment.waitEnd].Timestamp.Sub(original[segment.start].Timestamp)
		waitSimulated := frames[segment.waitEnd].Timestamp.Sub(frames[segment.start].Timestamp)
		if waitOriginal > waitSimulated && waitSimulated > 0 {
			frames[segment.start].Event.ReplayWait = &model.ReplayWaitData{
				OriginalDuration:  waitOriginal,
				SimulatedDuration: waitSimulated,
			}
		}

		saved += originalDuration - replayWaitCap
		i = segment.end
	}

	return frames
}

func replayAssistantCompressionSegments(frames []ReplayFrame) []replayCompressionSegment {
	segments := make([]replayCompressionSegment, 0)
	for i := 0; i < len(frames); i++ {
		if frames[i].Event.Type != model.AgentThinking || i+1 >= len(frames) {
			continue
		}
		switch frames[i+1].Event.Type {
		case model.ToolCallStart, model.AgentReply:
			segments = append(segments, replayCompressionSegment{start: i, end: i + 1, waitEnd: i + 1})
		case model.AgentReplyDelta:
			end, ok := assistantReplayEnd(frames, i+1)
			if !ok {
				continue
			}
			segments = append(segments, replayCompressionSegment{start: i, end: end, waitEnd: i + 1})
		}
	}
	return segments
}

func replayToolCompressionSegments(frames []ReplayFrame) []replayCompressionSegment {
	intervals := replayToolIntervals(frames)
	if len(intervals) == 0 {
		return nil
	}

	segments := make([]replayCompressionSegment, 0, len(intervals))
	current := replayCompressionSegment{
		start:   intervals[0].start,
		end:     intervals[0].end,
		waitEnd: intervals[0].end,
	}
	for _, interval := range intervals[1:] {
		if !frames[interval.start].Timestamp.After(frames[current.end].Timestamp) {
			if frames[interval.end].Timestamp.After(frames[current.end].Timestamp) {
				current.end = interval.end
				current.waitEnd = interval.end
			}
			continue
		}
		segments = append(segments, current)
		current = replayCompressionSegment{
			start:   interval.start,
			end:     interval.end,
			waitEnd: interval.end,
		}
	}
	segments = append(segments, current)
	return segments
}

func replayToolIntervals(frames []ReplayFrame) []toolReplayInterval {
	var intervals []toolReplayInterval
	pendingByID := make(map[string]int)
	pendingByTool := make(map[string][]int)
	var pendingLoadSkill []int

	for i, frame := range frames {
		switch frame.Event.Type {
		case model.ToolCallStart:
			toolName := strings.TrimSpace(frame.Event.ToolName)
			if toolName == "load_skill" {
				pendingLoadSkill = append(pendingLoadSkill, i)
				continue
			}
			if toolCallID := strings.TrimSpace(frame.Event.ToolCallID); toolCallID != "" {
				pendingByID[toolCallID] = i
				continue
			}
			pendingByTool[toolName] = append(pendingByTool[toolName], i)
		case model.ToolReplay:
			if start, ok := replayToolIntervalStart(frame.Event, pendingByID, pendingByTool); ok {
				intervals = append(intervals, toolReplayInterval{start: start, end: i})
			}
		case model.ToolSkill:
			if len(pendingLoadSkill) == 0 {
				continue
			}
			intervals = append(intervals, toolReplayInterval{start: pendingLoadSkill[0], end: i})
			pendingLoadSkill = pendingLoadSkill[1:]
		}
	}

	return intervals
}

func replayToolIntervalStart(ev model.Event, pendingByID map[string]int, pendingByTool map[string][]int) (int, bool) {
	if toolCallID := strings.TrimSpace(ev.ToolCallID); toolCallID != "" {
		start, ok := pendingByID[toolCallID]
		if !ok {
			return 0, false
		}
		delete(pendingByID, toolCallID)
		return start, true
	}

	toolName := strings.TrimSpace(ev.ToolName)
	queue := pendingByTool[toolName]
	if len(queue) == 0 {
		return 0, false
	}
	start := queue[0]
	pendingByTool[toolName] = queue[1:]
	return start, true
}

func assistantReplayEnd(frames []ReplayFrame, start int) (int, bool) {
	for i := start; i < len(frames); i++ {
		switch frames[i].Event.Type {
		case model.AgentReplyDelta:
			continue
		case model.AgentReply:
			return i, true
		default:
			return 0, false
		}
	}
	return 0, false
}

func scaleReplayDuration(offset, originalTotal, simulatedTotal time.Duration) time.Duration {
	if offset <= 0 || originalTotal <= 0 || simulatedTotal <= 0 {
		return 0
	}
	if offset >= originalTotal {
		return simulatedTotal
	}
	scaled := float64(offset) * float64(simulatedTotal) / float64(originalTotal)
	if scaled < 1 {
		return time.Nanosecond
	}
	return time.Duration(scaled)
}

// RestoreContext returns the system prompt and reconstructed non-system messages.
func (s *Session) RestoreContext() (string, []llm.Message) {
	if s == nil {
		return "", nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	messages := make([]llm.Message, len(s.snapshot.Messages))
	copy(messages, s.snapshot.Messages)
	return s.snapshot.SystemPrompt, messages
}

// UsageSnapshot returns a copy of the persisted provider-backed usage snapshot.
func (s *Session) UsageSnapshot() *UsageSnapshot {
	if s == nil {
		return nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()
	return cloneUsageSnapshot(s.snapshot.ProviderUsage)
}

// Path returns the trajectory file path.
func (s *Session) Path() string {
	if s == nil {
		return ""
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.path
}

// ID returns the session ID.
func (s *Session) ID() string {
	if s == nil {
		return ""
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.meta.SessionID
}

// Activate materializes session files and flushes buffered state to disk.
func (s *Session) Activate() error {
	if s == nil {
		return fmt.Errorf("session is nil")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.persisted {
		return nil
	}

	file, enc, err := openAppender(s.path, false)
	if err != nil {
		return err
	}

	s.file = file
	s.enc = enc

	if err := s.writeRecordLocked(s.meta); err != nil {
		s.cleanupActivationFailureLocked()
		return err
	}
	for _, record := range s.records {
		if err := s.writeRecordLocked(record); err != nil {
			s.cleanupActivationFailureLocked()
			return err
		}
	}
	if err := s.writeSnapshotLocked(); err != nil {
		s.cleanupActivationFailureLocked()
		return err
	}
	s.persisted = true
	return nil
}

// HasPersistedDialogue reports whether this session has persisted user/assistant dialogue.
func (s *Session) HasPersistedDialogue() bool {
	if s == nil {
		return false
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	if !s.persisted {
		return false
	}
	for _, record := range s.records {
		if record.Type == recordTypeUser || record.Type == recordTypeAssistant {
			return true
		}
	}
	return false
}

// Meta returns a copy of the persisted meta record.
func (s *Session) Meta() Meta {
	if s == nil {
		return Meta{}
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.meta
}

func (s *Session) summary() Summary {
	if s == nil {
		return Summary{}
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	updatedAt := s.meta.UpdatedAt
	if s.snapshot.UpdatedAt.After(updatedAt) {
		updatedAt = s.snapshot.UpdatedAt
	}
	if info, err := os.Stat(s.path); err == nil && info.ModTime().After(updatedAt) {
		updatedAt = info.ModTime()
	}
	if info, err := os.Stat(s.snapshotPath); err == nil && info.ModTime().After(updatedAt) {
		updatedAt = info.ModTime()
	}

	firstUserInput := ""
	hasDialogue := false
	for _, record := range s.records {
		switch record.Type {
		case recordTypeUser:
			hasDialogue = true
			if firstUserInput == "" {
				firstUserInput = sessionPreview(record.Content)
			}
		case recordTypeAssistant:
			hasDialogue = true
		}
	}
	if firstUserInput == "" {
		firstUserInput = "(no user input recorded)"
	}

	return Summary{
		SessionID:      s.meta.SessionID,
		CreatedAt:      s.meta.CreatedAt,
		UpdatedAt:      updatedAt,
		FirstUserInput: firstUserInput,
		HasDialogue:    hasDialogue,
	}
}

func sessionPreview(content string) string {
	lines := strings.Split(strings.ReplaceAll(content, "\r\n", "\n"), "\n")
	for _, line := range lines {
		line = strings.Join(strings.Fields(strings.TrimSpace(line)), " ")
		if line == "" {
			continue
		}
		const maxPreviewRunes = 96
		runes := []rune(line)
		if len(runes) <= maxPreviewRunes {
			return line
		}
		return string(runes[:maxPreviewRunes-3]) + "..."
	}
	return ""
}

// Close closes the underlying file.
func (s *Session) Close() error {
	if s == nil {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.file == nil {
		return nil
	}
	if err := s.file.Close(); err != nil {
		return fmt.Errorf("close trajectory: %w", err)
	}
	s.file = nil
	s.enc = nil
	return nil
}

func loadFromPath(path string, appendOnly bool) (*Session, error) {
	file, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("session not found: %s", path)
		}
		return nil, fmt.Errorf("open trajectory: %w", err)
	}

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)

	var (
		meta    Meta
		records []MessageRecord
		line    int
	)
	for scanner.Scan() {
		line++
		data := scanner.Bytes()
		var envelope struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(data, &envelope); err != nil {
			_ = file.Close()
			return nil, fmt.Errorf("decode trajectory line %d: %w", line, err)
		}

		switch envelope.Type {
		case recordTypeMeta:
			if line != 1 {
				_ = file.Close()
				return nil, fmt.Errorf("invalid trajectory: meta record must be first")
			}
			if err := json.Unmarshal(data, &meta); err != nil {
				_ = file.Close()
				return nil, fmt.Errorf("decode session meta: %w", err)
			}
		case recordTypeUser, recordTypeAssistant, recordTypeToolCall, recordTypeToolResult, recordTypeSkill:
			var record MessageRecord
			if err := json.Unmarshal(data, &record); err != nil {
				_ = file.Close()
				return nil, fmt.Errorf("decode message record: %w", err)
			}
			records = append(records, record)
		default:
			_ = file.Close()
			return nil, fmt.Errorf("unknown trajectory record type %q on line %d", envelope.Type, line)
		}
	}
	if err := scanner.Err(); err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("scan trajectory: %w", err)
	}
	if err := file.Close(); err != nil {
		return nil, fmt.Errorf("close trajectory: %w", err)
	}
	if line == 0 {
		return nil, fmt.Errorf("invalid trajectory: empty file")
	}
	if meta.Type != recordTypeMeta {
		return nil, fmt.Errorf("invalid trajectory: missing meta record")
	}

	snapPath := snapshotPath(path)
	snapshot, err := loadSnapshot(snapPath)
	if err != nil {
		return nil, err
	}
	if snapshot.SessionID == "" {
		snapshot = Snapshot{
			SessionID:    meta.SessionID,
			WorkDir:      meta.WorkDir,
			SystemPrompt: meta.SystemPrompt,
			UpdatedAt:    meta.UpdatedAt,
		}
	}

	sessionState := &Session{
		meta:         meta,
		records:      records,
		snapshot:     snapshot,
		path:         path,
		snapshotPath: snapPath,
	}
	if !appendOnly {
		return sessionState, nil
	}

	appender, enc, err := openAppender(path, true)
	if err != nil {
		return nil, err
	}
	sessionState.persisted = true
	sessionState.file = appender
	sessionState.enc = enc
	return sessionState, nil
}

func openAppender(path string, appendOnly bool) (*os.File, *json.Encoder, error) {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, nil, fmt.Errorf("create session directory: %w", err)
	}

	flags := os.O_CREATE | os.O_WRONLY
	if appendOnly {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}

	file, err := os.OpenFile(path, flags, 0600)
	if err != nil {
		return nil, nil, fmt.Errorf("open trajectory: %w", err)
	}

	return file, json.NewEncoder(file), nil
}

func (s *Session) writeRecordLocked(record any) error {
	if s.file == nil || s.enc == nil {
		return fmt.Errorf("session file is not open")
	}

	if err := s.enc.Encode(record); err != nil {
		return fmt.Errorf("encode trajectory record: %w", err)
	}
	if err := s.file.Sync(); err != nil {
		return fmt.Errorf("sync trajectory: %w", err)
	}
	return nil
}

// SaveSnapshot overwrites snapshot.json with the current restorable context.
func (s *Session) SaveSnapshot(systemPrompt string, messages []llm.Message) error {
	return s.SaveSnapshotWithUsage(systemPrompt, messages, nil)
}

// SaveSnapshotWithUsage overwrites snapshot.json with the current restorable context and usage snapshot.
func (s *Session) SaveSnapshotWithUsage(systemPrompt string, messages []llm.Message, usage *UsageSnapshot) error {
	if s == nil {
		return fmt.Errorf("session is nil")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.meta.SystemPrompt = systemPrompt
	s.snapshot.SystemPrompt = systemPrompt
	s.snapshot.UpdatedAt = time.Now()
	s.snapshot.Messages = make([]llm.Message, len(messages))
	copy(s.snapshot.Messages, messages)
	s.snapshot.ProviderUsage = cloneUsageSnapshot(usage)
	if !s.persisted {
		return nil
	}
	return s.writeSnapshotLocked()
}

func cloneUsageSnapshot(usage *UsageSnapshot) *UsageSnapshot {
	if usage == nil {
		return nil
	}
	copy := *usage
	return &copy
}

func (s *Session) cleanupActivationFailureLocked() {
	if s.file != nil {
		_ = s.file.Close()
	}
	s.file = nil
	s.enc = nil
	_ = os.Remove(s.path)
	_ = os.Remove(s.snapshotPath)
}

func (s *Session) writeSnapshotLocked() error {
	data, err := json.MarshalIndent(s.snapshot, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal snapshot: %w", err)
	}
	if err := os.WriteFile(s.snapshotPath, data, 0600); err != nil {
		return fmt.Errorf("write snapshot: %w", err)
	}
	return nil
}

func loadSnapshot(path string) (Snapshot, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return Snapshot{}, nil
		}
		return Snapshot{}, fmt.Errorf("read snapshot: %w", err)
	}

	var snapshot Snapshot
	if err := json.Unmarshal(data, &snapshot); err != nil {
		return Snapshot{}, fmt.Errorf("decode snapshot: %w", err)
	}
	return snapshot, nil
}

func normalizeWorkDir(workDir string) (string, error) {
	if strings.TrimSpace(workDir) == "" {
		return "", fmt.Errorf("workdir cannot be empty")
	}

	absWorkDir, err := filepath.Abs(workDir)
	if err != nil {
		return "", fmt.Errorf("resolve workdir: %w", err)
	}
	return filepath.Clean(absWorkDir), nil
}

func cleanupExpiredBucket(bucketPath string, cutoff time.Time) (int, error) {
	entries, err := os.ReadDir(bucketPath)
	if err != nil {
		return 0, fmt.Errorf("read session bucket: %w", err)
	}

	removed := 0
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		sessionDir := filepath.Join(bucketPath, entry.Name())
		updatedAt, err := sessionDirUpdatedAt(sessionDir)
		if err != nil {
			return removed, err
		}
		if !updatedAt.Before(cutoff) {
			continue
		}
		if err := os.RemoveAll(sessionDir); err != nil {
			return removed, fmt.Errorf("remove expired session %s: %w", entry.Name(), err)
		}
		removed++
	}

	return removed, nil
}

func sessionDirUpdatedAt(sessionDir string) (time.Time, error) {
	info, err := os.Stat(sessionDir)
	if err != nil {
		return time.Time{}, fmt.Errorf("stat session dir: %w", err)
	}

	updatedAt := info.ModTime()
	entries, err := os.ReadDir(sessionDir)
	if err != nil {
		return time.Time{}, fmt.Errorf("read session dir: %w", err)
	}
	for _, entry := range entries {
		info, err := entry.Info()
		if err != nil {
			return time.Time{}, fmt.Errorf("stat session entry: %w", err)
		}
		if info.ModTime().After(updatedAt) {
			updatedAt = info.ModTime()
		}
	}
	return updatedAt, nil
}

func workDirKey(absWorkDir string) string {
	key := filepath.Clean(absWorkDir)
	replacer := strings.NewReplacer(
		"/", "-",
		"\\", "-",
		":", "-",
		"*", "-",
		"?", "-",
		"\"", "-",
		"<", "-",
		">", "-",
		"|", "-",
	)
	key = replacer.Replace(key)
	key = strings.Map(func(r rune) rune {
		if unicode.IsControl(r) {
			return '-'
		}
		return r
	}, key)
	if strings.Trim(key, ".- ") == "" {
		return "workdir"
	}
	return key
}

func sessionBucketDir(key string) string {
	root, err := sessionRootDir()
	if err != nil {
		return filepath.Join(".", defaultSessionSubdir, key)
	}
	return filepath.Join(root, key)
}

func trajectoryPath(key, sessionID string) string {
	return filepath.Join(sessionBucketDir(key), sessionID, trajectoryFilename)
}

func snapshotPath(trajectoryPath string) string {
	return filepath.Join(filepath.Dir(trajectoryPath), snapshotFilename)
}

func nextSessionLocation(key string, now time.Time) (string, string, error) {
	for offset := 0; offset < 5; offset++ {
		id := "sess_" + now.Add(time.Duration(offset)*time.Second).Format("060102-150405")
		path := trajectoryPath(key, id)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			return id, path, nil
		} else if err != nil && !os.IsNotExist(err) {
			return "", "", fmt.Errorf("check session path: %w", err)
		}
	}
	return "", "", fmt.Errorf("failed to allocate unique session id")
}

func sessionRootDir() (string, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolve home dir: %w", err)
	}
	if strings.TrimSpace(homeDir) == "" {
		return "", fmt.Errorf("home dir cannot be empty")
	}
	return filepath.Join(homeDir, defaultSessionSubdir), nil
}

func describeToolCall(toolName string, raw json.RawMessage) string {
	var params map[string]any
	_ = json.Unmarshal(raw, &params)

	getString := func(keys ...string) string {
		for _, key := range keys {
			if v, ok := params[key].(string); ok {
				v = strings.TrimSpace(v)
				if v != "" {
					return v
				}
			}
		}
		return ""
	}

	switch toolName {
	case "shell":
		return getString("command")
	case "read", "edit", "write":
		return getString("path", "file_path")
	case "grep":
		pattern := getString("pattern")
		path := getString("path")
		switch {
		case pattern != "" && path != "":
			return fmt.Sprintf("%q in %s", pattern, path)
		case pattern != "":
			return pattern
		default:
			return path
		}
	case "glob":
		pattern := getString("pattern")
		path := getString("path")
		switch {
		case pattern != "" && path != "":
			return fmt.Sprintf("%s in %s", pattern, path)
		case pattern != "":
			return pattern
		default:
			return path
		}
	case "load_skill":
		return getString("name")
	default:
		preview := strings.TrimSpace(string(raw))
		if preview == "" {
			return toolName
		}
		return preview
	}
}

func skillSummary(skillName string) string {
	skillName = strings.TrimSpace(skillName)
	if skillName == "" {
		return ""
	}
	return fmt.Sprintf("loaded skill: %s", skillName)
}

func replayToolCallEvent(record MessageRecord) model.Event {
	return model.Event{
		Type:       model.ToolCallStart,
		ToolName:   record.ToolName,
		ToolCallID: record.ToolCallID,
		Message:    describeToolCall(record.ToolName, record.Arguments),
	}
}

func replayToolResultEvent(record MessageRecord) model.Event {
	return model.Event{
		Type:       model.ToolReplay,
		ToolName:   record.ToolName,
		ToolCallID: record.ToolCallID,
		Message:    record.Content,
	}
}

func replaySkillEvent(skillName, toolCallID string) model.Event {
	return model.Event{
		Type:       model.ToolSkill,
		ToolName:   "load_skill",
		ToolCallID: strings.TrimSpace(toolCallID),
		Message:    skillName,
		Summary:    skillSummary(skillName),
	}
}

func replayEvent(record MessageRecord) (model.Event, bool) {
	switch record.Type {
	case recordTypeUser:
		return model.Event{Type: model.UserInput, Message: record.Content}, true
	case recordTypeAssistant:
		return model.Event{Type: model.AgentReply, Message: record.Content}, true
	case recordTypeToolCall:
		return replayToolCallEvent(record), true
	case recordTypeToolResult:
		if record.ToolName == "load_skill" {
			return model.Event{}, false
		}
		return replayToolResultEvent(record), true
	case recordTypeSkill:
		return replaySkillEvent(record.SkillName, record.ToolCallID), true
	default:
		return model.Event{}, false
	}
}

func shouldInsertThinking(previous, next ReplayFrame) bool {
	if !next.Timestamp.After(previous.Timestamp) {
		return false
	}

	switch previous.Event.Type {
	case model.UserInput, model.ToolReplay, model.ToolSkill:
	default:
		return false
	}

	switch next.Event.Type {
	case model.AgentReply, model.ToolCallStart:
		return true
	default:
		return false
	}
}

func expandAssistantReplay(previous *ReplayFrame, current ReplayFrame) []ReplayFrame {
	if previous == nil || current.Event.Type != model.AgentReply || !current.Timestamp.After(previous.Timestamp) {
		return []ReplayFrame{current}
	}

	deltas := splitReplayAssistantDeltas(current.Event.Message, current.Timestamp.Sub(previous.Timestamp))
	if len(deltas) == 0 {
		return []ReplayFrame{current}
	}

	expanded := make([]ReplayFrame, 0, len(deltas)+1)
	totalDelay := current.Timestamp.Sub(previous.Timestamp)
	step := totalDelay / time.Duration(len(deltas)+1)
	if step <= 0 {
		return []ReplayFrame{current}
	}
	for i, delta := range deltas {
		expanded = append(expanded, ReplayFrame{
			Timestamp: previous.Timestamp.Add(step * time.Duration(i+1)),
			Event: model.Event{
				Type:    model.AgentReplyDelta,
				Message: delta,
			},
		})
	}
	expanded = append(expanded, current)
	return expanded
}

func splitReplayAssistantDeltas(message string, totalDelay time.Duration) []string {
	runes := []rune(message)
	if len(runes) == 0 || totalDelay <= 0 {
		return nil
	}

	chunkCount := int(totalDelay / (150 * time.Millisecond))
	if chunkCount < 1 {
		chunkCount = 1
	}

	maxByLength := (len(runes) + 7) / 8
	if maxByLength < 1 {
		maxByLength = 1
	}
	if chunkCount > maxByLength {
		chunkCount = maxByLength
	}

	if len(runes) > 1 && totalDelay > 200*time.Millisecond && chunkCount < 2 {
		chunkCount = 2
	}
	if chunkCount > 12 {
		chunkCount = 12
	}
	if chunkCount > len(runes) {
		chunkCount = len(runes)
	}

	return splitReplayRunesEvenly(runes, chunkCount)
}

func splitReplayRunesEvenly(runes []rune, chunkCount int) []string {
	if len(runes) == 0 || chunkCount <= 0 {
		return nil
	}
	if chunkCount > len(runes) {
		chunkCount = len(runes)
	}

	base := len(runes) / chunkCount
	extra := len(runes) % chunkCount
	chunks := make([]string, 0, chunkCount)
	start := 0
	for i := 0; i < chunkCount; i++ {
		size := base
		if i < extra {
			size++
		}
		end := start + size
		chunks = append(chunks, string(runes[start:end]))
		start = end
	}
	return chunks
}
