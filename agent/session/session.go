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

	"github.com/vigo999/ms-cli/integrations/llm"
	"github.com/vigo999/ms-cli/ui/model"
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
	defaultSessionSubdir = ".ms-cli/sessions"
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
	SessionID    string        `json:"session_id"`
	WorkDir      string        `json:"workdir"`
	SystemPrompt string        `json:"system_prompt"`
	UpdatedAt    time.Time     `json:"updated_at"`
	Messages     []llm.Message `json:"messages,omitempty"`
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

// Create allocates a new session under ~/.ms-cli/sessions.
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
	return loadFromPath(path)
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
	if s == nil {
		return nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	events := make([]model.Event, 0, len(s.records))
	for _, record := range s.records {
		switch record.Type {
		case recordTypeUser:
			events = append(events, model.Event{Type: model.UserInput, Message: record.Content})
		case recordTypeAssistant:
			events = append(events, model.Event{Type: model.AgentReply, Message: record.Content})
		case recordTypeToolCall:
			events = append(events, replayToolCallEvent(record))
		case recordTypeToolResult:
			if record.ToolName == "load_skill" {
				continue
			}
			events = append(events, replayToolResultEvent(record))
		case recordTypeSkill:
			events = append(events, replaySkillEvent(record.SkillName))
		}
	}
	return events
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

func loadFromPath(path string) (*Session, error) {
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

	appender, enc, err := openAppender(path, true)
	if err != nil {
		return nil, err
	}

	snapPath := snapshotPath(path)
	snapshot, err := loadSnapshot(snapPath)
	if err != nil {
		_ = appender.Close()
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

	return &Session{
		meta:         meta,
		records:      records,
		snapshot:     snapshot,
		path:         path,
		snapshotPath: snapPath,
		persisted:    true,
		file:         appender,
		enc:          enc,
	}, nil
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
	if !s.persisted {
		return nil
	}
	return s.writeSnapshotLocked()
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

func workDirKey(absWorkDir string) string {
	return strings.ReplaceAll(filepath.Clean(absWorkDir), string(os.PathSeparator), "-")
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
		Type:     model.ToolCallStart,
		ToolName: record.ToolName,
		Message:  describeToolCall(record.ToolName, record.Arguments),
	}
}

func replayToolResultEvent(record MessageRecord) model.Event {
	return model.Event{
		Type:     model.ToolReplay,
		ToolName: record.ToolName,
		Message:  record.Content,
	}
}

func replaySkillEvent(skillName string) model.Event {
	return model.Event{
		Type:     model.ToolSkill,
		ToolName: "load_skill",
		Message:  skillName,
		Summary:  skillSummary(skillName),
	}
}
