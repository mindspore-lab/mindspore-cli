package permission

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"
)

// FilePermissionStore 基于文件的权限存储
type FilePermissionStore struct {
	mu        sync.RWMutex
	filepath  string
	decisions []PermissionDecision
}

type permissionSettingsFile struct {
	Permissions *permissionRuleBuckets `json:"permissions"`
}

type permissionRuleBuckets struct {
	Allow []string `json:"allow,omitempty"`
	Ask   []string `json:"ask,omitempty"`
	Deny  []string `json:"deny,omitempty"`
}

// NewFilePermissionStore 创建文件权限存储
func NewFilePermissionStore(path string) (*FilePermissionStore, error) {
	if path == "" {
		path = ".mscode/permissions.json"
	}

	// 确保目录存在
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("create directory: %w", err)
	}

	store := &FilePermissionStore{
		filepath:  path,
		decisions: make([]PermissionDecision, 0),
	}

	// 尝试加载已有数据
	if err := store.load(); err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("load permissions: %w", err)
	}

	return store, nil
}

// SaveDecision 保存权限决策
func (s *FilePermissionStore) SaveDecision(decision PermissionDecision) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// 检查是否已存在相同的决策
	for i, d := range s.decisions {
		if d.Tool == decision.Tool && d.Action == decision.Action && d.Path == decision.Path {
			s.decisions[i] = decision
			return s.save()
		}
	}

	// 添加新决策
	s.decisions = append(s.decisions, decision)
	return s.save()
}

// LoadDecisions 加载所有权限决策
func (s *FilePermissionStore) LoadDecisions() ([]PermissionDecision, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make([]PermissionDecision, len(s.decisions))
	copy(result, s.decisions)
	return result, nil
}

// ClearDecisions 清除所有权限决策
func (s *FilePermissionStore) ClearDecisions() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.decisions = make([]PermissionDecision, 0)
	return s.save()
}

// GetDecisionForTool 获取指定工具的决策
func (s *FilePermissionStore) GetDecisionForTool(tool string) *PermissionDecision {
	s.mu.RLock()
	defer s.mu.RUnlock()

	for _, d := range s.decisions {
		if d.Tool == tool && d.Action == "" && d.Path == "" {
			return &d
		}
	}
	return nil
}

// GetDecisionForCommand 获取指定命令的决策
func (s *FilePermissionStore) GetDecisionForCommand(command string) *PermissionDecision {
	s.mu.RLock()
	defer s.mu.RUnlock()

	cmd := extractCommandName(command)
	for _, d := range s.decisions {
		if d.Tool == "shell" {
			storedCmd := extractCommandName(d.Action)
			if storedCmd == cmd {
				return &d
			}
		}
	}
	return nil
}

// RemoveExpiredDecisions 移除过期决策
func (s *FilePermissionStore) RemoveExpiredDecisions(maxAge time.Duration) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	cutoff := time.Now().Add(-maxAge)
	valid := make([]PermissionDecision, 0, len(s.decisions))

	for _, d := range s.decisions {
		if d.Timestamp.After(cutoff) {
			valid = append(valid, d)
		}
	}

	s.decisions = valid
	return s.save()
}

// save 保存到文件（必须持有锁）
func (s *FilePermissionStore) save() error {
	settings := settingsFromDecisions(s.decisions)
	data, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal settings: %w", err)
	}

	return os.WriteFile(s.filepath, data, 0644)
}

// load 从文件加载（必须持有锁）
func (s *FilePermissionStore) load() error {
	data, err := os.ReadFile(s.filepath)
	if err != nil {
		return err
	}

	decisions, err := decodeDecisions(data)
	if err != nil {
		return err
	}
	s.decisions = decisions
	return nil
}

// GetFilePath 获取存储文件路径
func (s *FilePermissionStore) GetFilePath() string {
	return s.filepath
}

// MemoryPermissionStore 内存权限存储
type MemoryPermissionStore struct {
	mu        sync.RWMutex
	decisions []PermissionDecision
}

// NewMemoryPermissionStore 创建内存权限存储
func NewMemoryPermissionStore() *MemoryPermissionStore {
	return &MemoryPermissionStore{
		decisions: make([]PermissionDecision, 0),
	}
}

// SaveDecision 保存权限决策
func (s *MemoryPermissionStore) SaveDecision(decision PermissionDecision) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// 更新或添加
	for i, d := range s.decisions {
		if d.Tool == decision.Tool && d.Action == decision.Action && d.Path == decision.Path {
			s.decisions[i] = decision
			return nil
		}
	}

	s.decisions = append(s.decisions, decision)
	return nil
}

// LoadDecisions 加载所有权限决策
func (s *MemoryPermissionStore) LoadDecisions() ([]PermissionDecision, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make([]PermissionDecision, len(s.decisions))
	copy(result, s.decisions)
	return result, nil
}

// ClearDecisions 清除所有权限决策
func (s *MemoryPermissionStore) ClearDecisions() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.decisions = make([]PermissionDecision, 0)
	return nil
}

// PermissionStoreConfig 权限存储配置
type PermissionStoreConfig struct {
	Type   string        // "file" or "memory"
	Path   string        // for file store
	MaxAge time.Duration // 决策最大有效期
}

// DefaultPermissionStoreConfig 返回默认配置
func DefaultPermissionStoreConfig() PermissionStoreConfig {
	return PermissionStoreConfig{
		Type:   "file",
		Path:   ".mscode/permissions.state.json",
		MaxAge: 7 * 24 * time.Hour, // 7 days
	}
}

// NewPermissionStore 创建权限存储
func NewPermissionStore(cfg PermissionStoreConfig) (PermissionStore, error) {
	switch cfg.Type {
	case "file":
		return NewFilePermissionStore(cfg.Path)
	case "memory":
		return NewMemoryPermissionStore(), nil
	default:
		return nil, fmt.Errorf("unknown store type: %s", cfg.Type)
	}
}

// PermissionStats 权限统计
type PermissionStats struct {
	TotalDecisions int
	ByLevel        map[PermissionLevel]int
	ByTool         map[string]int
	OldestDecision time.Time
	NewestDecision time.Time
}

// GetStats 获取权限决策统计
func (s *FilePermissionStore) GetStats() PermissionStats {
	s.mu.RLock()
	defer s.mu.RUnlock()

	stats := PermissionStats{
		TotalDecisions: len(s.decisions),
		ByLevel:        make(map[PermissionLevel]int),
		ByTool:         make(map[string]int),
	}

	if len(s.decisions) == 0 {
		return stats
	}

	stats.OldestDecision = s.decisions[0].Timestamp
	stats.NewestDecision = s.decisions[0].Timestamp

	for _, d := range s.decisions {
		stats.ByLevel[d.Level]++
		stats.ByTool[d.Tool]++

		if d.Timestamp.Before(stats.OldestDecision) {
			stats.OldestDecision = d.Timestamp
		}
		if d.Timestamp.After(stats.NewestDecision) {
			stats.NewestDecision = d.Timestamp
		}
	}

	return stats
}

// ExportToFile 导出权限决策到文件
func (s *FilePermissionStore) ExportToFile(exportPath string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	data, err := json.MarshalIndent(settingsFromDecisions(s.decisions), "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(exportPath, data, 0644)
}

// ImportFromFile 从文件导入权限决策
func (s *FilePermissionStore) ImportFromFile(importPath string) error {
	data, err := os.ReadFile(importPath)
	if err != nil {
		return err
	}

	imported, err := decodeDecisions(data)
	if err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// 合并导入的决策
	for _, newDecision := range imported {
		found := false
		for i, existing := range s.decisions {
			if existing.Tool == newDecision.Tool &&
				existing.Action == newDecision.Action &&
				existing.Path == newDecision.Path {
				// 更新现有决策
				s.decisions[i] = newDecision
				found = true
				break
			}
		}
		if !found {
			s.decisions = append(s.decisions, newDecision)
		}
	}

	return s.save()
}

func decodeDecisions(data []byte) ([]PermissionDecision, error) {
	trimmed := strings.TrimSpace(string(data))
	if trimmed == "" {
		return []PermissionDecision{}, nil
	}

	if strings.HasPrefix(trimmed, "[") {
		return nil, fmt.Errorf("invalid settings format: expected object with permissions buckets, got legacy array")
	}

	var settings permissionSettingsFile
	if err := json.Unmarshal(data, &settings); err != nil {
		return nil, fmt.Errorf("invalid settings JSON: %w", err)
	}
	if !hasPermissionsObject(settings) {
		return nil, fmt.Errorf("invalid settings format: missing \"permissions\" object")
	}
	normalizeLegacySettingsToolCase(&settings)

	return decisionsFromSettings(settings)
}

func hasPermissionsObject(settings permissionSettingsFile) bool {
	return settings.Permissions != nil
}

func decisionsFromSettings(settings permissionSettingsFile) ([]PermissionDecision, error) {
	if settings.Permissions == nil {
		return nil, fmt.Errorf("missing permissions object")
	}
	out := make([]PermissionDecision, 0, len(settings.Permissions.Allow)+len(settings.Permissions.Ask)+len(settings.Permissions.Deny))
	now := time.Now()
	appendRules := func(bucket string, rules []string, level PermissionLevel) error {
		for idx, raw := range rules {
			decision, err := decisionFromRule(raw, level, now, bucket, idx)
			if err != nil {
				return err
			}
			out = append(out, decision)
		}
		return nil
	}
	if err := appendRules("allow", settings.Permissions.Allow, PermissionAllowSession); err != nil {
		return nil, err
	}
	if err := appendRules("ask", settings.Permissions.Ask, PermissionAsk); err != nil {
		return nil, err
	}
	if err := appendRules("deny", settings.Permissions.Deny, PermissionDeny); err != nil {
		return nil, err
	}
	return out, nil
}

func decisionFromRule(raw string, level PermissionLevel, ts time.Time, bucket string, idx int) (PermissionDecision, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return PermissionDecision{}, fmt.Errorf("permissions.%s[%d]: rule cannot be empty", bucket, idx)
	}
	if err := validateRuleToolCase(raw, bucket, idx); err != nil {
		return PermissionDecision{}, err
	}
	rule, err := ParsePermissionRule(raw)
	if err != nil {
		return PermissionDecision{}, fmt.Errorf("permissions.%s[%d]: %w", bucket, idx, err)
	}
	d := PermissionDecision{
		Level:     level,
		Timestamp: ts,
	}
	switch rule.Tool {
	case "bash":
		d.Tool = "shell"
		d.Action = rule.Specifier
	case "read":
		d.Tool = "read"
		d.Path = rule.Specifier
	case "edit":
		d.Tool = "edit"
		d.Path = rule.Specifier
	case "write":
		d.Tool = "write"
		d.Path = rule.Specifier
	case "webfetch":
		d.Tool = "webfetch"
		d.Action = rule.Specifier
	case "agent":
		d.Tool = "agent"
		d.Action = rule.Specifier
	default:
		d.Tool = rule.Tool
		d.Action = rule.Specifier
	}
	if strings.TrimSpace(d.Tool) == "" {
		return PermissionDecision{}, fmt.Errorf("permissions.%s[%d]: invalid rule %q", bucket, idx, raw)
	}
	return d, nil
}

func validateRuleToolCase(raw, bucket string, idx int) error {
	tool := raw
	if open := strings.Index(raw, "("); open >= 0 {
		tool = raw[:open]
	}
	tool = strings.TrimSpace(tool)
	if tool == "" {
		return fmt.Errorf("permissions.%s[%d]: invalid rule %q", bucket, idx, raw)
	}
	if strings.HasPrefix(strings.ToLower(tool), "mcp__") {
		return nil
	}
	r, sz := utf8.DecodeRuneInString(tool)
	if r == utf8.RuneError && sz == 0 {
		return fmt.Errorf("permissions.%s[%d]: invalid rule %q", bucket, idx, raw)
	}
	if !unicode.IsUpper(r) {
		suggestion := strings.ToUpper(string(r)) + tool[sz:]
		return fmt.Errorf("permissions.%s[%d]: %q: Tool names must start with uppercase. Use %q", bucket, idx, tool, suggestion)
	}
	return nil
}

func normalizeLegacySettingsToolCase(settings *permissionSettingsFile) {
	if settings == nil || settings.Permissions == nil {
		return
	}
	normalizeBucket := func(rules []string) {
		for i, rule := range rules {
			rules[i] = normalizeLegacyToolRule(rule)
		}
	}
	normalizeBucket(settings.Permissions.Allow)
	normalizeBucket(settings.Permissions.Ask)
	normalizeBucket(settings.Permissions.Deny)
}

func normalizeLegacyToolRule(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return raw
	}

	open := strings.Index(raw, "(")
	tool := raw
	suffix := ""
	if open >= 0 {
		tool = strings.TrimSpace(raw[:open])
		suffix = raw[open:]
	}
	if !shouldNormalizeLegacyToolLiteral(tool) {
		return raw
	}
	return uppercaseToolLiteral(tool) + suffix
}

func shouldNormalizeLegacyToolLiteral(tool string) bool {
	tool = strings.TrimSpace(tool)
	if tool == "" {
		return false
	}
	lower := strings.ToLower(tool)
	if strings.HasPrefix(lower, "mcp__") {
		return false
	}
	r, size := utf8.DecodeRuneInString(tool)
	if r == utf8.RuneError && size == 0 {
		return false
	}
	if !unicode.IsLower(r) {
		return false
	}
	// Older session state persisted custom tool names like "load_skill"
	// verbatim. Normalize those snake/kebab-case names on load so resume can
	// restore legacy session permissions without surfacing a settings error.
	return strings.ContainsAny(tool, "_-")
}

func settingsFromDecisions(decisions []PermissionDecision) permissionSettingsFile {
	settings := permissionSettingsFile{
		Permissions: &permissionRuleBuckets{
			Allow: make([]string, 0),
			Ask:   make([]string, 0),
			Deny:  make([]string, 0),
		},
	}

	seenAllow := map[string]struct{}{}
	seenAsk := map[string]struct{}{}
	seenDeny := map[string]struct{}{}

	for _, d := range decisions {
		rule := strings.TrimSpace(ruleFromDecision(d))
		if rule == "" {
			continue
		}
		switch d.Level {
		case PermissionDeny:
			if _, ok := seenDeny[rule]; ok {
				continue
			}
			seenDeny[rule] = struct{}{}
			settings.Permissions.Deny = append(settings.Permissions.Deny, rule)
		case PermissionAsk:
			if _, ok := seenAsk[rule]; ok {
				continue
			}
			seenAsk[rule] = struct{}{}
			settings.Permissions.Ask = append(settings.Permissions.Ask, rule)
		default:
			if _, ok := seenAllow[rule]; ok {
				continue
			}
			seenAllow[rule] = struct{}{}
			settings.Permissions.Allow = append(settings.Permissions.Allow, rule)
		}
	}

	if len(settings.Permissions.Allow) == 0 {
		settings.Permissions.Allow = nil
	}
	if len(settings.Permissions.Ask) == 0 {
		settings.Permissions.Ask = nil
	}
	if len(settings.Permissions.Deny) == 0 {
		settings.Permissions.Deny = nil
	}
	return settings
}

func ruleFromDecision(d PermissionDecision) string {
	tool := strings.ToLower(strings.TrimSpace(d.Tool))
	action := strings.TrimSpace(d.Action)
	path := strings.TrimSpace(d.Path)

	if tool == "shell" {
		if action == "" {
			return "Bash"
		}
		return fmt.Sprintf("Bash(%s)", action)
	}
	if path != "" {
		spec := path
		if filepath.IsAbs(spec) {
			spec = "//" + strings.TrimPrefix(filepath.ToSlash(spec), "/")
		}
		switch tool {
		case "read", "grep", "glob":
			return fmt.Sprintf("Read(%s)", spec)
		case "write":
			return fmt.Sprintf("Write(%s)", spec)
		default:
			return fmt.Sprintf("Edit(%s)", spec)
		}
	}
	if action != "" {
		switch tool {
		case "webfetch":
			return fmt.Sprintf("WebFetch(%s)", action)
		case "agent":
			return fmt.Sprintf("Agent(%s)", action)
		default:
			return fmt.Sprintf("%s(%s)", canonicalRuleTool(tool), action)
		}
	}
	return canonicalRuleTool(tool)
}

func canonicalRuleTool(tool string) string {
	switch strings.ToLower(strings.TrimSpace(tool)) {
	case "shell":
		return "Bash"
	case "read", "grep", "glob":
		return "Read"
	case "edit":
		return "Edit"
	case "write":
		return "Write"
	case "webfetch":
		return "WebFetch"
	case "agent":
		return "Agent"
	default:
		return uppercaseToolLiteral(tool)
	}
}

func uppercaseToolLiteral(tool string) string {
	tool = strings.TrimSpace(tool)
	if tool == "" {
		return ""
	}
	if strings.HasPrefix(strings.ToLower(tool), "mcp__") {
		return tool
	}
	r, size := utf8.DecodeRuneInString(tool)
	if r == utf8.RuneError && size == 0 {
		return tool
	}
	return strings.ToUpper(string(r)) + tool[size:]
}
