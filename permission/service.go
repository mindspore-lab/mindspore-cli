package permission

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/mindspore-lab/mindspore-cli/configs"
)

// PermissionService controls tool-call permissions.
type PermissionService interface {
	// Request requests permission to execute a tool.
	// Returns (granted, error).
	Request(ctx context.Context, tool, action, path string) (bool, error)

	// Check checks the current permission level without triggering interaction.
	Check(tool, action string) PermissionLevel

	// CheckCommand checks permission for a specific command.
	CheckCommand(command string) PermissionLevel

	// CheckPath checks permission for a specific path.
	CheckPath(path string) PermissionLevel

	// Grant grants permission for a tool.
	Grant(tool string, level PermissionLevel)

	// GrantCommand grants permission for a specific command.
	GrantCommand(command string, level PermissionLevel)

	// GrantPath grants permission for a specific path pattern.
	GrantPath(pattern string, level PermissionLevel)

	// Revoke revokes permission for a tool.
	Revoke(tool string)

	// RevokeCommand revokes permission for a specific command.
	RevokeCommand(command string)

	// RevokePath revokes permission for a specific path pattern.
	RevokePath(pattern string)
}

// DefaultPermissionService is the default permission service implementation.
type DefaultPermissionService struct {
	mu              sync.RWMutex
	policies        map[string]PermissionLevel
	commandPolicies map[string]PermissionLevel
	pathPatterns    []PathPermission
	denyRules       []compiledRule
	askRules        []compiledRule
	allowRules      []compiledRule
	ruleSeq         int
	default_        PermissionLevel
	skipAsk         bool
	ui              PermissionUI
	store           PermissionStore
}

const (
	ruleSourceConfig  = "config"
	ruleSourceProject = "project"
	ruleSourceUser    = "user"
	ruleSourceSession = "session"
	ruleSourceState   = "state"
	ruleSourceRuntime = "runtime"
	ruleSourceManaged = "managed"
)

var ErrManagedRuleLocked = errors.New("managed rule is immutable")

// PathPermission 路径权限
type PathPermission struct {
	Pattern string
	Level   PermissionLevel
}

// PermissionUI is the interface for permission UI interaction.
type PermissionUI interface {
	// RequestPermission asks the user for permission.
	// Returns (granted, remember, error).
	RequestPermission(tool, action, path string) (bool, bool, error)
}

// NewDefaultPermissionService creates a new permission service.
func NewDefaultPermissionService(cfg configs.PermissionsConfig) *DefaultPermissionService {
	svc := &DefaultPermissionService{
		policies:        make(map[string]PermissionLevel),
		commandPolicies: make(map[string]PermissionLevel),
		pathPatterns:    make([]PathPermission, 0),
		denyRules:       make([]compiledRule, 0),
		askRules:        make([]compiledRule, 0),
		allowRules:      make([]compiledRule, 0),
		ruleSeq:         0,
		default_:        ParsePermissionLevel(cfg.DefaultLevel),
		skipAsk:         cfg.SkipRequests,
	}

	for _, rule := range cfg.Deny {
		_ = svc.addRuleNoLock(rule, PermissionDeny, configRuleSource(cfg, rule))
	}
	for _, rule := range cfg.Ask {
		_ = svc.addRuleNoLock(rule, PermissionAsk, configRuleSource(cfg, rule))
	}
	for _, rule := range cfg.Allow {
		_ = svc.addRuleNoLock(rule, PermissionAllowAlways, configRuleSource(cfg, rule))
	}

	// Load tool policies
	for tool, level := range cfg.ToolPolicies {
		svc.grantWithSource(tool, ParsePermissionLevel(level), ruleSourceConfig)
	}

	// Load allowed tools as allow_always
	for _, tool := range cfg.AllowedTools {
		svc.grantWithSource(tool, PermissionAllowAlways, ruleSourceConfig)
	}

	// Load blocked tools as deny
	for _, tool := range cfg.BlockedTools {
		svc.grantWithSource(tool, PermissionDeny, ruleSourceConfig)
	}

	return svc
}

func configRuleSource(cfg configs.PermissionsConfig, rule string) string {
	rule = strings.TrimSpace(rule)
	if rule == "" {
		return ruleSourceConfig
	}
	if cfg.RuleSources == nil {
		return ruleSourceConfig
	}
	if src, ok := cfg.RuleSources[rule]; ok && strings.TrimSpace(src) != "" {
		return strings.ToLower(strings.TrimSpace(src))
	}
	return ruleSourceConfig
}

// SetUI sets the permission UI.
func (s *DefaultPermissionService) SetUI(ui PermissionUI) {
	s.ui = ui
}

// SetStore sets the permission store.
func (s *DefaultPermissionService) SetStore(store PermissionStore) {
	s.store = store
	if s.store == nil {
		return
	}
	decisions, err := s.store.LoadDecisions()
	if err != nil {
		return
	}
	for _, d := range decisions {
		if d.Action != "" && d.Tool == "shell" {
			s.grantCommandWithSource(d.Action, d.Level, ruleSourceState)
			continue
		}
		if d.Path != "" {
			s.grantPathWithSource(d.Path, d.Level, ruleSourceState)
			continue
		}
		s.grantWithSource(d.Tool, d.Level, ruleSourceState)
	}
}

// Request requests permission.
func (s *DefaultPermissionService) Request(ctx context.Context, tool, action, path string) (bool, error) {
	shellIntent := shellIntentGeneral
	if tool == "shell" {
		shellIntent = classifyShellCommand(action)
		if shellIntent == shellIntentFileAuthoring && (s.Check("write", "") == PermissionDeny || s.Check("edit", "") == PermissionDeny) {
			return false, fmt.Errorf("shell file authoring is blocked; use write/edit tool permissions")
		}
	}

	// 1. 检查工具级别权限
	toolLevel := s.Check(tool, action)
	if tool == "shell" && shellIntent == shellIntentReadOnlyExplore {
		toolLevel = PermissionAllowAlways
	}
	level := toolLevel
	cmdLevel := PermissionAllowAlways
	pathLevel := PermissionAllowAlways

	// 2. 检查命令级别权限（如果是 shell 工具）
	if tool == "shell" && action != "" {
		cmdLevel = s.CheckCommand(action)
		if cmdLevel < level {
			level = cmdLevel
		}
	}

	// 3. 检查路径级别权限
	if path != "" {
		pathLevel = s.CheckPath(path)
		if pathLevel < level {
			level = pathLevel
		}
	}

	switch level {
	case PermissionDeny:
		return false, fmt.Errorf("tool %q is blocked", tool)

	case PermissionAllowAlways, PermissionAllowSession:
		return true, nil

	case PermissionAllowOnce:
		// Consume "allow once" on the scope that granted it.
		switch {
		case tool == "shell" && action != "" && cmdLevel == PermissionAllowOnce:
			s.grantCommandWithSource(action, PermissionAsk, ruleSourceRuntime)
		case path != "" && pathLevel == PermissionAllowOnce:
			s.grantPathWithSource(path, PermissionAsk, ruleSourceRuntime)
		case toolLevel == PermissionAllowOnce:
			s.grantWithSource(tool, PermissionAsk, ruleSourceRuntime)
		default:
			s.grantWithSource(tool, PermissionAsk, ruleSourceRuntime)
		}
		return true, nil

	case PermissionAsk:
		if s.skipAsk {
			return true, nil
		}

		// Interactive permission request
		if s.ui != nil {
			granted, remember, err := s.ui.RequestPermission(tool, action, path)
			if err != nil {
				return false, err
			}

			if granted && remember {
				shouldPersist := true
				decisions := make([]PermissionDecision, 0, 5)
				switch {
				case isEditPermissionTool(tool):
					// Claude Code style: "allow all edits during this session"
					// applies to both edit and write tools and is not persisted.
					s.grantWithSource("edit", PermissionAllowSession, ruleSourceSession)
					s.grantWithSource("write", PermissionAllowSession, ruleSourceSession)
					shouldPersist = false
				case tool == "shell" && action != "":
					parts := splitShellCommand(normalizeCommandInput(action))
					if len(parts) == 0 {
						parts = []string{normalizeCommandInput(action)}
					}
					if len(parts) > 5 {
						parts = parts[:5]
					}
					for _, cmd := range parts {
						if strings.TrimSpace(cmd) == "" {
							continue
						}
						s.grantCommandWithSource(cmd, PermissionAllowSession, ruleSourceState)
						decisions = append(decisions, PermissionDecision{
							Tool:      tool,
							Action:    cmd,
							Path:      path,
							Level:     PermissionAllowSession,
							Timestamp: time.Now(),
						})
					}
				case path != "":
					s.grantPathWithSource(path, PermissionAllowSession, ruleSourceState)
					decisions = append(decisions, PermissionDecision{
						Tool:      tool,
						Action:    action,
						Path:      path,
						Level:     PermissionAllowSession,
						Timestamp: time.Now(),
					})
				default:
					s.grantWithSource(tool, PermissionAllowSession, ruleSourceState)
					decisions = append(decisions, PermissionDecision{
						Tool:      tool,
						Action:    action,
						Path:      path,
						Level:     PermissionAllowSession,
						Timestamp: time.Now(),
					})
				}
				// 持久化决策
				if shouldPersist && s.store != nil {
					for _, d := range decisions {
						_ = s.store.SaveDecision(d)
					}
				}
			}

			return granted, nil
		}

		// No UI, default to allow
		return true, nil
	}

	return false, fmt.Errorf("unknown permission level")
}

// Check checks the permission level.
func (s *DefaultPermissionService) Check(tool, action string) PermissionLevel {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if level, matched := s.evaluateRules(tool, action, ""); matched {
		return level
	}

	// Check specific tool policy
	if level, ok := s.policies[tool]; ok {
		return level
	}

	// Check read/write patterns
	if tool == "read" || tool == "glob" || tool == "grep" || tool == "list_dir" {
		return PermissionAllowAlways
	}

	// Check destructive commands
	if tool == "write" || tool == "edit" {
		return maxPermission(s.default_, PermissionAsk)
	}

	if tool == "shell" {
		return maxPermission(s.default_, PermissionAsk)
	}

	return s.default_
}

// CheckCommand checks permission for a specific command.
func (s *DefaultPermissionService) CheckCommand(command string) PermissionLevel {
	s.mu.RLock()
	defer s.mu.RUnlock()

	command = normalizeCommandInput(command)
	if strings.TrimSpace(command) != "" {
		parts := splitShellCommand(command)
		if len(parts) > 1 {
			level := PermissionAllowAlways
			for _, part := range parts {
				partLevel := s.checkShellCommandSingle(part)
				if partLevel < level {
					level = partLevel
				}
			}
			return level
		}
	}
	return s.checkShellCommandSingle(command)
}

// safeReadOnlyCommands are shell commands that only read state and cannot
// modify files, network, or system configuration. These are auto-allowed
// without prompting.
var safeReadOnlyCommands = map[string]bool{
	"ls": true, "ll": true, "la": true,
	"cat": true, "head": true, "tail": true, "less": true, "more": true,
	"find": true, "locate": true, "which": true, "whereis": true,
	"pwd": true, "echo": true, "printf": true,
	"wc": true, "sort": true, "uniq": true, "diff": true, "comm": true,
	"grep": true, "rg": true, "ag": true, "awk": true, "sed": true,
	"file": true, "stat": true, "du": true, "df": true,
	"tree": true, "realpath": true, "basename": true, "dirname": true,
	"env": true, "printenv": true, "uname": true, "whoami": true, "id": true,
	"date": true, "cal": true, "uptime": true, "hostname": true,
	"ps": true, "top": true, "htop": true, "free": true,
	"go": true, "python": true, "python3": true, "node": true, "cargo": true,
	"git": true, "jq": true, "yq": true, "xargs": true,
	"md5sum": true, "sha256sum": true, "sha1sum": true,
	"tar": true, "gzip": true, "gunzip": true, "zcat": true,
	"true": true, "false": true, "test": true,
}

type shellIntent string

const (
	shellIntentGeneral         shellIntent = "general"
	shellIntentReadOnlyExplore shellIntent = "read_only_explore"
	shellIntentFileAuthoring   shellIntent = "file_authoring"
)

var readOnlyExploreCommands = map[string]bool{
	"ls": true, "ll": true, "la": true,
	"find": true, "tree": true, "pwd": true,
	"stat": true, "realpath": true, "basename": true, "dirname": true,
	"rg": true,
}

func classifyShellCommand(command string) shellIntent {
	command = normalizeCommandInput(command)
	if strings.TrimSpace(command) == "" {
		return shellIntentGeneral
	}

	parts := splitShellCommand(command)
	if len(parts) == 0 {
		parts = []string{command}
	}

	allReadOnly := true
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if isShellFileAuthoringCommand(part) {
			return shellIntentFileAuthoring
		}
		if !isReadOnlyExploreCommand(part) {
			allReadOnly = false
		}
	}
	if allReadOnly {
		return shellIntentReadOnlyExplore
	}
	return shellIntentGeneral
}

func isReadOnlyExploreCommand(command string) bool {
	command = strings.TrimSpace(normalizeCommandInput(command))
	if command == "" || isShellFileAuthoringCommand(command) {
		return false
	}
	name := extractCommandName(command)
	if !readOnlyExploreCommands[name] {
		return false
	}
	if name == "rg" && !strings.Contains(command, "--files") {
		return false
	}
	return true
}

func isShellFileAuthoringCommand(command string) bool {
	command = strings.TrimSpace(normalizeCommandInput(command))
	if command == "" {
		return false
	}
	lower := strings.ToLower(command)
	if strings.Contains(command, ">>") || strings.Contains(command, " >") || strings.Contains(command, "> ") {
		return true
	}
	if strings.Contains(command, "<<") {
		return true
	}
	if strings.Contains(lower, " tee ") || strings.HasPrefix(lower, "tee ") {
		return true
	}
	if strings.HasPrefix(lower, "sed ") && strings.Contains(lower, " -i") {
		return true
	}
	if strings.HasPrefix(lower, "perl ") && strings.Contains(lower, "-pi") {
		return true
	}
	return false
}

func (s *DefaultPermissionService) checkShellCommandSingle(command string) PermissionLevel {
	if level, matched := s.evaluateRules("shell", command, ""); matched {
		return level
	}

	// Parse command name
	cmd := extractCommandName(command)

	// Check command specific policy
	if level, ok := s.commandPolicies[cmd]; ok {
		return level
	}

	// Check dangerous commands
	if IsDangerousCommand(command) {
		return minPermission(s.default_, PermissionAsk)
	}

	// Auto-allow safe read-only commands
	if safeReadOnlyCommands[cmd] {
		return PermissionAllowAlways
	}

	return s.default_
}

// CheckPath checks permission for a specific path.
func (s *DefaultPermissionService) CheckPath(path string) PermissionLevel {
	s.mu.RLock()
	defer s.mu.RUnlock()

	matched := false
	level := PermissionAllowAlways
	for _, tool := range []string{"read", "edit", "write"} {
		if l, ok := s.evaluateRules(tool, "", path); ok {
			matched = true
			if l < level {
				level = l
			}
		}
	}
	if matched {
		return level
	}

	// Legacy path patterns
	for _, pp := range s.pathPatterns {
		if matched, _ := filepath.Match(pp.Pattern, path); matched {
			return pp.Level
		}
	}

	return PermissionAllowAlways
}

// Grant grants permission.
// Compatibility API: prefer AddRule for explicit source-aware rule management.
func (s *DefaultPermissionService) Grant(tool string, level PermissionLevel) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.grantWithSource(tool, level, ruleSourceRuntime)
}

func (s *DefaultPermissionService) grantWithSource(tool string, level PermissionLevel, source string) {
	rule := ruleFromToolLiteral(tool)
	if !s.canMutateRuleWithSourceNoLock(rule, source) {
		return
	}
	s.policies[tool] = level
	_ = s.addRuleNoLock(rule, level, source)
}

// GrantCommand grants permission for a specific command.
// Compatibility API: prefer AddRule(Bash(...)) for explicit source-aware rule management.
func (s *DefaultPermissionService) GrantCommand(command string, level PermissionLevel) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.grantCommandWithSource(command, level, ruleSourceRuntime)
}

func (s *DefaultPermissionService) grantCommandWithSource(command string, level PermissionLevel, source string) {
	cmd := extractCommandName(command)
	if cmd == "" {
		return
	}
	rule := fmt.Sprintf("Bash(%s *)", cmd)
	if !s.canMutateRuleWithSourceNoLock(rule, source) {
		return
	}
	s.commandPolicies[cmd] = level
	_ = s.addRuleNoLock(rule, level, source)
}

// GrantPath grants permission for a specific path pattern.
// Compatibility API: prefer AddRule(Edit(...)) for explicit source-aware rule management.
func (s *DefaultPermissionService) GrantPath(pattern string, level PermissionLevel) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.grantPathWithSource(pattern, level, ruleSourceRuntime)
}

func (s *DefaultPermissionService) grantPathWithSource(pattern string, level PermissionLevel, source string) {
	rule := fmt.Sprintf("Edit(%s)", pattern)
	if !s.canMutateRuleWithSourceNoLock(rule, source) {
		return
	}
	// 查找是否已存在
	for i, pp := range s.pathPatterns {
		if pp.Pattern == pattern {
			s.pathPatterns[i].Level = level
			_ = s.addRuleNoLock(rule, level, source)
			return
		}
	}

	// 添加新的
	s.pathPatterns = append(s.pathPatterns, PathPermission{
		Pattern: pattern,
		Level:   level,
	})
	_ = s.addRuleNoLock(rule, level, source)
}

// Revoke revokes permission.
// Compatibility API: prefer RemoveRule for explicit source-aware rule management.
func (s *DefaultPermissionService) Revoke(tool string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	rule := ruleFromToolLiteral(tool)
	if !s.canMutateRuleWithSourceNoLock(rule, ruleSourceRuntime) {
		return
	}
	delete(s.policies, tool)
	s.removeRuleNoLock(rule)
}

// RevokeCommand revokes permission for a specific command.
// Compatibility API: prefer RemoveRule(Bash(...)) for explicit source-aware rule management.
func (s *DefaultPermissionService) RevokeCommand(command string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	cmd := extractCommandName(command)
	if cmd == "" {
		return
	}
	rule := fmt.Sprintf("Bash(%s *)", cmd)
	if !s.canMutateRuleWithSourceNoLock(rule, ruleSourceRuntime) {
		return
	}
	delete(s.commandPolicies, cmd)
	s.removeRuleNoLock(rule)
}

// RevokePath revokes permission for a specific path pattern.
// Compatibility API: prefer RemoveRule(Edit(...)) for explicit source-aware rule management.
func (s *DefaultPermissionService) RevokePath(pattern string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	rule := fmt.Sprintf("Edit(%s)", pattern)
	if !s.canMutateRuleWithSourceNoLock(rule, ruleSourceRuntime) {
		return
	}
	for i, pp := range s.pathPatterns {
		if pp.Pattern == pattern {
			s.pathPatterns = append(s.pathPatterns[:i], s.pathPatterns[i+1:]...)
			break
		}
	}
	s.removeRuleNoLock(rule)
}

// GetPolicies returns a copy of all policies.
func (s *DefaultPermissionService) GetPolicies() map[string]PermissionLevel {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make(map[string]PermissionLevel, len(s.policies))
	for k, v := range s.policies {
		result[k] = v
	}
	return result
}

// GetCommandPolicies returns command policies.
func (s *DefaultPermissionService) GetCommandPolicies() map[string]PermissionLevel {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make(map[string]PermissionLevel, len(s.commandPolicies))
	for k, v := range s.commandPolicies {
		result[k] = v
	}
	return result
}

// GetPathPolicies returns path policies.
func (s *DefaultPermissionService) GetPathPolicies() []PathPermission {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make([]PathPermission, len(s.pathPatterns))
	copy(result, s.pathPatterns)
	return result
}

// AddRule adds a rule in Tool/Tool(specifier) syntax.
func (s *DefaultPermissionService) AddRule(rule string, level PermissionLevel) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.canMutateRuleNoLock(rule) {
		return fmt.Errorf("%w: %s", ErrManagedRuleLocked, strings.TrimSpace(rule))
	}
	return s.addRuleNoLock(rule, level, ruleSourceProject)
}

// RemoveRule removes a rule from all buckets.
func (s *DefaultPermissionService) RemoveRule(rule string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.canMutateRuleNoLock(rule) {
		return false, fmt.Errorf("%w: %s", ErrManagedRuleLocked, strings.TrimSpace(rule))
	}
	return s.removeRuleNoLock(rule), nil
}

// RuleSource returns the source of an exact rule when present.
func (s *DefaultPermissionService) RuleSource(rule string) (string, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	found, ok := s.findRuleNoLock(rule)
	if !ok {
		return "", false
	}
	return found.Source, true
}

// GetRuleViews returns normalized rules grouped by effective level bucket order.
func (s *DefaultPermissionService) GetRuleViews() []RuleViewItem {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]RuleViewItem, 0, len(s.denyRules)+len(s.askRules)+len(s.allowRules))
	for _, r := range orderedRules(s.denyRules) {
		out = append(out, RuleViewItem{
			Rule:   r.Rule.Raw,
			Level:  PermissionDeny,
			Source: r.Source,
		})
	}
	for _, r := range orderedRules(s.askRules) {
		out = append(out, RuleViewItem{
			Rule:   r.Rule.Raw,
			Level:  PermissionAsk,
			Source: r.Source,
		})
	}
	for _, r := range orderedRules(s.allowRules) {
		out = append(out, RuleViewItem{
			Rule:   r.Rule.Raw,
			Level:  r.Level,
			Source: r.Source,
		})
	}
	return out
}

func (s *DefaultPermissionService) evaluateRules(tool, action, path string) (PermissionLevel, bool) {
	denyRule, denyOK := bestMatchBySource(s.denyRules, tool, action, path)
	askRule, askOK := bestMatchBySource(s.askRules, tool, action, path)
	allowRule, allowOK := bestMatchBySource(s.allowRules, tool, action, path)

	top := -1
	if denyOK {
		top = max(top, sourcePriority(denyRule.Source))
	}
	if askOK {
		top = max(top, sourcePriority(askRule.Source))
	}
	if allowOK {
		top = max(top, sourcePriority(allowRule.Source))
	}
	if top < 0 {
		return PermissionAsk, false
	}

	if denyOK && sourcePriority(denyRule.Source) == top {
		return PermissionDeny, true
	}
	if askOK && sourcePriority(askRule.Source) == top {
		return PermissionAsk, true
	}
	if allowOK && sourcePriority(allowRule.Source) == top {
		return allowRule.Level, true
	}
	return PermissionAsk, false
}

func (s *DefaultPermissionService) addRuleNoLock(rule string, level PermissionLevel, source string) error {
	parsed, err := ParsePermissionRule(rule)
	if err != nil {
		return err
	}
	parsed.Raw = strings.TrimSpace(rule)
	if parsed.Raw == "" {
		parsed.Raw = rule
	}
	order := 0
	if existing, ok := s.findRuleNoLock(parsed.Raw); ok {
		if sourcePriority(existing.Source) > sourcePriority(source) {
			return nil
		}
		order = existing.Order
	}
	s.removeRuleNoLock(parsed.Raw)
	if order == 0 {
		s.ruleSeq++
		order = s.ruleSeq
	}

	cr := compiledRule{
		Rule:   parsed,
		Level:  level,
		Source: source,
		Order:  order,
	}
	switch level {
	case PermissionDeny:
		s.denyRules = append(s.denyRules, cr)
	case PermissionAsk:
		s.askRules = append(s.askRules, cr)
	default:
		s.allowRules = append(s.allowRules, cr)
	}
	return nil
}

func (s *DefaultPermissionService) removeRuleNoLock(rule string) bool {
	rule = strings.TrimSpace(rule)
	if rule == "" {
		return false
	}
	removed := false
	filter := func(in []compiledRule) []compiledRule {
		out := in[:0]
		for _, item := range in {
			if strings.EqualFold(strings.TrimSpace(item.Rule.Raw), rule) {
				removed = true
				continue
			}
			out = append(out, item)
		}
		return out
	}
	s.denyRules = filter(s.denyRules)
	s.askRules = filter(s.askRules)
	s.allowRules = filter(s.allowRules)
	return removed
}

func (s *DefaultPermissionService) findRuleNoLock(rule string) (compiledRule, bool) {
	rule = strings.TrimSpace(rule)
	if rule == "" {
		return compiledRule{}, false
	}
	for _, item := range s.denyRules {
		if strings.EqualFold(strings.TrimSpace(item.Rule.Raw), rule) {
			return item, true
		}
	}
	for _, item := range s.askRules {
		if strings.EqualFold(strings.TrimSpace(item.Rule.Raw), rule) {
			return item, true
		}
	}
	for _, item := range s.allowRules {
		if strings.EqualFold(strings.TrimSpace(item.Rule.Raw), rule) {
			return item, true
		}
	}
	return compiledRule{}, false
}

func (s *DefaultPermissionService) canMutateRuleNoLock(rule string) bool {
	return s.canMutateRuleWithSourceNoLock(rule, ruleSourceProject)
}

func (s *DefaultPermissionService) canMutateRuleWithSourceNoLock(rule, source string) bool {
	found, ok := s.findRuleNoLock(rule)
	if !ok {
		return true
	}
	return sourcePriority(source) >= sourcePriority(found.Source)
}

func sourcePriority(source string) int {
	switch strings.ToLower(strings.TrimSpace(source)) {
	case ruleSourceManaged:
		return 100
	case ruleSourceSession:
		return 90
	case ruleSourceState:
		return 80
	case ruleSourceProject:
		return 70
	case ruleSourceUser:
		return 60
	case ruleSourceConfig:
		return 50
	case ruleSourceRuntime:
		return 40
	default:
		return 10
	}
}

func bestMatchBySource(rules []compiledRule, tool, action, path string) (compiledRule, bool) {
	best := compiledRule{}
	bestPri := -1
	for _, r := range orderedRules(rules) {
		if !matchRule(r.Rule, tool, action, path) {
			continue
		}
		p := sourcePriority(r.Source)
		if p > bestPri {
			best = r
			bestPri = p
		}
	}
	if bestPri < 0 {
		return compiledRule{}, false
	}
	return best, true
}

func orderedRules(in []compiledRule) []compiledRule {
	if len(in) <= 1 {
		return in
	}
	out := make([]compiledRule, len(in))
	copy(out, in)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Order == out[j].Order {
			return out[i].Rule.Raw < out[j].Rule.Raw
		}
		return out[i].Order < out[j].Order
	})
	return out
}

func maxPermission(a, b PermissionLevel) PermissionLevel {
	if a > b {
		return a
	}
	return b
}

func minPermission(a, b PermissionLevel) PermissionLevel {
	if a < b {
		return a
	}
	return b
}

// extractCommandName 从命令字符串中提取命令名
func extractCommandName(command string) string {
	// 去除前导空格
	command = strings.TrimLeft(command, " \t")

	// 提取第一个词
	parts := strings.Fields(command)
	if len(parts) == 0 {
		return ""
	}

	return parts[0]
}

func normalizeCommandInput(command string) string {
	command = strings.TrimSpace(command)
	if command == "" {
		return ""
	}

	var payload struct {
		Command string `json:"command"`
	}
	if err := json.Unmarshal([]byte(command), &payload); err == nil {
		if cmd := strings.TrimSpace(payload.Command); cmd != "" {
			return cmd
		}
	}

	return command
}

func isEditPermissionTool(tool string) bool {
	switch strings.TrimSpace(strings.ToLower(tool)) {
	case "edit", "write":
		return true
	default:
		return false
	}
}

// NoOpPermissionService is a permission service that always allows.
type NoOpPermissionService struct{}

// NewNoOpPermissionService creates a no-op permission service.
func NewNoOpPermissionService() *NoOpPermissionService {
	return &NoOpPermissionService{}
}

// Request always grants permission.
func (s *NoOpPermissionService) Request(ctx context.Context, tool, action, path string) (bool, error) {
	return true, nil
}

// Check always returns allow always.
func (s *NoOpPermissionService) Check(tool, action string) PermissionLevel {
	return PermissionAllowAlways
}

// CheckCommand always returns allow always.
func (s *NoOpPermissionService) CheckCommand(command string) PermissionLevel {
	return PermissionAllowAlways
}

// CheckPath always returns allow always.
func (s *NoOpPermissionService) CheckPath(path string) PermissionLevel {
	return PermissionAllowAlways
}

// Grant is a no-op.
func (s *NoOpPermissionService) Grant(tool string, level PermissionLevel) {}

// GrantCommand is a no-op.
func (s *NoOpPermissionService) GrantCommand(command string, level PermissionLevel) {}

// GrantPath is a no-op.
func (s *NoOpPermissionService) GrantPath(pattern string, level PermissionLevel) {}

// Revoke is a no-op.
func (s *NoOpPermissionService) Revoke(tool string) {}

// RevokeCommand is a no-op.
func (s *NoOpPermissionService) RevokeCommand(command string) {}

// RevokePath is a no-op.
func (s *NoOpPermissionService) RevokePath(pattern string) {}

// PermissionStore 权限存储接口
type PermissionStore interface {
	SaveDecision(decision PermissionDecision) error
	LoadDecisions() ([]PermissionDecision, error)
	ClearDecisions() error
}
