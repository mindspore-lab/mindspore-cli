package app

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/vigo999/ms-cli/permission"
)

type scopedPermissionSettingsFile struct {
	Permissions scopedPermissionBuckets `json:"permissions"`
}

type scopedPermissionBuckets struct {
	Allow []string `json:"allow,omitempty"`
	Ask   []string `json:"ask,omitempty"`
	Deny  []string `json:"deny,omitempty"`
}

type scopedPermissionFileSpec struct {
	path string
}

func parsePermissionScopeArgs(args []string) (scope string, hasScope bool, rest []string) {
	scope = "project"
	if len(args) == 0 {
		return scope, false, nil
	}
	rest = make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		token := strings.TrimSpace(args[i])
		if token == "--scope" && i+1 < len(args) {
			scope = strings.ToLower(strings.TrimSpace(args[i+1]))
			hasScope = true
			i++
			continue
		}
		rest = append(rest, args[i])
	}
	return scope, hasScope, rest
}

func (a *Application) savePermissionRuleToScope(rule string, level permission.PermissionLevel, scope string) (string, error) {
	path, err := resolvePermissionScopePath(a.WorkDir, scope)
	if err != nil {
		return "", err
	}
	cfg, err := readScopedPermissionSettings(path)
	if err != nil {
		return "", err
	}
	bucket := permissionBucketByLevel(level)
	cfg.Permissions.Allow = removeRule(cfg.Permissions.Allow, rule)
	cfg.Permissions.Ask = removeRule(cfg.Permissions.Ask, rule)
	cfg.Permissions.Deny = removeRule(cfg.Permissions.Deny, rule)
	switch bucket {
	case "deny":
		cfg.Permissions.Deny = append(cfg.Permissions.Deny, rule)
	case "ask":
		cfg.Permissions.Ask = append(cfg.Permissions.Ask, rule)
	default:
		cfg.Permissions.Allow = append(cfg.Permissions.Allow, rule)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return "", fmt.Errorf("create settings dir: %w", err)
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return "", fmt.Errorf("marshal settings: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return "", fmt.Errorf("write settings: %w", err)
	}
	return path, nil
}

func resolvePermissionScopePath(workDir, scope string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(scope)) {
	case "", "project":
		return filepath.Join(workDir, ".ms-cli", "permissions.json"), nil
	case "user":
		home, err := os.UserHomeDir()
		if err != nil || strings.TrimSpace(home) == "" {
			return "", fmt.Errorf("resolve user settings path: %w", err)
		}
		return filepath.Join(home, ".ms-cli", "permissions.json"), nil
	default:
		return "", fmt.Errorf("invalid scope %q", scope)
	}
}

func readScopedPermissionSettings(path string) (scopedPermissionSettingsFile, error) {
	out := scopedPermissionSettingsFile{
		Permissions: scopedPermissionBuckets{
			Allow: []string{},
			Ask:   []string{},
			Deny:  []string{},
		},
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return out, nil
		}
		return out, fmt.Errorf("read settings: %w", err)
	}
	trimmed := strings.TrimSpace(string(data))
	if trimmed == "" {
		return out, nil
	}
	if strings.HasPrefix(trimmed, "[") {
		return out, fmt.Errorf("invalid settings format: expected object with permissions buckets, got legacy array")
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return out, fmt.Errorf("invalid settings JSON: %w", err)
	}
	for i, rule := range out.Permissions.Allow {
		if _, err := permission.ParsePermissionRule(rule); err != nil {
			return out, fmt.Errorf("permissions.allow[%d]: %w", i, err)
		}
	}
	for i, rule := range out.Permissions.Ask {
		if _, err := permission.ParsePermissionRule(rule); err != nil {
			return out, fmt.Errorf("permissions.ask[%d]: %w", i, err)
		}
	}
	for i, rule := range out.Permissions.Deny {
		if _, err := permission.ParsePermissionRule(rule); err != nil {
			return out, fmt.Errorf("permissions.deny[%d]: %w", i, err)
		}
	}
	return out, nil
}

func permissionBucketByLevel(level permission.PermissionLevel) string {
	switch level {
	case permission.PermissionDeny:
		return "deny"
	case permission.PermissionAsk:
		return "ask"
	default:
		return "allow"
	}
}

func removeRule(in []string, rule string) []string {
	rule = strings.TrimSpace(rule)
	if rule == "" {
		return in
	}
	out := in[:0]
	for _, item := range in {
		if strings.EqualFold(strings.TrimSpace(item), rule) {
			continue
		}
		out = append(out, item)
	}
	return out
}

func scopedPermissionFiles(workDir string) []scopedPermissionFileSpec {
	files := make([]scopedPermissionFileSpec, 0, 2)
	home, _ := os.UserHomeDir()
	if strings.TrimSpace(home) != "" {
		files = append(files, scopedPermissionFileSpec{path: filepath.Join(home, ".ms-cli", "permissions.json")})
	}
	files = append(files, scopedPermissionFileSpec{path: filepath.Join(workDir, ".ms-cli", "permissions.json")})
	return files
}

func preloadScopedPermissionRules(permSvc *permission.DefaultPermissionService, workDir string) *permissionSettingsIssue {
	if permSvc == nil {
		return nil
	}
	for _, spec := range scopedPermissionFiles(workDir) {
		cfg, err := readScopedPermissionSettings(spec.path)
		if err != nil {
			return &permissionSettingsIssue{
				FilePath: spec.path,
				Detail:   err.Error(),
			}
		}
		for _, rule := range cfg.Permissions.Allow {
			if err := permSvc.AddRule(rule, permission.PermissionAllowAlways); err != nil {
				return &permissionSettingsIssue{
					FilePath: spec.path,
					Detail:   err.Error(),
				}
			}
		}
		for _, rule := range cfg.Permissions.Ask {
			if err := permSvc.AddRule(rule, permission.PermissionAsk); err != nil {
				return &permissionSettingsIssue{
					FilePath: spec.path,
					Detail:   err.Error(),
				}
			}
		}
		for _, rule := range cfg.Permissions.Deny {
			if err := permSvc.AddRule(rule, permission.PermissionDeny); err != nil {
				return &permissionSettingsIssue{
					FilePath: spec.path,
					Detail:   err.Error(),
				}
			}
		}
	}
	return nil
}
