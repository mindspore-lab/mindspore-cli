package permission

import "strings"

// PermissionMode controls top-level permission behavior profile.
type PermissionMode string

const (
	ModeDefault           PermissionMode = "default"
	ModeAcceptEdits       PermissionMode = "acceptEdits"
	ModePlan              PermissionMode = "plan"
	ModeDontAsk           PermissionMode = "dontAsk"
	ModeBypassPermissions PermissionMode = "bypassPermissions"
)

func (m PermissionMode) String() string {
	return string(m)
}

func ParsePermissionMode(s string) PermissionMode {
	switch strings.TrimSpace(strings.ToLower(s)) {
	case "", "default":
		return ModeDefault
	case "acceptedits", "accept_edits":
		return ModeAcceptEdits
	case "plan":
		return ModePlan
	case "dontask", "dont_ask":
		return ModeDontAsk
	case "bypasspermissions", "bypass_permissions":
		return ModeBypassPermissions
	default:
		return ModeDefault
	}
}

func IsValidPermissionMode(mode PermissionMode) bool {
	switch mode {
	case ModeDefault, ModeAcceptEdits, ModePlan, ModeDontAsk, ModeBypassPermissions:
		return true
	default:
		return false
	}
}
