package permission

import (
	"strings"
	"time"
)

// PermissionLevel represents the permission level for a tool.
type PermissionLevel int

const (
	// PermissionDeny always denies the tool.
	PermissionDeny PermissionLevel = iota
	// PermissionAsk asks user for permission each time.
	PermissionAsk
	// PermissionAllowOnce allows once without asking.
	PermissionAllowOnce
	// PermissionAllowSession allows for the current session.
	PermissionAllowSession
	// PermissionAllowAlways always allows.
	PermissionAllowAlways
)

// String returns the string representation.
func (p PermissionLevel) String() string {
	switch p {
	case PermissionDeny:
		return "deny"
	case PermissionAsk:
		return "ask"
	case PermissionAllowOnce:
		return "allow_once"
	case PermissionAllowSession:
		return "allow_session"
	case PermissionAllowAlways:
		return "allow_always"
	default:
		return "unknown"
	}
}

// ParsePermissionLevel parses a permission level string.
func ParsePermissionLevel(s string) PermissionLevel {
	switch strings.ToLower(s) {
	case "deny":
		return PermissionDeny
	case "ask":
		return PermissionAsk
	case "allow_once":
		return PermissionAllowOnce
	case "allow_session":
		return PermissionAllowSession
	case "allow_always", "allow":
		return PermissionAllowAlways
	default:
		return PermissionAsk
	}
}

// PermissionDecision records a persisted decision.
type PermissionDecision struct {
	Tool      string
	Action    string
	Path      string
	Level     PermissionLevel
	Timestamp time.Time
}
