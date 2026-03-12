// Package probes defines a unified probe result model for readiness checks.
package probes

// Status represents the outcome of a probe.
type Status string

const (
	StatusPending Status = "pending"
	StatusPass    Status = "pass"
	StatusFail    Status = "fail"
)

// Scope identifies whether a probe checks local or target state.
type Scope string

const (
	ScopeLocal  Scope = "local"
	ScopeTarget Scope = "target"
)

// Result is the outcome of a single probe check.
type Result struct {
	Scope    Scope          `json:"scope"`
	Name     string         `json:"name"`
	Status   Status         `json:"status"`
	Summary  string         `json:"summary"`
	Critical bool           `json:"critical"`
	Details  map[string]any `json:"details,omitempty"`
}
