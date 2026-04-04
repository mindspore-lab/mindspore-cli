package model

import "github.com/vigo999/mindspore-cli/internal/bugs"

// Bug-specific UI event types.
const (
	BugIndexOpen  EventType = "BugIndexOpen"
	BugDetailOpen EventType = "BugDetailOpen"
)

type BugMode int

const (
	BugModeNone BugMode = iota
	BugModeIndex
	BugModeDetail
)

type BugIndexState struct {
	Items  []bugs.Bug
	Cursor int
	Filter string
	Err    string
}

type BugDetailState struct {
	ID        int
	Bug       *bugs.Bug
	Activity  []bugs.Activity
	Err       string
	FromIndex bool
}

type BugViewState struct {
	Mode   BugMode
	Index  BugIndexState
	Detail BugDetailState
}

func (s BugViewState) Active() bool {
	return s.Mode != BugModeNone
}

// BugEventData carries bug-specific payloads on Event.
type BugEventData struct {
	Filter    string
	Items     []bugs.Bug
	ID        int
	Bug       *bugs.Bug
	Activity  []bugs.Activity
	FromIndex bool
	Err       error
}
