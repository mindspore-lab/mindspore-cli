package model

import issuepkg "github.com/vigo999/ms-cli/internal/issues"

const (
	IssueIndexOpen  EventType = "IssueIndexOpen"
	IssueDetailOpen EventType = "IssueDetailOpen"
)

type IssueMode int

const (
	IssueModeNone IssueMode = iota
	IssueModeIndex
	IssueModeDetail
)

type IssueIndexState struct {
	Items  []issuepkg.Issue
	Cursor int
	Filter string
	Err    string
}

type IssueDetailState struct {
	ID        int
	Issue     *issuepkg.Issue
	Notes     []issuepkg.Note
	Activity  []issuepkg.Activity
	Err       string
	FromIndex bool
}

type IssueViewState struct {
	Mode   IssueMode
	Index  IssueIndexState
	Detail IssueDetailState
}

func (s IssueViewState) Active() bool {
	return s.Mode != IssueModeNone
}

type IssueEventData struct {
	Filter    string
	Items     []issuepkg.Issue
	ID        int
	Issue     *issuepkg.Issue
	Notes     []issuepkg.Note
	Activity  []issuepkg.Activity
	FromIndex bool
	Err       error
}
