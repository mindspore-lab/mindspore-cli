package model

import "strings"

// Train-specific UI event types.
const (
	TrainModeOpen      EventType = "TrainModeOpen"
	TrainModeClose     EventType = "TrainModeClose"
	TrainSetup         EventType = "TrainSetup"
	TrainConnect       EventType = "TrainConnect"
	TrainPlanReady     EventType = "TrainPlanReady"
	TrainReady         EventType = "TrainReady"
	TrainStarted       EventType = "TrainStarted"
	TrainIssueDetected EventType = "TrainIssueDetected"
	TrainLogLine       EventType = "TrainLogLine"
	TrainMetric        EventType = "TrainMetric"
	TrainDone          EventType = "TrainDone"
	TrainStopped       EventType = "TrainStopped"
	TrainError         EventType = "TrainError"

	TrainEvalStarted     EventType = "TrainEvalStarted"
	TrainEvalCompleted   EventType = "TrainEvalCompleted"
	TrainDriftDetected   EventType = "TrainDriftDetected"
	TrainAnalyzing       EventType = "TrainAnalyzing"
	TrainAnalysisStarted EventType = "TrainAnalysisStarted"
	TrainAnalysisReady   EventType = "TrainAnalysisReady"
	TrainActionSuggested EventType = "TrainActionSuggested"
	TrainFixApplied      EventType = "TrainFixApplied"
	TrainActionApplied   EventType = "TrainActionApplied"
	TrainRerunStarted    EventType = "TrainRerunStarted"
	TrainVerified        EventType = "TrainVerified"
)

// TrainEventData carries training-specific fields on Event.
type TrainEventData struct {
	RunID      string
	RunLabel   string
	RawInput   string
	Model      string
	Method     string
	Check      string
	Status     string
	Detail     string
	Host       string
	Address    string
	Lane       string
	Step       int
	TotalSteps int
	Loss       float64
	LR         float64
	Throughput float64

	BaselineAcc  float64
	CandidateAcc float64
	Drift        float64

	IssueType    string
	IssueID      string
	IssueTitle   string
	IssueDetail  string
	Confidence   string
	FixSummary   string
	DiffText     string
	ActionID     string
	ActionKind   string
	ActionLabel  string
	ActionSource string
	PlanID       string

	Scope    string
	Critical bool
	IsDiff   bool

	RepoPath     string
	RepoSource   string
	ScriptPath   string
	BaseModelRef string
	ConfigPath   string
	EnvKind      string
	Workdir      string
}

type WorkspaceStage string

const (
	StageSetup     WorkspaceStage = "setup"
	StageReady     WorkspaceStage = "ready"
	StageRunning   WorkspaceStage = "running"
	StageAnalyzing WorkspaceStage = "analyzing"
	StageFixing    WorkspaceStage = "fixing"
	StageDone      WorkspaceStage = "done"
)

type TrainPhase string

const (
	TrainPhaseSetup     TrainPhase = "setup"
	TrainPhaseReady     TrainPhase = "ready"
	TrainPhaseRunning   TrainPhase = "running"
	TrainPhaseFailed    TrainPhase = "failed"
	TrainPhaseCompleted TrainPhase = "completed"

	TrainPhaseEvaluating    TrainPhase = "evaluating"
	TrainPhaseDriftDetected TrainPhase = "drift_detected"
	TrainPhaseAnalyzing     TrainPhase = "analyzing"
	TrainPhaseFixing        TrainPhase = "fixing"
	TrainPhaseStopped       TrainPhase = "stopped"
)

type TrainPanelID string

const (
	TrainPanelRunList TrainPanelID = "run_list"
	TrainPanelStatus  TrainPanelID = "status"
	TrainPanelMetrics TrainPanelID = "metrics"
	TrainPanelIssue   TrainPanelID = "issue"
	TrainPanelActions TrainPanelID = "actions"
	TrainPanelLogs    TrainPanelID = "logs"
	TrainPanelAgent   TrainPanelID = "agent"
	TrainPanelCompare TrainPanelID = "compare"
)

// PanelDisplayState controls train workspace panel presentation.
type PanelDisplayState struct {
	Focused   bool
	Collapsed bool
	// Maximized is reserved for a future single-panel zoom mode.
	Maximized bool
}

type TrainCheckGroup string

const (
	TrainCheckGroupLocal  TrainCheckGroup = "local"
	TrainCheckGroupTarget TrainCheckGroup = "target"
)

type TrainCheckStatus string

const (
	TrainCheckPending TrainCheckStatus = "pending"
	TrainCheckPass    TrainCheckStatus = "pass"
	TrainCheckFail    TrainCheckStatus = "fail"
	TrainCheckRunning TrainCheckStatus = "checking"
)

type ChecklistItem struct {
	Group    TrainCheckGroup
	Name     string
	Status   TrainCheckStatus
	Summary  string
	Critical bool
}

type MetricItem struct {
	Name  string
	Value string
}

type LogBuffer struct {
	Lines      []string
	AutoFollow bool
}

type TrainRequestSummary struct {
	RawInput   string
	Model      string
	Mode       string
	Dataset    string
	TargetName string
	TargetKind string
	Provider   string
}

type SetupContext struct {
	LocalReady   bool
	TargetReady  bool
	RepoPath     string
	ScriptPath   string
	BaseModelRef string
	ConfigPath   string
	EnvKind      string
	Workdir      string
	TargetName   string
}

type TrainPlan struct {
	ID         string
	RunID      string
	Framework  string
	RepoSource string
	ScriptPath string
	BaseModel  string
	ConfigPath string
	EnvKind    string
	Workdir    string
	TargetName string
	Ready      bool
}

type RunConfig struct {
	RunID      string
	Model      string
	Method     string
	Dataset    string
	Framework  string
	Device     string
	TargetName string
	ScriptPath string
	ConfigPath string
}

type TrainAction struct {
	ID      string
	Label   string
	Enabled bool
	Primary bool
}

type TrainActionsState struct {
	Items         []TrainAction
	SelectedIndex int
	InputValue    string
	InputActive   bool
}

// SelectionPopup is a popup menu shown when an action needs user input.
type SelectionPopup struct {
	Title       string
	SearchQuery string
	Options     []SelectionOption
	Selected    int
	SearchBase  int
	ActionID    string // which action triggered the popup
}

type ModelBrowserFocus int

const (
	ModelBrowserFocusProvider ModelBrowserFocus = iota
	ModelBrowserFocusModel
)

type ModelBrowserProviderInput struct {
	Option SelectionOption
	Label  string
	Value  string
	Error  string
}

type ProviderImportSuggestion struct {
	ProviderID    string
	ProviderLabel string
	Source        string
	SourceLabel   string
	Protocol      string
	BaseURL       string
	APIKeyEnvVar  string
	BaseURLEnvVar string
}

type ModelBrowserPopup struct {
	Title                   string
	Providers               SelectionPopup
	Models                  SelectionPopup
	Focus                   ModelBrowserFocus
	ProvidersVisible        bool
	ProviderInput           *ModelBrowserProviderInput
	ImportSuggestions       []ProviderImportSuggestion
	AnimationOffset         int
	PendingDeleteProviderID string
}

type SelectionOption struct {
	ID               string
	Label            string
	Desc             string
	Header           bool
	Separator        bool
	Disabled         bool // grayed out, not selectable (e.g. coming soon)
	RequiresInput    bool
	Featured         bool
	DetailRow        bool
	ProviderRow      bool
	DeleteProviderID string
}

// SetupScreen identifies which screen of the model setup popup is shown.
type SetupScreen int

const (
	SetupScreenModeSelect   SetupScreen = iota // "mscli-provided" vs "your own model"
	SetupScreenPresetPicker                    // pick from preset list
	SetupScreenTokenInput                      // enter token for selected preset
	SetupScreenEnvInfo                         // show env var examples
)

// SetupPopup holds the full state of the multi-step model setup popup.
type SetupPopup struct {
	Title          string
	InputLabel     string
	SearchQuery    string
	Screen         SetupScreen
	ModeSelected   int // 0 = mscli-provided, 1 = your own model
	PresetOptions  []SelectionOption
	PresetSelected int
	SearchBase     int
	SelectedPreset SelectionOption // set when user picks a preset
	TokenValue     string
	TokenError     string // inline error message
	CurrentMode    string // "mscli-provided", "own", or "" — for (current) badge
	CurrentPreset  string // preset ID currently active — for (current) badge
	CanEscape      bool   // false on first boot (no config to fall back to)
	BackCloses     bool
}

// MoveModeSelection moves the mode cursor by delta, wrapping around 2 options.
func (p *SetupPopup) MoveModeSelection(delta int) {
	p.ModeSelected = (p.ModeSelected + delta%2 + 2) % 2
}

// MovePresetSelection moves the preset cursor by delta, wrapping around all options.
// Disabled options are skipped.
func (p *SetupPopup) MovePresetSelection(delta int) {
	p.PresetSelected = moveFilteredSelection(p.PresetSelected, delta, p.PresetOptions, p.SearchQuery)
}

func (p *SetupPopup) FirstSelectablePreset() int {
	return firstSelectableFilteredOption(p.PresetOptions, p.SearchQuery)
}

func (p *SetupPopup) EnsureSelectablePreset() {
	p.PresetSelected = ensureFilteredSelection(p.PresetSelected, p.PresetOptions, p.SearchQuery)
}

func (p *SetupPopup) BeginSearchIfNeeded() {
	if strings.TrimSpace(p.SearchQuery) == "" {
		p.SearchBase = p.PresetSelected
	}
}

func (p *SetupPopup) RestoreSearchBaseIfNeeded() {
	if strings.TrimSpace(p.SearchQuery) == "" && p.SearchBase >= 0 && p.SearchBase < len(p.PresetOptions) {
		p.PresetSelected = p.SearchBase
	}
}

func (p *SelectionPopup) MoveSelection(delta int) {
	p.Selected = moveFilteredSelection(p.Selected, delta, p.Options, p.SearchQuery)
}

func (p *SelectionPopup) EnsureSelectableSelection() {
	p.Selected = ensureFilteredSelection(p.Selected, p.Options, p.SearchQuery)
}

func (p *SelectionPopup) BeginSearchIfNeeded() {
	if strings.TrimSpace(p.SearchQuery) == "" {
		p.SearchBase = p.Selected
	}
}

func (p *SelectionPopup) RestoreSearchBaseIfNeeded() {
	if strings.TrimSpace(p.SearchQuery) == "" && p.SearchBase >= 0 && p.SearchBase < len(p.Options) {
		p.Selected = p.SearchBase
	}
}

func (p *ModelBrowserPopup) HasModels() bool {
	if p == nil {
		return false
	}
	for _, opt := range p.Models.Options {
		if opt.Selectable() {
			return true
		}
	}
	return false
}

func (o SelectionOption) Selectable() bool {
	return !o.Disabled && !o.Header && !o.Separator
}

func filteredOptionIndices(options []SelectionOption, query string) []int {
	if len(options) == 0 {
		return nil
	}
	query = strings.ToLower(strings.TrimSpace(query))
	if query == "" {
		idxs := make([]int, 0, len(options))
		for i := range options {
			idxs = append(idxs, i)
		}
		return idxs
	}

	type group struct {
		header int
		items  []int
	}
	var (
		groups        []group
		currentHeader = -1
		currentItems  []int
	)
	flush := func() {
		if len(currentItems) == 0 {
			return
		}
		items := make([]int, len(currentItems))
		copy(items, currentItems)
		groups = append(groups, group{header: currentHeader, items: items})
		currentItems = nil
	}

	for i, opt := range options {
		if opt.Separator {
			continue
		}
		if opt.Header {
			flush()
			currentHeader = i
			continue
		}
		if !opt.Selectable() {
			continue
		}
		text := strings.ToLower(strings.TrimSpace(opt.Label + " " + opt.Desc))
		if strings.Contains(text, query) {
			currentItems = append(currentItems, i)
		}
	}
	flush()

	result := make([]int, 0)
	for gi, group := range groups {
		if gi > 0 {
			result = append(result, -1)
		}
		if group.header >= 0 {
			result = append(result, group.header)
		}
		result = append(result, group.items...)
	}
	return result
}

func FilteredOptionIndicesForRender(options []SelectionOption, query string) []int {
	return filteredOptionIndices(options, query)
}

func firstSelectableFilteredOption(options []SelectionOption, query string) int {
	for _, idx := range filteredOptionIndices(options, query) {
		if idx >= 0 && options[idx].Selectable() {
			return idx
		}
	}
	return 0
}

func ensureFilteredSelection(current int, options []SelectionOption, query string) int {
	for _, idx := range filteredOptionIndices(options, query) {
		if idx >= 0 && idx == current && options[idx].Selectable() {
			return current
		}
	}
	return firstSelectableFilteredOption(options, query)
}

func moveFilteredSelection(current, delta int, options []SelectionOption, query string) int {
	if delta == 0 {
		return current
	}
	selectable := make([]int, 0)
	for _, idx := range filteredOptionIndices(options, query) {
		if idx >= 0 && options[idx].Selectable() {
			selectable = append(selectable, idx)
		}
	}
	if len(selectable) == 0 {
		return 0
	}
	pos := 0
	for i, idx := range selectable {
		if idx == current {
			pos = i
			break
		}
	}
	return selectable[(pos+delta%len(selectable)+len(selectable))%len(selectable)]
}

type TrainMetricsView struct {
	Step       int
	TotalSteps int
	Loss       float64
	LR         float64
	Throughput float64
}

type TrainPoint struct {
	Step  int
	Value float64
}

type TrainIssueView struct {
	Type       string
	Title      string
	Detail     string
	Confidence string
	FixSummary string
	DiffText   string
}

type IssueKind string

const (
	IssueBootstrap   IssueKind = "bootstrap"
	IssueFailure     IssueKind = "failure"
	IssueAccuracy    IssueKind = "accuracy"
	IssuePerformance IssueKind = "performance"
)

type BootstrapIssueKind string

const (
	BootstrapIssueRepoMissing      BootstrapIssueKind = "repo_missing"
	BootstrapIssueCodeSourceNeeded BootstrapIssueKind = "code_source_needed"
	BootstrapIssueScriptMissing    BootstrapIssueKind = "script_missing"
	BootstrapIssueModelMissing     BootstrapIssueKind = "model_missing"
	BootstrapIssueConfigMissing    BootstrapIssueKind = "config_missing"
	BootstrapIssueEnvMissing       BootstrapIssueKind = "env_missing"
	BootstrapIssueTargetInvalid    BootstrapIssueKind = "target_invalid"
	BootstrapIssueWorkdirInvalid   BootstrapIssueKind = "workdir_invalid"
)

type IssueRecord struct {
	ID        string
	RunID     string
	Kind      IssueKind
	Phase     string
	Summary   string
	Signature map[string]any
	Details   map[string]any
}

type AgentActionKind string

const (
	ActionPrepareEnv   AgentActionKind = "prepare_env"
	ActionDownloadCode AgentActionKind = "download_code"
	ActionScaffoldCode AgentActionKind = "scaffold_code"
	ActionApplyPatch   AgentActionKind = "apply_patch"
	ActionChangeEnv    AgentActionKind = "change_env"
	ActionChangeConfig AgentActionKind = "change_config"
	ActionRerun        AgentActionKind = "rerun"
)

type AgentAction struct {
	ID      string
	RunID   string
	Kind    AgentActionKind
	Label   string
	Source  string
	Payload map[string]any
}

type TrainHostView struct {
	Name    string
	Address string
	Status  string
}

type TrainRunState struct {
	ID         string
	Label      string
	Framework  string
	Device     string
	TargetName string
	CardCount  int
	Role       string

	Phase          TrainPhase
	Ready          bool
	Checks         []ChecklistItem
	Metrics        []MetricItem
	Logs           LogBuffer
	CurrentMetrics TrainMetricsView
	LossSeries     []TrainPoint
	StatusMessage  string
	ErrorMessage   string
	RunLabel       string

	// Per-run issue and action state.
	Issue        *TrainIssueView
	CurrentIssue *IssueRecord
	Issues       []IssueRecord
	AgentActions []AgentAction

	FixApplied bool // true after a fix has been applied (shows "rerun" instead of "start")
}

type CompareViewState struct {
	Enabled      bool
	LeftRunID    string
	RightRunID   string
	Metrics      []MetricItem
	Summary      string
	BaselineAcc  float64
	CandidateAcc float64
	Drift        float64
	Status       string
}

type TrainWorkspaceState struct {
	Active bool

	Request TrainRequestSummary
	Runs    []TrainRunState

	Stage        WorkspaceStage
	SetupContext SetupContext
	TrainPlan    *TrainPlan
	RunConfig    *RunConfig
	ActiveRunID  string
	Compare      *CompareViewState
	Hosts        []TrainHostView

	Panels map[TrainPanelID]*PanelDisplayState
	Focus  TrainPanelID

	GlobalActions  TrainActionsState
	SelectionPopup *SelectionPopup
}

// TrainViewState remains as a compatibility alias while the rest of the UI
// migrates from the old single-session naming to the workspace terminology.
type TrainViewState = TrainWorkspaceState

func NewTrainWorkspaceState() *TrainWorkspaceState {
	s := &TrainWorkspaceState{
		Active: false,
		Stage:  StageSetup,
		Runs: []TrainRunState{
			newTrainRunState(
				"primary",
				"run-1",
				"PyTorch",
				"Ascend",
				"",
				"primary",
			),
		},
		ActiveRunID: "primary",
		Panels: map[TrainPanelID]*PanelDisplayState{
			TrainPanelRunList: {
				Focused:   false,
				Collapsed: false,
				Maximized: false,
			},
			TrainPanelStatus: {
				Focused:   false,
				Collapsed: false,
				Maximized: false,
			},
			TrainPanelMetrics: {
				Focused:   false,
				Collapsed: true,
				Maximized: false,
			},
			TrainPanelIssue: {
				Focused:   false,
				Collapsed: false,
				Maximized: false,
			},
			TrainPanelActions: {
				Focused:   true,
				Collapsed: false,
				Maximized: false,
			},
			TrainPanelLogs: {
				Focused:   false,
				Collapsed: true,
				Maximized: false,
			},
			TrainPanelCompare: {
				Focused:   false,
				Collapsed: true,
				Maximized: false,
			},
		},
		Focus: TrainPanelActions,
		GlobalActions: TrainActionsState{
			Items:         []TrainAction{},
			SelectedIndex: 0,
			InputValue:    "",
			InputActive:   false,
		},
	}
	s.SetFocus(TrainPanelActions)
	s.RefreshActions()
	return s
}

func NewTrainViewState() *TrainWorkspaceState {
	return NewTrainWorkspaceState()
}

func newTrainRunState(id, label, framework, device, targetName, role string) TrainRunState {
	return TrainRunState{
		ID:         id,
		Label:      label,
		Framework:  framework,
		Device:     device,
		TargetName: targetName,
		Role:       role,
		Phase:      TrainPhaseSetup,
		Checks:     []ChecklistItem{},
		Metrics:    []MetricItem{},
		Logs: LogBuffer{
			Lines:      []string{},
			AutoFollow: true,
		},
	}
}

func (s *TrainWorkspaceState) focusOrder() []TrainPanelID {
	return []TrainPanelID{
		TrainPanelRunList,
		TrainPanelStatus,
		TrainPanelMetrics,
		TrainPanelLogs,
		TrainPanelAgent,
		TrainPanelActions,
	}
}

func (s *TrainWorkspaceState) SetFocus(panel TrainPanelID) {
	s.Focus = panel
	for id, st := range s.Panels {
		st.Focused = id == panel
	}
}

func (s *TrainWorkspaceState) FocusNext() {
	order := s.focusOrder()
	idx := 0
	for i, id := range order {
		if id == s.Focus {
			idx = i
			break
		}
	}
	s.SetFocus(order[(idx+1)%len(order)])
}

func (s *TrainWorkspaceState) FocusPrev() {
	order := s.focusOrder()
	idx := 0
	for i, id := range order {
		if id == s.Focus {
			idx = i
			break
		}
	}
	s.SetFocus(order[(idx-1+len(order))%len(order)])
}

func (s *TrainWorkspaceState) ActiveRun() *TrainRunState {
	return s.RunByID(s.ActiveRunID)
}

func (s *TrainWorkspaceState) RunByID(runID string) *TrainRunState {
	for i := range s.Runs {
		if s.Runs[i].ID == runID {
			return &s.Runs[i]
		}
	}
	return nil
}

func (s *TrainWorkspaceState) EnsureRun(runID, label, framework, device, targetName, role string) *TrainRunState {
	if runID == "" {
		runID = "primary"
	}
	run := s.RunByID(runID)
	if run != nil {
		if label != "" {
			run.Label = label
		}
		if framework != "" {
			run.Framework = framework
		}
		if device != "" {
			run.Device = device
		}
		if targetName != "" {
			run.TargetName = targetName
		}
		if role != "" {
			run.Role = role
		}
		return run
	}

	s.Runs = append(s.Runs, newTrainRunState(runID, label, framework, device, targetName, role))
	if s.ActiveRunID == "" {
		s.ActiveRunID = runID
	}
	return &s.Runs[len(s.Runs)-1]
}

func (s *TrainWorkspaceState) SetActiveRun(runID string) {
	run := s.RunByID(runID)
	if run == nil {
		return
	}
	s.ActiveRunID = runID
	s.RefreshActions()
}

func (s *TrainWorkspaceState) ActiveRunIndex() int {
	for i := range s.Runs {
		if s.Runs[i].ID == s.ActiveRunID {
			return i
		}
	}
	return 0
}

func (s *TrainWorkspaceState) SelectNextRun() {
	if len(s.Runs) == 0 {
		return
	}
	idx := s.ActiveRunIndex()
	s.SetActiveRun(s.Runs[(idx+1)%len(s.Runs)].ID)
}

func (s *TrainWorkspaceState) SelectPrevRun() {
	if len(s.Runs) == 0 {
		return
	}
	idx := s.ActiveRunIndex()
	s.SetActiveRun(s.Runs[(idx-1+len(s.Runs))%len(s.Runs)].ID)
}

func (s *TrainWorkspaceState) SetRunPhase(runID string, phase TrainPhase) {
	run := s.EnsureRun(runID, "", "", "", "", "")
	run.Phase = phase

	// Derive stage from phase; SetStage handles panel collapse layout.
	switch phase {
	case TrainPhaseSetup:
		run.Ready = false
		s.SetStage(StageSetup)
	case TrainPhaseReady:
		run.Ready = true
		s.SetStage(StageReady)
	case TrainPhaseRunning, TrainPhaseEvaluating:
		run.StatusMessage = ""
		run.Logs.AutoFollow = true
		s.SetStage(StageRunning)
	case TrainPhaseAnalyzing:
		run.Ready = false
		s.SetStage(StageAnalyzing)
	case TrainPhaseFixing:
		run.Ready = false
		s.SetStage(StageFixing)
	case TrainPhaseStopped:
		run.Ready = false
		s.SetStage(StageDone)
	case TrainPhaseFailed, TrainPhaseDriftDetected:
		run.Ready = false
		s.SetStage(StageRunning)
	case TrainPhaseCompleted:
		run.Logs.AutoFollow = false
		s.SetStage(StageDone)
	}

	s.RefreshActions()
}

func (s *TrainWorkspaceState) SetStage(stage WorkspaceStage) {
	s.Stage = stage
	switch stage {
	case StageSetup, StageReady:
		s.Panels[TrainPanelStatus].Collapsed = false
		s.Panels[TrainPanelMetrics].Collapsed = true
		s.Panels[TrainPanelLogs].Collapsed = true
	case StageRunning:
		s.Panels[TrainPanelStatus].Collapsed = true
		s.Panels[TrainPanelMetrics].Collapsed = false
		s.Panels[TrainPanelLogs].Collapsed = false
	case StageAnalyzing, StageFixing:
		s.Panels[TrainPanelStatus].Collapsed = false
		s.Panels[TrainPanelMetrics].Collapsed = true
		s.Panels[TrainPanelLogs].Collapsed = false
	case StageDone:
		s.Panels[TrainPanelStatus].Collapsed = false
		s.Panels[TrainPanelMetrics].Collapsed = false
		s.Panels[TrainPanelLogs].Collapsed = false
	}
	s.RefreshActions()
}

func (s *TrainWorkspaceState) TogglePanelCollapse(panel TrainPanelID) {
	if panel == TrainPanelActions || panel == TrainPanelCompare {
		return
	}
	state := s.Panels[panel]
	if state == nil {
		return
	}
	state.Collapsed = !state.Collapsed
	if state.Collapsed && state.Maximized {
		state.Maximized = false
	}
}

func (s *TrainWorkspaceState) TogglePanelMaximize(panel TrainPanelID) {
	state := s.Panels[panel]
	if state == nil {
		return
	}
	next := !state.Maximized
	for id, panelState := range s.Panels {
		if id == panel {
			panelState.Maximized = next
			if next {
				panelState.Collapsed = false
			}
			continue
		}
		panelState.Maximized = false
	}
}

func (s *TrainWorkspaceState) MaximizedPanel() (TrainPanelID, bool) {
	for id, state := range s.Panels {
		if state != nil && state.Maximized {
			return id, true
		}
	}
	return "", false
}

func (s *TrainWorkspaceState) UpsertCheck(runID string, item ChecklistItem) {
	run := s.EnsureRun(runID, "", "", "", "", "")
	for i := range run.Checks {
		if run.Checks[i].Group == item.Group && run.Checks[i].Name == item.Name {
			run.Checks[i] = item
			return
		}
	}
	run.Checks = append(run.Checks, item)
}

func (s *TrainWorkspaceState) AppendLog(runID, line string) {
	run := s.EnsureRun(runID, "", "", "", "", "")
	run.Logs.Lines = append(run.Logs.Lines, line)
	const maxLogs = 500
	if len(run.Logs.Lines) > maxLogs {
		run.Logs.Lines = run.Logs.Lines[len(run.Logs.Lines)-maxLogs:]
	}
}

func (s *TrainWorkspaceState) UpsertMetric(runID, name, value string) {
	run := s.EnsureRun(runID, "", "", "", "", "")
	for i := range run.Metrics {
		if run.Metrics[i].Name == name {
			run.Metrics[i].Value = value
			return
		}
	}
	run.Metrics = append(run.Metrics, MetricItem{Name: name, Value: value})
}

func (s *TrainWorkspaceState) ChecksByGroup(runID string, group TrainCheckGroup) []ChecklistItem {
	run := s.RunByID(runID)
	if run == nil {
		return nil
	}
	items := make([]ChecklistItem, 0)
	for _, item := range run.Checks {
		if item.Group == group {
			items = append(items, item)
		}
	}
	return items
}

func (s *TrainWorkspaceState) RefreshActions() {
	run := s.ActiveRun()
	if run != nil && len(run.AgentActions) > 0 {
		items := make([]TrainAction, 0, len(run.AgentActions)+1)
		for i, action := range run.AgentActions {
			items = append(items, TrainAction{
				ID:      action.ID,
				Label:   action.Label,
				Enabled: true,
				Primary: i == 0,
			})
		}
		s.GlobalActions.Items = items
		if s.GlobalActions.SelectedIndex >= len(s.GlobalActions.Items) {
			s.GlobalActions.SelectedIndex = 0
		}
		return
	}

	phase := TrainPhaseSetup
	if run := s.ActiveRun(); run != nil {
		phase = run.Phase
	}

	switch phase {
	case TrainPhaseSetup:
		s.GlobalActions.Items = []TrainAction{}
	case TrainPhaseReady:
		label := "start"
		if run := s.ActiveRun(); run != nil && run.FixApplied {
			label = "rerun"
		}
		s.GlobalActions.Items = []TrainAction{
			{ID: "start", Label: label, Enabled: true, Primary: true},
		}
	case TrainPhaseRunning, TrainPhaseEvaluating:
		s.GlobalActions.Items = []TrainAction{
			{ID: "stop", Label: "stop", Enabled: true, Primary: true},
		}
	case TrainPhaseFailed:
		s.GlobalActions.Items = []TrainAction{
			{ID: "retry", Label: "retry", Enabled: true, Primary: true},
			{ID: "diagnose", Label: "diagnose", Enabled: true},
		}
	case TrainPhaseDriftDetected:
		s.GlobalActions.Items = []TrainAction{
			{ID: "diagnose", Label: "diagnose", Enabled: true, Primary: true},
		}
	case TrainPhaseAnalyzing, TrainPhaseFixing:
		s.GlobalActions.Items = []TrainAction{
			{ID: "stop", Label: "stop", Enabled: true, Primary: true},
		}
	case TrainPhaseCompleted:
		items := []TrainAction{
			{ID: "rerun", Label: "rerun", Enabled: true, Primary: true},
			{ID: "analyze_perf", Label: "analyze perf", Enabled: true},
			{ID: "add_algo_feature", Label: "algo-feature", Enabled: true},
			{ID: "add_perf_feature", Label: "perf-feature", Enabled: true},
		}
		s.GlobalActions.Items = items
	default:
		s.GlobalActions.Items = []TrainAction{}
	}

	if s.GlobalActions.SelectedIndex >= len(s.GlobalActions.Items) {
		s.GlobalActions.SelectedIndex = 0
	}
}

func (s *TrainWorkspaceState) SetIssue(issue IssueRecord) {
	run := s.EnsureRun(issue.RunID, "", "", "", "", "")
	run.CurrentIssue = &issue
	found := false
	for i := range run.Issues {
		if run.Issues[i].ID == issue.ID {
			run.Issues[i] = issue
			found = true
			break
		}
	}
	if !found {
		run.Issues = append(run.Issues, issue)
	}
}

func (s *TrainWorkspaceState) SetAgentActions(runID string, actions []AgentAction) {
	run := s.EnsureRun(runID, "", "", "", "", "")
	run.AgentActions = actions
	s.RefreshActions()
}
