package app

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/mindspore-lab/mindspore-cli/agent/session"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/permission"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

type sessionPickerRequest struct {
	Mode        model.SessionPickerMode
	ReplaySpeed float64
}

type sessionSwitchOptions struct {
	Target      string
	Replay      bool
	ReplaySpeed float64
}

type loadedConversation struct {
	runtimeSession *session.Session
	systemPrompt   string
	messages       []llm.Message
	usageSnapshot  *session.UsageSnapshot
	replayBacklog  []model.Event
	replayTimeline []session.ReplayFrame
}

func (a *Application) cmdResume(args []string) {
	if len(args) == 0 {
		a.openSessionPicker(model.SessionPickerResume, 0)
		return
	}
	if len(args) != 1 {
		a.emitToolError("session", "usage: /resume [sess_xxx]")
		return
	}
	if looksLikeTrajectoryPath(args[0]) {
		a.emitToolError("session", "usage: /resume [sess_xxx]")
		return
	}
	a.switchConversation(sessionSwitchOptions{
		Target: strings.TrimSpace(args[0]),
	})
}

func (a *Application) cmdReplay(args []string) {
	target, speed, err := parseReplayTargetAndSpeed(args)
	if err != nil {
		a.emitToolError("session", "%v", err)
		return
	}
	if strings.TrimSpace(target) == "" {
		a.openSessionPicker(model.SessionPickerReplay, speed)
		return
	}
	a.switchConversation(sessionSwitchOptions{
		Target:      target,
		Replay:      true,
		ReplaySpeed: speed,
	})
}

func (a *Application) openSessionPicker(mode model.SessionPickerMode, replaySpeed float64) {
	if a == nil || a.EventCh == nil {
		return
	}

	summaries, err := session.ListForWorkDir(a.WorkDir)
	if err != nil {
		a.emitToolError("session", "Failed to list saved sessions: %v", err)
		return
	}

	items := make([]model.SessionPickerItem, 0, len(summaries))
	for _, summary := range summaries {
		items = append(items, model.SessionPickerItem{
			ID:             summary.SessionID,
			CreatedAt:      summary.CreatedAt,
			UpdatedAt:      summary.UpdatedAt,
			FirstUserInput: summary.FirstUserInput,
		})
	}

	a.EventCh <- model.Event{
		Type: model.SessionPickerOpen,
		SessionPicker: &model.SessionPicker{
			Mode:         mode,
			Items:        items,
			ReplaySpeed:  replaySpeedOrDefault(replaySpeed),
			EmptyMessage: "No saved sessions found for this workdir.",
		},
	}
}

func (a *Application) switchConversation(opts sessionSwitchOptions) {
	if a == nil {
		return
	}

	target := strings.TrimSpace(opts.Target)
	if target == "" {
		a.emitToolError("session", "session target cannot be empty")
		return
	}

	resumeHint := ""
	if a.session != nil && a.sessionLLMActivity.Load() {
		resumeHint = inlineResumeHintForSession(a.session.ID())
	}
	if err := a.persistSessionSnapshot(); err != nil {
		a.emitToolError("session", "Failed to preserve the current conversation: %v", err)
		return
	}
	shouldClearScreen := a.shouldClearScreenOnSessionSwitch()

	a.interruptReplay()
	a.interruptActiveTasks()

	loaded, err := a.loadConversation(target, opts.Replay)
	if err != nil {
		a.emitToolError("session", "Failed to load %s: %v", target, err)
		return
	}

	previous := a.session
	a.bindConversation(loaded, opts)
	a.startupSessionPicker = nil
	if previous != nil && previous != a.session {
		_ = previous.Close()
	}

	if shouldClearScreen {
		a.EventCh <- model.Event{
			Type:    model.ClearScreen,
			Message: "Conversation switched.",
			Summary: resumeHint,
		}
	}
	a.startReplayHistory()
}

func (a *Application) shouldClearScreenOnSessionSwitch() bool {
	if a == nil {
		return false
	}
	if a.sessionLLMActivity.Load() {
		return true
	}
	if a.ctxManager == nil {
		return false
	}
	return len(a.ctxManager.GetNonSystemMessages()) > 0
}

func (a *Application) loadConversation(target string, replay bool) (*loadedConversation, error) {
	if looksLikeTrajectoryPath(target) {
		return a.loadReplayPathConversation(target)
	}

	runtimeSession, err := session.LoadByID(a.WorkDir, target)
	if err != nil {
		return nil, err
	}

	systemPrompt, messages := runtimeSession.RestoreContext()
	loaded := &loadedConversation{
		runtimeSession: runtimeSession,
		systemPrompt:   systemPrompt,
		messages:       messages,
		usageSnapshot:  runtimeSession.UsageSnapshot(),
	}
	if replay {
		loaded.replayTimeline = runtimeSession.PlaybackTimeline()
	} else {
		loaded.replayBacklog = runtimeSession.ReplayEvents()
	}
	return loaded, nil
}

func (a *Application) loadReplayPathConversation(target string) (*loadedConversation, error) {
	source, err := session.LoadReplayPath(target)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = source.Close()
	}()

	metaWorkDir := strings.TrimSpace(source.Meta().WorkDir)
	if metaWorkDir != "" && !sameWorkDir(metaWorkDir, a.WorkDir) {
		return nil, fmt.Errorf("replaying a trajectory from another workdir is not supported inside the running app")
	}

	systemPrompt, messages := source.RestoreContext()
	runtimeSession, err := session.Create(a.WorkDir, systemPrompt)
	if err != nil {
		return nil, err
	}

	return &loadedConversation{
		runtimeSession: runtimeSession,
		systemPrompt:   systemPrompt,
		messages:       messages,
		usageSnapshot:  source.UsageSnapshot(),
		replayTimeline: source.PlaybackTimeline(),
	}, nil
}

func (a *Application) bindConversation(loaded *loadedConversation, opts sessionSwitchOptions) {
	if a == nil || loaded == nil {
		return
	}

	a.session = loaded.runtimeSession
	a.replayBacklog = loaded.replayBacklog
	a.replayTimeline = loaded.replayTimeline
	a.deferHistoryReplay = false
	a.historyReplayStarted.Store(false)
	a.replayOnly = opts.Replay
	a.replaySpeed = replaySpeedOrDefault(opts.ReplaySpeed)
	a.sessionLLMActivity.Store(false)
	a.sessionStoreReady.Store(false)

	if a.ctxManager != nil {
		a.ctxManager.SetSystemPrompt(loaded.systemPrompt)
		a.ctxManager.SetNonSystemMessages(loaded.messages)
		restoreProviderUsageSnapshot(a.ctxManager, loaded.usageSnapshot)
	}

	if permSvc, ok := a.permService.(*permission.DefaultPermissionService); ok {
		permSvc.ResetSessionState()
	}
	a.loadSessionPermissionStore()

	if a.llmDebugDumper != nil && a.session != nil {
		a.llmDebugDumper = llm.NewDebugDumper(filepath.Dir(a.session.Path()))
	}
	a.refreshEngineSessionBindings()
}

func (a *Application) loadSessionPermissionStore() {
	if a == nil || a.permService == nil || a.session == nil {
		return
	}

	storeSetter, ok := a.permService.(interface {
		SetStore(permission.PermissionStore)
	})
	if !ok {
		return
	}

	storeCfg := sessionPermissionStoreConfig(a.session)
	store, err := permission.NewPermissionStore(storeCfg)
	if err != nil {
		if a.permissionSettingsIssue == nil {
			a.permissionSettingsIssue = &permissionSettingsIssue{
				FilePath: normalizePermissionSettingsPath(storeCfg.Path, a.WorkDir),
				Detail:   err.Error(),
			}
		}
		return
	}

	storeSetter.SetStore(store)
	a.sessionStoreReady.Store(true)
}

func sameWorkDir(aPath, bPath string) bool {
	aAbs, aErr := filepath.Abs(aPath)
	bAbs, bErr := filepath.Abs(bPath)
	if aErr != nil || bErr != nil {
		return filepath.Clean(aPath) == filepath.Clean(bPath)
	}
	return filepath.Clean(aAbs) == filepath.Clean(bAbs)
}
