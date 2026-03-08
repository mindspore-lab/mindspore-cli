package ui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/vigo999/ms-cli/ui/components"
	"github.com/vigo999/ms-cli/ui/model"
	"github.com/vigo999/ms-cli/ui/slash"
)

func TestHandleTrainKeyRetryOnFailedDashboard(t *testing.T) {
	userCh := make(chan string, 1)
	train := model.NewTrainDashboard()
	train.Status = "failed"
	app := App{
		viewMode: viewModeTrain,
		train:    train,
		userCh:   userCh,
	}

	_, _ = app.handleTrainKey(tea.KeyMsg{Type: tea.KeyCtrlR})

	select {
	case input := <-userCh:
		if input != "/train retry" {
			t.Fatalf("expected retry command, got %q", input)
		}
	default:
		t.Fatalf("expected retry command to be sent")
	}
}

func TestHandleTrainKeyRetryIgnoredWhenNotFailed(t *testing.T) {
	userCh := make(chan string, 1)
	train := model.NewTrainDashboard()
	train.Status = "running"
	app := App{
		viewMode: viewModeTrain,
		train:    train,
		userCh:   userCh,
	}

	_, _ = app.handleTrainKey(tea.KeyMsg{Type: tea.KeyCtrlR})

	select {
	case input := <-userCh:
		t.Fatalf("did not expect retry command, got %q", input)
	default:
	}
}

func TestRenderTrainViewDoesNotInsertChatDivider(t *testing.T) {
	train := model.NewTrainDashboard()
	train.Status = "running"

	app := App{
		viewMode: viewModeTrain,
		train:    train,
		width:    80,
		height:   24,
	}

	rendered := app.renderTrainView()
	if strings.Contains(rendered, "\n"+strings.Repeat("─", app.width)+"\n") {
		t.Fatalf("did not expect chat divider line in train view")
	}
}

func TestHandleTrainChatEnterRoutesFreeFormInputWithPrefix(t *testing.T) {
	userCh := make(chan string, 1)
	train := model.NewTrainDashboard()
	train.Status = "success"
	train.CurrentStage = "dashboard"
	train.StageStatus["dashboard"] = model.TrainStageSuccess

	app := App{
		viewMode:      viewModeTrain,
		train:         train,
		userCh:        userCh,
		trainSlash:    slash.DefaultRegistry.Without("/train"),
		width:         120,
		height:        36,
		viewport:      components.NewViewport(1, 1),
		trainViewport: components.NewViewport(1, 1),
		input:         components.NewTextInput(),
	}
	app.input.Model.SetValue("analyze the loss trend and rerun if needed")
	app.syncInputMode()
	app.updateViewport()

	next, _ := app.handleTrainKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	select {
	case input := <-userCh:
		want := model.TrainChatInputPrefix + "analyze the loss trend and rerun if needed"
		if input != want {
			t.Fatalf("expected train-chat prefixed input %q, got %q", want, input)
		}
	default:
		t.Fatalf("expected train-chat input to be sent")
	}
}

func TestHandleTrainChatBlocksTrainSlashCommand(t *testing.T) {
	userCh := make(chan string, 1)
	train := model.NewTrainDashboard()
	train.Status = "success"
	train.CurrentStage = "dashboard"
	train.StageStatus["dashboard"] = model.TrainStageSuccess

	app := App{
		viewMode:      viewModeTrain,
		train:         train,
		userCh:        userCh,
		trainSlash:    slash.DefaultRegistry.Without("/train"),
		width:         120,
		height:        36,
		viewport:      components.NewViewport(1, 1),
		trainViewport: components.NewViewport(1, 1),
		input:         components.NewTextInput(),
	}
	app.input.Model.SetValue("/train retry")
	app.syncInputMode()
	app.updateViewport()

	next, _ := app.handleTrainKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	select {
	case input := <-userCh:
		t.Fatalf("did not expect /train slash command to be forwarded, got %q", input)
	default:
	}
	if len(app.state.Messages) == 0 {
		t.Fatalf("expected a local agent warning message")
	}
	if !strings.Contains(app.state.Messages[len(app.state.Messages)-1].Content, "does not accept `/train`") {
		t.Fatalf("expected /train warning message, got %#v", app.state.Messages[len(app.state.Messages)-1])
	}
}

func TestRenderTrainViewShowsAnalysisChatAfterDashboardStage(t *testing.T) {
	train := model.NewTrainDashboard()
	train.Status = "success"
	train.CurrentStage = "dashboard"
	train.StageStatus["dashboard"] = model.TrainStageSuccess

	app := App{
		viewMode:      viewModeTrain,
		train:         train,
		trainSlash:    slash.DefaultRegistry.Without("/train"),
		width:         120,
		height:        36,
		viewport:      components.NewViewport(1, 1),
		trainViewport: components.NewViewport(1, 1),
		input:         components.NewTextInput(),
	}
	app.syncInputMode()
	app.updateViewport()

	rendered := app.renderTrainView()
	if !strings.Contains(rendered, "Analysis Chat") {
		t.Fatalf("expected embedded analysis chat panel, got %q", rendered)
	}
}

func TestRenderTrainViewDefersAnalysisChatUntilDashboardStage(t *testing.T) {
	train := model.NewTrainDashboard()
	train.Active = true
	train.Status = "running"
	train.CurrentStage = "launch"
	train.StageStatus["launch"] = model.TrainStageRunning

	app := App{
		viewMode:      viewModeTrain,
		train:         train,
		trainSlash:    slash.DefaultRegistry.Without("/train"),
		width:         120,
		height:        36,
		viewport:      components.NewViewport(1, 1),
		trainViewport: components.NewViewport(1, 1),
		input:         components.NewTextInput(),
	}
	app.syncInputMode()
	app.updateViewport()

	rendered := app.renderTrainView()
	if strings.Contains(rendered, "Analysis Chat") {
		t.Fatalf("did not expect analysis chat panel before dashboard stage, got %q", rendered)
	}
	if !strings.Contains(rendered, "Connection Logs") {
		t.Fatalf("expected right panel to keep connection logs during launch stage, got %q", rendered)
	}
}

func TestTrainViewportHidesToolLogsAndPreTrainMessages(t *testing.T) {
	train := model.NewTrainDashboard()
	app := App{
		state: model.State{
			Messages: []model.Message{
				{Kind: model.MsgUser, Content: "/train test with examples/train.py"},
				{Kind: model.MsgTool, Content: "pre-train tool output"},
			},
		},
		viewMode:      viewModeChat,
		train:         train,
		trainSlash:    slash.DefaultRegistry.Without("/train"),
		width:         120,
		height:        36,
		viewport:      components.NewViewport(1, 1),
		trainViewport: components.NewViewport(1, 1),
		input:         components.NewTextInput(),
	}

	open := model.TrainUpdate{
		Kind:  model.TrainUpdateOpen,
		RunID: "run-001",
		Hosts: []string{"gpuA"},
	}
	next, _ := app.handleEvent(model.Event{Type: model.TrainUpdateEvent, Train: &open})
	app = next.(App)
	dashboard := model.TrainUpdate{
		Kind:   model.TrainUpdateStage,
		Stage:  "dashboard",
		Status: string(model.TrainStageRunning),
	}
	next, _ = app.handleEvent(model.Event{Type: model.TrainUpdateEvent, Train: &dashboard})
	app = next.(App)
	app.state = app.state.WithMessage(model.Message{Kind: model.MsgUser, Content: "analyze this run"})
	app.state = app.state.WithMessage(model.Message{Kind: model.MsgTool, Content: "$ python inspect.py\nstep logs"})
	app.state = app.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: "Loss is trending down."})
	app.syncInputMode()
	app.updateViewport()

	rendered := app.renderTrainView()
	if strings.Contains(rendered, "pre-train tool output") {
		t.Fatalf("did not expect pre-train content in embedded chat, got %q", rendered)
	}
	if strings.Contains(rendered, "/train test with examples/train.py") {
		t.Fatalf("did not expect triggering /train command in embedded chat, got %q", rendered)
	}
	if strings.Contains(rendered, "$ python inspect.py") || strings.Contains(rendered, "step logs") {
		t.Fatalf("did not expect tool output or logs in embedded chat, got %q", rendered)
	}
	if !strings.Contains(rendered, "analyze this run") || !strings.Contains(rendered, "Loss is trending down.") {
		t.Fatalf("expected embedded chat to keep conversational messages only, got %q", rendered)
	}
}
