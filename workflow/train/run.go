package train

import (
	"context"
)

// RunTraining orchestrates the running phase of training.
// It delegates to the backend for actual log/metric streaming.
func RunTraining(ctx context.Context, session *Session, backend Backend, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, ev) }

	if !e(Event{Kind: EventTrainStarted, RunID: session.ID, Message: "Training started.", DelayMs: 200}) {
		return ctx.Err()
	}

	err := backend.Run(ctx, session, sink)
	if err != nil {
		if ctx.Err() != nil {
			e(Event{Kind: EventTrainStopped, Message: "Training stopped."})
			return ctx.Err()
		}
		e(Event{
			Kind:        EventTrainFailed,
			RunID:       session.ID,
			IssueType:   "failure",
			IssueID:     "train-failure-" + session.ID,
			IssueTitle:  err.Error(),
			IssueDetail: err.Error(),
			Message:     err.Error(),
		})
		return err
	}

	return nil
}
