# How To Provide Plan Proposals Without a Separate Planner

## Goal

Provide a user-visible plan before execution when needed, without keeping a heavyweight `planner` package or a separate plan-and-execute architecture.

The core idea is:

- the agent loop remains the main execution model
- a plan is an optional pre-execution proposal
- the proposal can be shown to the user for confirmation
- after approval, the normal agent loop continues

This keeps the runtime simple while still supporting reviewable plans.

## When This Is Useful

This pattern is useful when:

- a task is complex enough that the user may want to review the approach first
- the task may be expensive, risky, or time-consuming
- the UI should show a short step list before execution
- the system needs a lightweight approval step

This is not the same as a full planner subsystem. It is just a small preflight capability on top of the normal agent loop.

## Design Principle

Do not model planning as a separate execution mode.

Instead:

1. the user submits a task
2. the app runtime optionally asks the model for a short plan proposal
3. the plan is returned to the app and UI
4. the user approves or revises the task
5. the normal agent loop executes the task

So the architecture stays:

```text
user input -> app runtime -> agent loop -> tools -> result
```

And when a plan is required:

```text
user input -> app runtime -> plan proposal -> user approval -> agent loop
```

## Recommended Runtime Types

Add a lightweight plan proposal type in the app runtime layer (for example `internal/app`).

```go
type PlanProposal struct {
    Goal          string
    Steps         []string
    Assumptions   []string
    NeedsApproval bool
}
```

Extend the run request with an approval hint:

```go
type PlanRequest struct {
	ID                  string
	Description         string
	RequirePlanApproval bool
}
```

This is intentionally small. It is enough to support plan display and approval without creating a full workflow planning layer.

## Recommended Event Model

The app runtime should emit plan-related events, not execute tools immediately when approval is required.

Suggested event types:

```go
const (
    EventPlanProposed = "PlanProposed"
    EventPlanApproved = "PlanApproved"
)
```

Suggested plan event payload:

```go
type PlanEventData struct {
    Goal        string
    Steps       []string
    Assumptions []string
}
```

This allows the UI to render plans separately from regular chat messages if needed.

## Runtime Behavior

The app runtime should support two paths.

### Normal path

If `RequirePlanApproval` is `false`:

- run the agent loop immediately
- do not create a separate plan artifact

### Plan proposal path

If `RequirePlanApproval` is `true`:

- call a small `ProposePlan(...)` helper
- return a `PlanProposal`
- emit a `PlanProposed` event
- do not start tool execution yet

After the user approves:

- the app resubmits the task for normal execution
- the agent loop runs as usual

This means there is still only one real executor: the agent loop.

## Suggested Runtime API

Keep the API minimal.

```go
func (a *Application) ProposePlan(ctx context.Context, req PlanRequest) (*PlanProposal, error)
func (a *Application) RunTask(ctx context.Context, req PlanRequest) ([]model.Event, error)
```

Expected behavior:

- `ProposePlan(...)` generates a short plan only
- `Run(...)` executes the task

Do not mix planning state, workflow graphs, or step scheduling into this layer in Phase 1.

## How The Model Should Produce The Plan

Do not use a large planning DSL.

Use a short constrained format, for example:

```text
GOAL:
- ...

STEPS:
1. ...
2. ...
3. ...

ASSUMPTIONS:
- ...
```

This is enough for:

- simple parsing
- stable rendering
- user review

If stronger structure is needed later, this can evolve into JSON, but Phase 1 should keep it lightweight.

## App Layer Integration

The app layer should own pending approval state.

Suggested shape:

```go
type PendingPlan struct {
	TaskID   string
	Request  PlanRequest
	Proposal *PlanProposal
}
```

Recommended flow:

1. user enters a task that requires approval
2. app calls `ProposePlan(...)`
3. app stores `PendingPlan`
4. UI shows the plan
5. user enters `approve`, `run`, or an edit instruction
6. app replays the original request through `Run(...)`

This keeps approval logic in the app layer rather than the agent loop.

## UI Integration

The UI can start with a simple chat-style rendering:

- show the plan as a formatted agent reply
- wait for user confirmation

If needed later, the same data can power a dedicated plan panel or approval widget.

That means the design supports both:

- conversational plan display
- structured plan rendering

without changing the core execution model.

## Relation To Skills And Factory

This design works well with skill and factory integration.

A proposal may naturally say things like:

1. inspect recent logs
2. query Factory for known failures
3. if matched, suggest fix
4. otherwise run a failure-analysis skill

This is valuable to the user because it explains:

- what the system will check
- whether Factory will be used
- whether a skill may be invoked

But none of this requires a standalone planner package.

## Why This Is Better Than A Heavy Planner For Phase 1

This approach keeps the system simpler because:

- there is no separate planning mode
- there is no second execution architecture
- there is no need for a workflow DSL
- there is no always-on extra LLM call

At the same time, it still supports:

- user-visible plans
- approval before execution
- future UI rendering of structured steps

## What Not To Build In Phase 1

Avoid adding these early:

- persistent plan graphs
- step-by-step approval
- resumable execution state per step
- planner-specific package hierarchy
- workflow scheduling abstractions

These may become useful later, but they are not needed for a first usable plan proposal feature.

## Recommended Phase 1 Scope

For Phase 1, implement only:

- `PlanProposal`
- `RequirePlanApproval` on `RunRequest`
- `ProposePlan(...)` on the orchestrator
- one app-side pending plan state
- simple UI rendering
- a user approval command or input pattern

That is enough to provide visible, reviewable plans without reintroducing a heavyweight planning system.

## Summary

Plan proposals should be treated as a lightweight pre-execution capability, not as a separate planner subsystem.

The recommended model is:

- use the normal agent loop for execution
- generate a short plan only when needed
- let the app and UI handle approval
- continue execution after approval

This gives `mscli` a practical plan review feature while staying aligned with a standard modern agent loop.
