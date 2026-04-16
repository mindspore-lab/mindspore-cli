# Claude Code Technical Route (Source-Validated)

## Purpose

This note records the Claude Code terminal UI architecture after comparing prior behavior-based guesses with source under `claude-code-full/package/src`.

It replaces the earlier inference-only model.

## What Was Confirmed

### 1. `ctrl+o` toggles transcript mode, not a tool-only viewer

Claude Code has an explicit screen state:

```text
'prompt' | 'transcript'
```

`ctrl+o` flips between those two screens. This is implemented in:

- `src/hooks/useGlobalKeybindings.tsx`
- `src/screens/REPL.tsx`

This means the earlier description of `ctrl+o` as "open historical tool output" was too narrow. Tool results are included, but the feature is a full transcript screen.

### 2. Claude Code keeps structured message history in app state

Transcript mode renders from in-memory message/tool state rather than from terminal scrollback.

Evidence:

- `src/screens/REPL.tsx` builds `transcriptMessagesElement` from `transcriptMessages`, `tools`, `commands`, `transcriptStreamingToolUses`
- `src/components/Messages.tsx` applies transcript-mode-specific filtering/grouping/collapse rules over normalized message state

This part of the earlier document was correct in spirit: old tool results are recoverable because the app retains structured history.

### 3. Transcript mode is a full-app rendering mode

Transcript mode is not just "expand one tool result". It changes how the whole conversation is rendered.

Evidence:

- `src/components/Messages.tsx`
- `src/components/MessageRow.tsx`
- `src/components/CompactSummary.tsx`
- `src/components/messages/*`

Examples of transcript-specific behavior:

- transcript mode can bypass some normal filtering
- transcript mode can hide past thinking blocks except the latest one
- transcript mode can show extra metadata such as timestamps/models
- transcript mode has its own footer/search/scroll behavior

### 4. Claude Code does have a real alternate-screen fullscreen path

Claude Code includes an `AlternateScreen` component that explicitly enters DEC 1049 alt screen and exits it on unmount.

Evidence:

- `src/ink/components/AlternateScreen.tsx`
- `src/utils/fullscreen.ts`

Key facts:

- it writes `ENTER_ALT_SCREEN` / `EXIT_ALT_SCREEN`
- it optionally enables mouse tracking
- it constrains rendering to terminal height
- comments explicitly say alt screen has no native scrollback

### 5. Fullscreen/alt-screen is environment-gated, not always on

Claude Code does not always run in fullscreen for every user.

From `src/utils/fullscreen.ts` and `src/components/FullscreenLayout.tsx`:

- fullscreen defaults on for internal `ant` users
- fullscreen defaults off for external users
- env vars can force opt-in/opt-out
- tmux `-CC` disables fullscreen automatically

So whether a user can still access pre-`claude` shell scrollback depends on which branch they are on.

### 6. There are two transcript-mode branches

Transcript mode itself has two rendering strategies:

1. Fullscreen + virtual scroll:
   - wrapped in `AlternateScreen`
   - uses `FullscreenLayout`
   - app owns scrolling
   - no native terminal scrollback for transcript content

2. Legacy non-fullscreen dump path:
   - no alt screen
   - dumps transcript into terminal scrollback
   - keeps a render cap plus `Ctrl+E`

This split is documented directly in comments in `src/screens/REPL.tsx`.

## What The Earlier Version Got Right

- Claude Code is not a pure "all history only lives in terminal scrollback" app.
- Claude Code retains structured history in memory and can reconstruct older tool results.
- There is a hybrid architecture rather than one single rendering model.
- The loss of pre-app shell history after entering the fullscreen transcript path is consistent with a buffer switch.

## What Needed Correction

- `ctrl+o` is not a dedicated tool-output viewer. It is a prompt/transcript screen toggle.
- The architecture is not simply "normal inline mode" plus "tool output view mode".
- Alt-screen is real in the codebase, but it is conditional, not universal.
- Prompt mode can also run inside `AlternateScreen` when fullscreen is enabled; transcript mode is not the only alt-screen consumer.
- Claude Code also supports a non-fullscreen transcript fallback, so user-observed behavior depends on runtime environment.

## Working Model

The best source-backed model is:

```text
REPL screen state
  - prompt
  - transcript

rendering strategy
  - fullscreen enabled:
      use AlternateScreen + FullscreenLayout + app-managed scrolling
  - fullscreen disabled:
      stay on main terminal buffer and rely more on terminal scrollback

ctrl+o
  - toggles prompt <-> transcript
  - transcript renders from structured message/tool state
  - not limited to tool output
```

## Practical Takeaways For mscli

If you want to learn from Claude Code, the important ideas are:

1. Separate "screen mode" from "message expansion state".
2. Keep a structured conversation/tool model in memory; do not depend on terminal scrollback as your source of truth.
3. Support two rendering regimes:
   - main-buffer / scrollback-friendly
   - fullscreen / alt-screen / app-scrolled
4. Treat transcript as a first-class screen, not just a temporary viewer for one block.
5. Make transcript-mode rendering rules explicit instead of trying to mutate already-printed terminal history in place.

## Status

Current status: **source-validated against local Claude Code source**
