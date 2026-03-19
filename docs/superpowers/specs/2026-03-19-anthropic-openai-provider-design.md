# Anthropic + OpenAI/OpenAI-Compatible Provider Unified Design

Date: 2026-03-19
Status: Draft Approved by User (Conversation)
Scope: ms-cli provider architecture and protocol-path support

## 1. Goal

Implement first-class, configuration-driven support for:

- OpenAI
- OpenAI-compatible
- Anthropic Messages API

Constraints:

- No protocol auto-detection
- Provider path decided only by config/env resolution
- Keep provider code highly centralized in one module directory

## 2. Requirements

### Functional

1. Support runtime routing to `openai`, `openai-compatible`, or `anthropic`.
2. Support provider selection via config and environment variables.
3. Support provider-specific message construction and response parsing.
4. Support tool call roundtrip in both protocol families.
5. Support streaming parsing for both protocol families.

### Non-Functional

1. Minimal behavioral regression for existing OpenAI-compatible users.
2. Provider concerns concentrated in one module.
3. Clear error context for diagnosis.
4. Easy extension for future providers.

## 3. Configuration Model and Priority

## 3.1 New/Updated Config Fields

Under `model`:

- `provider`: `openai` | `openai-compatible` | `anthropic`
- `url`: provider base URL
- `key`: generic key fallback
- `headers`: optional custom headers
- existing model options remain (`model`, `temperature`, `max_tokens`, `timeout_sec`)

Default in config:

- `model.provider = openai-compatible`

## 3.2 Resolution Priority

Provider resolution priority:

1. `MSCLI_PROVIDER`
2. `model.provider`
3. default `openai-compatible`

API key resolution (by resolved provider):

- `openai` / `openai-compatible`:
  1. `MSCLI_API_KEY`
  2. `OPENAI_API_KEY`
  3. `model.key`

- `anthropic`:
  1. `ANTHROPIC_AUTH_TOKEN`
  2. `ANTHROPIC_API_KEY`
  3. `model.key`

Notes:

- No URL-based guessing.
- No probe request for protocol inference.

## 4. Unified Provider Module Layout

All provider code is centralized under one directory:

`integrations/llm/provider/`

Planned files:

- `manager.go`: unified provider entrypoint and orchestration
- `resolver.go`: config/env -> resolved provider settings
- `registry.go`: provider kind -> factory mapping
- `cache.go`: client instance cache
- `http.go`: shared request send/error decode/header injection
- `types.go`: shared internal transport structs
- `client_openai.go`: OpenAI implementation
- `client_openai_compatible.go`: OpenAI-compatible implementation
- `client_anthropic.go`: Anthropic Messages implementation
- `codec_openai.go`: internal message <-> OpenAI payload/chunks
- `codec_anthropic.go`: internal message <-> Anthropic payload/chunks

Application wiring change:

- `internal/app/wire.go` uses provider manager/factory entrypoint.
- Remove direct dependency on `integrations/llm/openai` concrete constructor.

## 5. Message Construction and Parsing

Internal model remains based on existing `llm.Message`, `llm.ToolCall`, `llm.CompletionResponse`.

Protocol-specific conversion happens only in codecs.

### 5.1 OpenAI / OpenAI-Compatible Path

Request:

- `messages[].role/content`
- assistant tool calls -> `tool_calls`
- tool output -> `role=tool` + `tool_call_id`

Response parse:

- parse `choices[0].message.content`
- parse `choices[0].message.tool_calls`
- map `finish_reason=tool_calls` -> `llm.FinishToolCalls`

Streaming:

- parse SSE `data:` events
- accumulate `delta.content`
- incrementally assemble `delta.tool_calls`

### 5.2 Anthropic Path (Messages API)

Request:

- top-level `system` from internal system messages (joined by `\n\n`)
- `messages[]` only user/assistant
- text -> `{type:"text", text:"..."}` blocks
- assistant tool call -> `{type:"tool_use", id, name, input}` block
- tool output -> user block `{type:"tool_result", tool_use_id, content, is_error}`

Response parse:

- aggregate `content[]` text blocks into `CompletionResponse.Content`
- convert `tool_use` blocks into `CompletionResponse.ToolCalls`
- map `stop_reason=tool_use` -> `llm.FinishToolCalls`

Streaming:

- parse Anthropic SSE event sequence
- accumulate text deltas
- accumulate tool_use fragments, emit completed tool call state

## 6. Error Model and Observability

## 6.1 Unified Error Envelope

Shared error context fields:

- `provider_kind`
- `endpoint`
- `model`
- `status_code`
- `upstream_type`
- `upstream_message`

Both OpenAI and Anthropic decoders map upstream error payload into this envelope.

## 6.2 Policy

- No protocol fallback on failure.
- If provider is misconfigured (missing required key), fail fast with actionable message.

## 6.3 Trace/Debug

Add provider dimensions to trace/debug (masked secrets):

- `provider_kind`
- `base_url`
- `model`
- `stream`
- request identifier when available
- decision chain source (env/config/default)

## 7. CLI Behavior Changes

`/model` should accept:

- `/model <model-name>` (current behavior)
- `/model openai:<model>`
- `/model openai-compatible:<model>`
- `/model anthropic:<model>`

Model status display includes provider kind and key status.

## 8. Migration and Compatibility

1. Existing configs without `model.provider` continue to work using default `openai-compatible`.
2. Existing OpenAI-compatible endpoint/key usage remains valid.
3. Anthropic users can opt in by setting `model.provider=anthropic` and auth token env.

## 9. Test Plan (Acceptance)

Required tests:

1. Resolver priority tests:
   - env overrides config
   - config overrides default
2. Key selection tests per provider kind
3. OpenAI codec tests (request + response + stream + tool calls)
4. Anthropic codec tests (request + response + stream + tool use/result)
5. Unified error parse tests
6. Regression tests for existing openai-compatible behavior

Acceptance criteria:

1. `openai-compatible` default path works out of the box.
2. `model.provider=anthropic` + token env routes to Anthropic path.
3. Tool call loop works in both protocol families.
4. Streaming works in both protocol families.
5. Existing openai-compatible flows are not regressed.

## 10. Out of Scope (This Change)

1. Runtime protocol auto-detection or probe requests.
2. Silent provider fallback across protocol families.
3. Expanding to additional providers beyond the three declared kinds.
4. Non-provider architecture refactors unrelated to request routing/parsing.

## 11. Implementation Next Step

After this design is finalized and reviewed, create a concrete implementation plan using the writing-plans workflow and execute in incremental, test-backed changes.
