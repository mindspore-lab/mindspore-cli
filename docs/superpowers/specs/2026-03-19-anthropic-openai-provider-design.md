# Anthropic + OpenAI Completion/Responses Provider Unified Design

Date: 2026-03-19
Status: Draft Approved by User (Conversation)
Scope: mscli provider architecture and protocol-path support

## 1. Goal

Implement first-class, configuration-driven support for:

- OpenAI Chat Completions
- OpenAI Responses
- Anthropic Messages API

Constraints:

- No protocol auto-detection
- Provider path decided only by config/env resolution
- Keep provider code highly centralized in one module directory

## 2. Requirements

### Functional

1. Support runtime routing to `openai-completion`, `openai-responses`, or `anthropic`.
2. Support provider selection via config and environment variables.
3. Support provider-specific message construction and response parsing.
4. Support tool call roundtrip in both protocol families.
5. Support streaming parsing for both protocol families.

### Non-Functional

1. Minimal behavioral regression for existing Chat Completions compatible users.
2. Provider concerns concentrated in one module.
3. Clear error context for diagnosis.
4. Easy extension for future providers.

## 3. Configuration Model and Priority

## 3.1 New/Updated Config Fields

Under `model`:

- `provider`: `openai-completion` | `openai-responses` | `anthropic`
- `url`: provider base URL
- `key`: generic key fallback
- `headers`: optional custom headers
- existing model options remain (`model`, `timeout_sec`)

Request-shaping overrides are env-only:

- `MSCLI_TEMPERATURE`
- `MSCLI_MAX_TOKENS`

When those env vars are unset, the provider request omits the corresponding fields, except the Anthropic Messages path, which falls back to `max_tokens=64000`.

Default in config:

- `model.provider = openai-completion`

## 3.2 Resolution Priority

Provider resolution priority:

1. `MSCLI_PROVIDER`
2. `model.provider`
3. default `openai-completion`

API key resolution (by resolved provider):

- `openai-completion` / `openai-responses`:
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

## 3.3 URL/Header/Auth Resolution Matrix

The resolver must produce a fully explicit runtime transport config:

- `provider_kind`
- `base_url`
- `auth_header_name`
- `auth_header_value` (masked in logs/traces)
- `extra_headers`

Provider defaults:

1. `openai-completion`:
- default `base_url = https://api.openai.com/v1`
- auth header: `Authorization: Bearer <key>`
- default endpoint path: `/chat/completions`

2. `openai-responses`:
- default `base_url = https://api.openai.com/v1`
- auth header: `Authorization: Bearer <key>`
- default endpoint path: `/responses`

3. `anthropic`:
- default `base_url = https://api.anthropic.com/v1`
- auth header: `x-api-key: <key>`
- required version header: `anthropic-version: 2023-06-01`
- default endpoint path: `/messages`

Base URL precedence (applies to all providers):

1. `MSCLI_BASE_URL`
2. provider-specific env override:
- `OPENAI_BASE_URL` for `openai-completion` and `openai-responses`
- `ANTHROPIC_BASE_URL` for `anthropic`
3. `model.url`
4. provider default base URL

Header precedence:

1. provider-required auth/version headers
2. `model.headers` (can add extra headers, but cannot remove required auth/version headers)

Conflict rule:

- if `model.headers` provides the same header key as required auth/version headers, resolver keeps provider-required value and emits a debug warning.

## 4. Unified Provider Module Layout

All provider code is centralized under one directory:

`integrations/llm/`

Planned files:

- `manager.go`: unified provider entrypoint and orchestration
- `resolver.go`: config/env -> resolved provider settings
- `builder_registry.go`: provider kind -> factory mapping
- `cache.go`: client instance cache
- `http.go`: shared request send/error decode/header injection
- `provider_types.go`: shared provider kind and resolver types
- `client_openai_completion.go`: OpenAI Chat Completions implementation
- `client_openai_responses.go`: OpenAI Responses implementation
- `client_anthropic.go`: Anthropic Messages implementation
- `codec_openai_completion.go`: internal message <-> OpenAI Chat Completions payload/chunks
- `codec_openai_responses.go`: internal message <-> OpenAI Responses payload/chunks
- `codec_anthropic.go`: internal message <-> Anthropic payload/chunks

Application wiring change:

- `internal/app/wire.go` uses provider manager/factory entrypoint.
- Remove direct dependency on `integrations/llm/openai` concrete constructor.

## 5. Message Construction and Parsing

Internal model remains based on existing `llm.Message`, `llm.ToolCall`, `llm.CompletionResponse`.

Protocol-specific conversion happens only in codecs.

### 5.1 OpenAI Chat Completions Path

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

### 5.2 OpenAI Responses Path

Request:

- top-level `instructions` from internal system messages
- `input[]` carries user/assistant/tool history
- assistant tool calls encode as `function_call`
- tool output encodes as `function_call_output`
- tool schemas are sent with `strict: true`
- intra-task continuation uses `previous_response_id`

Response parse:

- aggregate output text into `CompletionResponse.Content`
- convert `function_call` output items into `CompletionResponse.ToolCalls`
- map response status/stop conditions into internal finish reason

Streaming:

- parse Responses SSE event sequence
- emit `response.output_text.delta` as text chunks immediately
- accumulate function call argument deltas until item completion
- use final `response.completed` event for response id/model/usage

### 5.3 Anthropic Path (Messages API)

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
- `/model openai-completion:<model>`
- `/model openai-responses:<model>`
- `/model anthropic:<model>`

Model status display includes provider kind and key status.

State mutation rules:

1. `/model <model-name>`:
- updates model name only
- keeps current provider unchanged
- persists to state (same behavior as current model switch persistence)

2. `/model <provider>:<model-name>`:
- updates both provider and model name
- persists both changes to state

3. Invalid provider prefix:
- fail fast with explicit supported values (`openai-completion`, `openai-responses`, `anthropic`)
- do not mutate state

## 8. Migration and Compatibility

1. Existing configs without `model.provider` now resolve to default `openai-completion`.
2. Existing OpenAI-compatible endpoint/key usage remains valid after renaming provider to `openai-completion`.
3. The legacy provider value `openai-compatible` is removed and should fail fast during validation.
4. Anthropic users can opt in by setting `model.provider=anthropic` and auth token env.

## 9. Test Plan (Acceptance)

Required tests:

1. Resolver priority tests:
   - env overrides config
   - config overrides default
2. Key selection tests per provider kind
3. OpenAI codec tests (request + response + stream + tool calls)
4. Anthropic codec tests (request + response + stream + tool use/result)
5. Unified error parse tests
6. Regression tests for existing openai-completion behavior
7. `/model` mutation tests:
   - unprefixed model switch keeps provider
   - prefixed model switch updates provider+model
   - invalid prefix does not persist state changes
8. Resolver transport resolution tests:
   - base URL precedence matrix
   - required header enforcement matrix
9. Anthropic stream assembly tests:
   - text delta accumulation
   - tool_use partial assembly and completion
   - stop reason mapping and final chunk behavior

Acceptance criteria:

1. `openai-completion` default path works out of the box.
2. `model.provider=anthropic` + token env routes to Anthropic path.
3. Tool call loop works in all three provider paths.
4. Streaming works in all three provider paths.
5. Existing openai-completion flows are not regressed.

## 10. Out of Scope (This Change)

1. Runtime protocol auto-detection or probe requests.
2. Silent provider fallback across protocol families.
3. Expanding to additional providers beyond the three declared kinds.
4. Non-provider architecture refactors unrelated to request routing/parsing.

## 11. Implementation Next Step

After this design is finalized and reviewed, create a concrete implementation plan using the writing-plans workflow and execute in incremental, test-backed changes.

## Appendix A: Anthropic Streaming Event Mapping Contract

This appendix defines the implementation contract for Anthropic streaming parsing.

State machine (per response stream):

1. Initialize empty accumulators:
- `text_buffer` (string)
- `tool_calls_by_index` (ordered map)
- `final_finish_reason` (empty)

2. On text delta event:
- append to `text_buffer`
- emit `StreamChunk{Content: <delta_text>}` immediately

3. On tool use start/delta events:
- locate/create tool call slot by index (or stable stream id when index absent)
- accumulate fields:
  - `id`
  - `name`
  - `input_json_fragment`
- parse/validate assembled `input` JSON only when provider signals the input block is complete

4. Tool call completion condition:
- a tool call is considered complete only after the stream signals end of the corresponding tool_use block.
- on completion, emit chunk containing current complete tool calls snapshot.

5. Finish condition:
- map Anthropic stop reason to internal finish reason:
  - `end_turn` -> `stop`
  - `max_tokens` -> `length`
  - `tool_use` -> `tool_calls`
- when finish arrives, iterator returns final chunk with mapped finish reason and aggregated usage (if present).

6. Error condition:
- malformed partial tool JSON before tool block close is treated as stream parse error.
- include provider/model/event context in error and stop iteration.
