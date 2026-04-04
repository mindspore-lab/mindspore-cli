# Command Design: /diagnose and /fix

## Problem

Users encounter three categories of problems during training:

- **Failure**: crashes, runtime errors, missing operators, device communication issues
- **Accuracy**: accuracy regression, numerical drift, wrong evaluation results
- **Performance**: low throughput, high latency, memory pressure

From the user's perspective, the workflow is always the same regardless of
problem type: diagnose, find root cause, propose fix, apply fix, verify. The
difference is only in what evidence to collect and which factory cards to
consult.

## Commands

### /diagnose — analyze only

```
/diagnose my training crashed with DSA operator error
/diagnose accuracy dropped 16.8 pts after switching to ascend
/diagnose throughput is only 340 tok/s, expected 520
```

Runs phases 1-3 of the diagnosis workflow:

1. **Collect evidence** (logs, metrics, config, environment)
2. **Query factory** for known issues and relevant cards
3. **Find root cause** (generate hypotheses, distill to 1-2 most likely)

Stops after presenting the diagnosis report. User decides what to do next.
This is a read-only, safe operation.

### /fix — full cycle

```
/fix my training crashed with DSA operator error
/fix accuracy dropped 16.8 pts after switching to ascend
/fix throughput is only 340 tok/s, expected 520
```

Runs the complete workflow:

1. **Diagnose** (same as /diagnose phases 1-3)
2. **Propose fix plan** with diff preview
3. **Get user confirmation** before applying
4. **Apply fix** (patch config, code, or environment)
5. **Verify** (rerun training, re-evaluate, re-profile)

`/fix` includes `/diagnose` as its first phase. The user confirmation gate
(phase 3) ensures nothing is applied without explicit approval.

## Routing

The CLI does lightweight symptom classification to load the right skill:

```
user input -> keyword classification -> load skill -> run diagnosis workflow
```

Classification rules (deterministic, not LLM-based):

| keywords | symptom | skill loaded |
|----------|---------|--------------|
| crash, error, failed, operator, CANN, not implemented, RuntimeError | failure | failure-agent |
| accuracy, drift, regression, eval, loss diverge, wrong result | accuracy | accuracy-agent |
| throughput, latency, slow, memory, OOM, tok/s, bottleneck | performance | performance-agent |

If classification is ambiguous, the CLI asks the user to clarify:

```
> /diagnose something is wrong with my training

Which type of problem are you seeing?
1. Training crashed or failed to start
2. Accuracy or evaluation results are wrong
3. Performance is worse than expected
```

This is a thin deterministic router in mscli, not an LLM routing call. The
three skills share the same workflow structure but differ in evidence
collection and factory card queries.

## Relationship to Skills

Both `/diagnose` and `/fix` activate the same skill. The command tells the
skill which mode to run:

- `/diagnose` -> skill runs in `diagnose` mode (stops after root cause)
- `/fix` -> skill runs in `fix` mode (full cycle with apply + verify)

The skill receives this as part of the task description, not as a separate
parameter. The SKILL.md instructions handle both modes.

## Relationship to Factory Cards

During the diagnosis phase, skills query factory cards:

| symptom | primary cards queried | secondary cards queried |
|---------|----------------------|------------------------|
| failure | `known_issue` (symptom=failure), `operator` | `model` (for expected behavior) |
| accuracy | `known_issue` (symptom=accuracy), `algo_feature` | `model` (for baselines), `operator` |
| performance | `known_issue` (symptom=performance), `perf_feature` | `model` (for expected throughput), `operator` |

The `known_issue.symptom` field enables filtering: each skill queries only
the issues relevant to its problem category.

## User Experience Flow

```
User: /fix my training crashed at step 50 with DSA operator error

CLI: [routes to failure-agent skill in fix mode]

Agent: Collecting evidence...
       - Error log: DSA operator (torch.ops.npu.dsa) not implemented in torch 2.7
       - Platform: Ascend 910B, torch 2.7, CANN 8.0.RC2

       Querying factory for known issues...
       - Match: known_issue/dsa-torch27-ascend (confidence: high)
         Root cause: DSA op not available in torch 2.7 for Ascend backend

       Diagnosis:
       - Root cause: DSA operator not implemented in torch 2.7
       - Severity: critical
       - Fix: Implement DSA kernel in torch_npu and recompile

       Proposed fix:
       [diff preview]

       Apply this fix? [y/n]

User: y

Agent: Applying fix...
       Verifying... rerunning training...
       Training passed step 50 successfully.
       Fix verified.
```

## Design Decisions

1. **Two commands, not three** — `/diagnose` and `/fix` cover all symptom
   types. We don't need `/diagnose-crash`, `/diagnose-accuracy`,
   `/diagnose-perf` as separate commands. Symptom routing happens
   automatically inside the command.

2. **Deterministic routing, not LLM routing** — keyword classification is
   reliable and fast. The LLM is used for diagnosis, not for deciding which
   skill to load.

3. **User confirmation gate** — `/fix` always pauses before applying changes.
   This is inspired by the aeitroc/commands `/fix` pattern where user
   confirmation is mandatory before implementation.

4. **Shared workflow, specialized evidence** — all three skills follow the
   same diagnosis structure. Only the evidence collection and card queries
   differ. This keeps the design consistent and the skills composable.
