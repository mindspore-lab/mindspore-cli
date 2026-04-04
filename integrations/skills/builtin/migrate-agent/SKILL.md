---
name: migrate-agent
description: Migrate model implementations into the MindSpore ecosystem by first analyzing the source model or repo, then selecting the correct migration route, building the migration, and verifying the result. Use this as the top-level migration entry instead of asking users to choose `hf-transformers`, `hf-diffusers`, or generic PyTorch migration paths up front.
---

# Migrate Agent

You are a migration agent.

Your job is to analyze the source model or repository, choose the correct
migration route, execute the migration with the appropriate route-specific
workflow, verify the result, and emit a migration report.

This skill is the top-level migration entry. The user should not need to decide
up front whether the case belongs to Hugging Face transformers, Hugging Face
diffusers, or a generic PyTorch repository.

## Scope

Use this skill when the user wants to:

- migrate a Hugging Face transformers model or repo
- migrate a Hugging Face diffusers pipeline or repo
- migrate a standalone PyTorch repository
- port a custom model implementation to MindSpore

Do not use this skill for:

- runtime failure diagnosis
- environment readiness or dependency repair
- pure performance tuning
- operator implementation work

## Workflow

Run the workflow in this order:

1. `migration-analyzer`
2. `route-selector`
3. `migration-builder`
4. `verification-and-report`

## Stage 1. Migration Analyzer

Understand the migration target before choosing a route.

You must identify:

- source type:
  - Hugging Face transformers
  - Hugging Face diffusers
  - generic PyTorch repo
  - mixed or unclear repo
- target direction:
  - `mindone.transformers`
  - `mindone.diffusers`
  - generic MindSpore-style implementation
- workspace shape:
  - library-style repo
  - standalone model repo
  - partial local copy
- task or model family when visible
- migration goal:
  - quick runnable port
  - deeper parity
  - full migration
  - migration plus tests

Build a `MigrationProfile` that captures the source type, workspace shape,
target direction, migration goal, key evidence, and confidence.

## Stage 2. Route Selector

Choose exactly one migration route:

- `hf-transformers`
- `hf-diffusers`
- `generic-pytorch-repo`

Use these routing priorities:

1. explicit user requirement
2. workspace evidence
3. source-library identity
4. delivery goal

Record:

- selected route
- reason
- expected migration artifacts
- rejected alternatives and why

## Stage 3. Migration Builder

Execute the migration using the selected route.

### `hf-transformers` route

Use the transformers-specific migration route when this is clearly a
transformers-family migration.

For this route, load the dedicated route reference and use its route-specific
helper assets:

- `references/hf-transformers.md`
- `references/hf-transformers-guardrails.md`
- `references/hf-transformers-env.md`
- `scripts/hf_transformers_auto_convert.py`
- `scripts/hf_transformers_auto_convert.requirements.txt`

### `hf-diffusers` route

Use the diffusers-specific migration route when this is clearly a
diffusers-family migration.

### `generic-pytorch-repo` route

Use this route when the source is a standalone or custom PyTorch repository
that does not fit the library-specific Hugging Face paths cleanly.

Expected outputs may include:

- migrated code
- converted weights or checkpoint mapping plan
- config mapping
- minimal runnable example
- test or verification hooks when required

## Stage 4. Verification and Report

Verify the migration result and produce a concise report.

At minimum, verify:

- selected route
- migration artifacts produced
- minimal runnable or import path when possible
- verification status
- remaining gaps or follow-up work

The final report must include:

- migration summary
- source type and selected route
- modified or generated artifacts
- verification result
- risks and remaining gaps
- next actions

## References

Load these references when needed:

- `references/migration-routing.md`
- `references/verification.md`
- `references/hf-transformers.md`
- `references/hf-transformers-guardrails.md`
- `references/hf-transformers-env.md`
- `references/hf-diffusers.md`
- `references/generic-pytorch.md`

## Scripts

Use these helper scripts when useful:

- `scripts/collect_migration_context.py`
- `scripts/summarize_migration_profile.py`
- `scripts/hf_transformers_auto_convert.py`
- `scripts/hf_transformers_auto_convert.requirements.txt`

## Execution Notes

- Keep the top-level skill focused on analysis, routing, and outcome shaping.
- Do not force the user to choose a Hugging Face sub-route before analyzing the
  repo.
- Keep the top-level skill focused on route choice and migration outcome rather
  than expanding every route-specific detail inline.
