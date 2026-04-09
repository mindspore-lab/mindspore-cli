# Provider Import Duplicate Options Design

**Goal:** Keep env-detected provider import candidates visible in the `Import` section without removing the original provider from `Popular` or `Other`.

## Problem

When a provider such as `kimi-for-coding` is detected from Claude Code environment variables, the current `/model` provider list shows only the import candidate at the top. The original catalog provider entry disappears from the normal provider sections.

This collapses two different behaviors into one visible choice:

- Import candidate: may reuse the detected `ANTHROPIC_API_KEY` and skip manual API key entry
- Normal provider entry: must behave like every other provider option and require manual API key entry

## Approved Design

- Keep both visible options in the provider list.
- Preserve the current `Import` group presentation and detail rows.
- Keep the visible label unchanged for both options.
- Distinguish the import option only with an internal option ID.
- Allow env API key fallback only when the selected option came from the `Import` group.
- Require manual API key entry for the normal provider option even if matching env vars are present.

## Minimal Change Boundary

- Do not change provider catalog structures.
- Do not change auth/model state persistence formats.
- Do not change import detection or display copy.
- Restrict logic changes to provider option construction, provider connect parsing, and targeted tests.
