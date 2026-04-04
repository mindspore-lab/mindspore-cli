# Readiness-Agent Env-Fix Policy

`fix` mode is limited to safe user-space changes.

Allowed actions:

- install `uv` into the user environment when needed
- create or repair a workspace-local environment such as `.venv`
- install missing framework or runtime packages into the selected env
- scaffold a bundled example entry script for a known recipe
- download explicitly declared model or dataset assets when `allow_network=true`

Disallowed actions:

- modify driver, firmware, or CANN
- mutate system Python
- rewrite user model logic to make smoke pass
- make distributed or cluster-level environment changes

Every successful fix must be followed by revalidation of the affected scopes
before final certification.
