#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ── Parse args ──────────────────────────────────────────────────
VERSION=""
NOTES=""
SKIP_GITHUB=0
SKIP_LOCAL=0

for arg in "$@"; do
  case "$arg" in
    --skip-github) SKIP_GITHUB=1 ;;
    --skip-local)  SKIP_LOCAL=1 ;;
    *)
      if [ -z "$VERSION" ]; then
        VERSION="$arg"
      elif [ -z "$NOTES" ]; then
        NOTES="$arg"
      fi
      ;;
  esac
done

if [ -z "$VERSION" ]; then
  echo "Usage: ./scripts/release-all.sh <version> [notes] [--skip-github] [--skip-local]"
  echo ""
  echo "Examples:"
  echo "  ./scripts/release-all.sh v0.5.1 \"Fix bug\"        # full release"
  echo "  ./scripts/release-all.sh v0.5.1 --skip-github     # local mirror only"
  echo "  ./scripts/release-all.sh v0.5.1 \"notes\" --skip-local  # GitHub only"
  exit 1
fi

if [[ "${VERSION}" != v* ]]; then
  echo "Error: version must include a leading v, for example v0.5.1" >&2
  exit 1
fi

: "${NOTES:="Release $VERSION"}"

# ── Ensure we're on main ────────────────────────────────────────
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$CURRENT_BRANCH" != "main" ]; then
  echo "Error: must release from main (currently on $CURRENT_BRANCH)" >&2
  exit 1
fi

# ── Step 0: Update embedded skills ─────────────────────────────
echo "==> Update embedded skills"
"${SCRIPT_DIR}/update-skills.sh"

# ── Step 1: Build + GitHub release ──────────────────────────────
if [ "$SKIP_GITHUB" -eq 0 ]; then
  echo "==> GitHub release"
  "${SCRIPT_DIR}/release.sh" "$VERSION" "$NOTES"
else
  echo "==> Skipping GitHub release"
  if [ ! -d "dist" ] || [ ! -f "dist/manifest.json" ]; then
    echo "Error: dist/ directory missing. Run without --skip-github first to build binaries." >&2
    exit 1
  fi
fi

# ── Step 2: Local mirror deploy ────────────────────────────────
if [ "$SKIP_LOCAL" -eq 0 ]; then
  echo ""
  echo "==> Local mirror deploy"
  "${SCRIPT_DIR}/publish-local-caddy.sh" "$VERSION"
else
  echo ""
  echo "==> Skipping local mirror deploy"
fi

echo ""
echo "Done. Release $VERSION complete."
