#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ── Parse args ──────────────────────────────────────────────────
VERSION=""
NOTES=""
SKIP_GITHUB=0
SKIP_MIRROR=0

for arg in "$@"; do
  case "$arg" in
    --skip-github) SKIP_GITHUB=1 ;;
    --skip-mirror) SKIP_MIRROR=1 ;;
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
  echo "Usage: ./scripts/release-all.sh <version> [notes] [--skip-github] [--skip-mirror]"
  echo ""
  echo "Examples:"
  echo "  ./scripts/release-all.sh v0.5.1 \"Fix bug\"        # full release"
  echo "  ./scripts/release-all.sh v0.5.1 --skip-github     # mirror only"
  echo "  ./scripts/release-all.sh v0.5.1 \"notes\" --skip-mirror  # GitHub only"
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

# ── Step 2: Mirror deploy ──────────────────────────────────────
if [ "$SKIP_MIRROR" -eq 0 ]; then
  echo ""
  echo "==> Mirror deploy"
  # Source mirror defaults and export for publish-caddy-mirror.sh
  if [ -f "${SCRIPT_DIR}/mirror.conf" ]; then
    source "${SCRIPT_DIR}/mirror.conf"
  fi
  export MSCLI_MIRROR_HOST MSCLI_MIRROR_USER MSCLI_MIRROR_PORT MSCLI_MIRROR_SSH_KEY
  export MSCLI_MIRROR_ROOT MSCLI_MIRROR_BASE_URL
  "${SCRIPT_DIR}/publish-caddy-mirror.sh" "$VERSION"
else
  echo ""
  echo "==> Skipping mirror deploy"
fi

echo ""
echo "Done. Release $VERSION complete."
