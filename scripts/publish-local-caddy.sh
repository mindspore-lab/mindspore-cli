#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

VERSION="${1:-}"

if [ -z "${VERSION}" ]; then
  echo "Usage: ./scripts/publish-local-caddy.sh <version>"
  echo "Example: ./scripts/publish-local-caddy.sh v0.4.26"
  exit 1
fi

if [[ "${VERSION}" != v* ]]; then
  echo "Error: version must include leading v, for example v0.4.26" >&2
  exit 1
fi

DIST_DIR="${MSCLI_DIST_DIR:-${REPO_ROOT}/dist}"
MIRROR_ROOT="${MSCLI_MIRROR_ROOT:-/opt/downloads/mscli/releases}"
TARGET_DIR="${MIRROR_ROOT}/${VERSION}"
LATEST_LINK="${MIRROR_ROOT}/latest"
PUBLIC_ROOT="$(dirname "${MIRROR_ROOT}")"
INSTALL_SCRIPT_SOURCE="${MSCLI_INSTALL_SCRIPT_SOURCE:-${REPO_ROOT}/scripts/install.sh}"
MIRROR_BASE_URL="${MSCLI_MIRROR_BASE_URL:-https://mscli.dev/mscli/releases}"
INSTALL_SCRIPT_PATH="${PUBLIC_ROOT}/install.sh"

required_files=(
  "manifest.json"
  "mscli-linux-amd64"
  "mscli-linux-arm64"
  "mscli-darwin-amd64"
  "mscli-darwin-arm64"
  "mscli-windows-amd64.exe"
  "mscli-server-linux-amd64"
)

for file in "${required_files[@]}"; do
  if [ ! -f "${DIST_DIR}/${file}" ]; then
    echo "Error: missing required asset: ${DIST_DIR}/${file}" >&2
    exit 1
  fi
done

echo "Publishing ${VERSION} from ${DIST_DIR} to ${TARGET_DIR}"

# Rewrite manifest download_base for the local mirror.
if [ -f "${DIST_DIR}/manifest.json" ]; then
  python3 -c "
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
d['download_base'] = sys.argv[2].rstrip('/')
p.write_text(json.dumps(d, indent=2) + '\n')
" "${DIST_DIR}/manifest.json" "${MIRROR_BASE_URL}"
fi

mkdir -p "${TARGET_DIR}"
cp "${DIST_DIR}"/* "${TARGET_DIR}/"
cp "${INSTALL_SCRIPT_SOURCE}" "${INSTALL_SCRIPT_PATH}"
chmod -R a+rX "${TARGET_DIR}"
chmod a+rX "${INSTALL_SCRIPT_PATH}"
ln -sfn "${TARGET_DIR}" "${LATEST_LINK}"

echo ""
echo "Published ${VERSION} to local Caddy mirror:"
echo "  ${TARGET_DIR}"
echo "Latest link:"
echo "  ${LATEST_LINK} -> ${TARGET_DIR}"
echo "Public install script:"
echo "  ${INSTALL_SCRIPT_PATH}"
