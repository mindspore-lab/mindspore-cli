#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${SCRIPT_DIR}/mirror.conf" ] && source "${SCRIPT_DIR}/mirror.conf"

VERSION="${1:-}"

if [ -z "${VERSION}" ]; then
  echo "Usage: ./scripts/publish-caddy-mirror.sh <version>"
  echo "Example: MSCLI_MIRROR_HOST=ecs.example.com MSCLI_MIRROR_BASE_URL=https://download.example.com/mscli/releases ./scripts/publish-caddy-mirror.sh v0.4.25"
  exit 1
fi

if [[ "${VERSION}" != v* ]]; then
  echo "Error: version must include a leading v, for example v0.4.25" >&2
  exit 1
fi

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command not found: $1" >&2
    exit 1
  fi
}

need_cmd ssh
need_cmd scp
need_cmd mktemp
need_cmd python3

MIRROR_HOST="${MSCLI_MIRROR_HOST:-}"
MIRROR_USER="${MSCLI_MIRROR_USER:-root}"
MIRROR_PORT="${MSCLI_MIRROR_PORT:-22}"
MIRROR_SSH_KEY="${MSCLI_MIRROR_SSH_KEY:-}"
DIST_DIR="${MSCLI_DIST_DIR:-dist}"
MIRROR_ROOT="${MSCLI_MIRROR_ROOT:-/opt/downloads/mscli/releases}"
MIRROR_BASE_URL="${MSCLI_MIRROR_BASE_URL:-}"
MIRROR_PUBLIC_ROOT="${MSCLI_MIRROR_PUBLIC_ROOT:-$(dirname "${MIRROR_ROOT}")}"
REMOTE_INSTALL_SCRIPT_PATH="${MSCLI_REMOTE_INSTALL_SCRIPT_PATH:-${MIRROR_PUBLIC_ROOT}/install.sh}"
INSTALL_SCRIPT_SOURCE="${MSCLI_INSTALL_SCRIPT_SOURCE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/scripts/install.sh}"
REMOTE_CHMOD="${MSCLI_REMOTE_CHMOD:-a+rX}"
REMOTE_LATEST_LINK="${MSCLI_REMOTE_LATEST_LINK:-1}"

if [ -z "${MIRROR_HOST}" ]; then
  echo "Error: MSCLI_MIRROR_HOST is required" >&2
  exit 1
fi

SSH_KEY_OPT=()
if [ -n "${MIRROR_SSH_KEY}" ] && [ -f "${MIRROR_SSH_KEY}" ]; then
  SSH_KEY_OPT=(-i "${MIRROR_SSH_KEY}")
fi

if [ ! -d "${DIST_DIR}" ]; then
  echo "Error: dist directory not found: ${DIST_DIR}" >&2
  echo "Build first with ./scripts/release.sh ${VERSION} or create the binaries under ${DIST_DIR}" >&2
  exit 1
fi

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

if [ ! -f "${INSTALL_SCRIPT_SOURCE}" ]; then
  echo "Error: install script not found: ${INSTALL_SCRIPT_SOURCE}" >&2
  exit 1
fi

remote_target="${MIRROR_ROOT}/${VERSION}"
remote_latest="${MIRROR_ROOT}/latest"
ssh_target="${MIRROR_USER}@${MIRROR_HOST}"
tmp_manifest=""

cleanup() {
  if [ -n "${tmp_manifest}" ] && [ -f "${tmp_manifest}" ]; then
    rm -f "${tmp_manifest}"
  fi
}
trap cleanup EXIT

manifest_source="${DIST_DIR}/manifest.json"
manifest_to_upload="${manifest_source}"

if [ -n "${MIRROR_BASE_URL}" ]; then
  tmp_manifest="$(mktemp)"
  python3 - "${manifest_source}" "${tmp_manifest}" "${MIRROR_BASE_URL}" <<'PY'
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
base = sys.argv[3]

data = json.loads(src.read_text(encoding="utf-8"))
data["download_base"] = base.rstrip("/")
dst.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
  manifest_to_upload="${tmp_manifest}"
fi

echo "Publishing ${VERSION} to ${ssh_target}:${remote_target}"
ssh "${SSH_KEY_OPT[@]}" -p "${MIRROR_PORT}" "${ssh_target}" "mkdir -p '${remote_target}'"

for file in "${required_files[@]}"; do
  local_path="${DIST_DIR}/${file}"
  if [ "${file}" = "manifest.json" ]; then
    local_path="${manifest_to_upload}"
  fi
  echo "  -> ${file}"
  scp "${SSH_KEY_OPT[@]}" -P "${MIRROR_PORT}" "${local_path}" "${ssh_target}:${remote_target}/${file}"
done

ssh "${SSH_KEY_OPT[@]}" -p "${MIRROR_PORT}" "${ssh_target}" "chmod -R ${REMOTE_CHMOD} '${remote_target}'"
scp "${SSH_KEY_OPT[@]}" -P "${MIRROR_PORT}" "${INSTALL_SCRIPT_SOURCE}" "${ssh_target}:${REMOTE_INSTALL_SCRIPT_PATH}"
ssh "${SSH_KEY_OPT[@]}" -p "${MIRROR_PORT}" "${ssh_target}" "chmod ${REMOTE_CHMOD%% *} '${REMOTE_INSTALL_SCRIPT_PATH}'"

if [ "${REMOTE_LATEST_LINK}" = "1" ]; then
  ssh "${SSH_KEY_OPT[@]}" -p "${MIRROR_PORT}" "${ssh_target}" "ln -sfn '${remote_target}' '${remote_latest}' && chmod -h ${REMOTE_CHMOD%% *} '${remote_latest}' 2>/dev/null || true"
fi

echo ""
echo "Mirror upload complete."
echo "Remote path: ${remote_target}"
echo "Public install script: ${REMOTE_INSTALL_SCRIPT_PATH}"
if [ -n "${MIRROR_BASE_URL}" ]; then
  echo "Download base: ${MIRROR_BASE_URL%/}/${VERSION}"
  echo "Manifest download_base set to: ${MIRROR_BASE_URL%/}"
fi
