#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

VERSION="${1:-}"

if [ -z "${VERSION}" ]; then
  echo "Usage: ./scripts/build-release-assets.sh <version>"
  echo "Example: ./scripts/build-release-assets.sh v0.5.0-beta.2"
  exit 1
fi

if [[ "${VERSION}" != v* ]]; then
  echo "Error: version must include a leading v, for example v0.5.0-beta.2" >&2
  exit 1
fi

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command not found: $1" >&2
    exit 1
  fi
}

need_cmd go
need_cmd mktemp
need_cmd sudo

PLATFORMS=(
  "linux/amd64"
  "linux/arm64"
  "darwin/amd64"
  "darwin/arm64"
  "windows/amd64"
)

[ -f "${SCRIPT_DIR}/mirror.conf" ] && source "${SCRIPT_DIR}/mirror.conf"
MIRROR_ROOT="${MSCLI_MIRROR_ROOT:-/opt/downloads/mscli/releases}"
MIRROR_BASE_URL="${MSCLI_MIRROR_BASE_URL:-http://47.115.175.134/mscli/releases}"
TARGET_DIR="${MIRROR_ROOT}/${VERSION}"
LATEST_LINK="${MIRROR_ROOT}/latest"
PUBLIC_ROOT="$(dirname "${MIRROR_ROOT}")"
INSTALL_SCRIPT_PATH="${PUBLIC_ROOT}/install.sh"
PLAIN_VERSION="${VERSION#v}"
BUILD_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${BUILD_DIR}"
}
trap cleanup EXIT

echo "Building ${VERSION} into temporary directory ${BUILD_DIR}"

cd "${REPO_ROOT}"

for platform in "${PLATFORMS[@]}"; do
  GOOS="${platform%/*}"
  GOARCH="${platform#*/}"
  output="mscli-${GOOS}-${GOARCH}"
  if [ "${GOOS}" = "windows" ]; then
    output="${output}.exe"
  fi
  echo "  -> ${output}"
  GOOS="${GOOS}" GOARCH="${GOARCH}" go build \
    -ldflags "-X github.com/mindspore-lab/mindspore-cli/internal/version.Version=${PLAIN_VERSION}" \
    -o "${BUILD_DIR}/${output}" \
    ./cmd/mscli/
done

SERVER_GOOS="$(go env GOOS)"
SERVER_GOARCH="$(go env GOARCH)"
SERVER_OUTPUT="mscli-server-${SERVER_GOOS}-${SERVER_GOARCH}"
if [ "${SERVER_GOOS}" = "windows" ]; then
  SERVER_OUTPUT="${SERVER_OUTPUT}.exe"
fi

echo "  -> ${SERVER_OUTPUT}"
GOOS="${SERVER_GOOS}" GOARCH="${SERVER_GOARCH}" go build \
  -ldflags "-X github.com/mindspore-lab/mindspore-cli/internal/version.Version=${PLAIN_VERSION}" \
  -o "${BUILD_DIR}/${SERVER_OUTPUT}" \
  ./cmd/mscli-server/

cat > "${BUILD_DIR}/manifest.json" <<MANIFEST
{
  "latest": "${PLAIN_VERSION}",
  "min_allowed": "",
  "download_base": "${MIRROR_BASE_URL}"
}
MANIFEST

echo ""
echo "Installing assets to ${TARGET_DIR}"
sudo mkdir -p "${TARGET_DIR}"
sudo cp "${BUILD_DIR}"/* "${TARGET_DIR}/"
sudo cp "${REPO_ROOT}/scripts/install.sh" "${INSTALL_SCRIPT_PATH}"
sudo chmod -R a+rX "${TARGET_DIR}"
sudo chmod a+rX "${INSTALL_SCRIPT_PATH}"
sudo ln -sfn "${TARGET_DIR}" "${LATEST_LINK}"

echo ""
echo "Release assets ready:"
echo "  ${TARGET_DIR}"
echo "Latest link:"
echo "  ${LATEST_LINK} -> ${TARGET_DIR}"
echo "Public install script:"
echo "  ${INSTALL_SCRIPT_PATH}"
echo "Manifest download_base:"
echo "  ${MIRROR_BASE_URL}"
