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

MIRROR_ROOT="${MSCODE_MIRROR_ROOT:-/opt/downloads/mscode/releases}"
MIRROR_BASE_URL="${MSCODE_MIRROR_BASE_URL:-http://47.115.175.134/mscode/releases}"
TARGET_DIR="${MIRROR_ROOT}/${VERSION}"
LATEST_LINK="${MIRROR_ROOT}/latest"
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
  output="mscode-${GOOS}-${GOARCH}"
  if [ "${GOOS}" = "windows" ]; then
    output="${output}.exe"
  fi
  echo "  -> ${output}"
  GOOS="${GOOS}" GOARCH="${GOARCH}" go build \
    -ldflags "-X github.com/vigo999/mindspore-code/internal/version.Version=${PLAIN_VERSION}" \
    -o "${BUILD_DIR}/${output}" \
    ./cmd/mscode/
done

SERVER_GOOS="$(go env GOOS)"
SERVER_GOARCH="$(go env GOARCH)"
SERVER_OUTPUT="mscode-server-${SERVER_GOOS}-${SERVER_GOARCH}"
if [ "${SERVER_GOOS}" = "windows" ]; then
  SERVER_OUTPUT="${SERVER_OUTPUT}.exe"
fi

echo "  -> ${SERVER_OUTPUT}"
GOOS="${SERVER_GOOS}" GOARCH="${SERVER_GOARCH}" go build \
  -ldflags "-X github.com/vigo999/mindspore-code/internal/version.Version=${PLAIN_VERSION}" \
  -o "${BUILD_DIR}/${SERVER_OUTPUT}" \
  ./cmd/mscode-server/

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
sudo chmod -R a+rX "${TARGET_DIR}"
sudo ln -sfn "${TARGET_DIR}" "${LATEST_LINK}"

echo ""
echo "Release assets ready:"
echo "  ${TARGET_DIR}"
echo "Latest link:"
echo "  ${LATEST_LINK} -> ${TARGET_DIR}"
echo "Manifest download_base:"
echo "  ${MIRROR_BASE_URL}"
