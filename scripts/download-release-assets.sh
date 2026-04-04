#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-}"
OUTPUT_DIR="${2:-}"

if [ -z "${VERSION}" ]; then
  echo "Usage: ./scripts/download-release-assets.sh <version> [output-dir]"
  echo "Example: ./scripts/download-release-assets.sh v0.5.0-beta.2"
  exit 1
fi

if [[ "${VERSION}" != v* ]]; then
  echo "Error: version must include a leading v, for example v0.5.0-beta.2" >&2
  exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
  OUTPUT_DIR="${VERSION}"
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl is required" >&2
  exit 1
fi

BASE_URL="https://github.com/vigo999/mindspore-code/releases/download/${VERSION}"

assets=(
  "manifest.json"
  "mscli-linux-amd64"
  "mscli-linux-arm64"
  "mscli-darwin-amd64"
  "mscli-darwin-arm64"
  "mscli-windows-amd64.exe"
  "mscli-server-linux-amd64"
)

mkdir -p "${OUTPUT_DIR}"

echo "Downloading release assets for ${VERSION} into ${OUTPUT_DIR}"

for asset in "${assets[@]}"; do
  echo "  -> ${asset}"
  curl -fL "${BASE_URL}/${asset}" -o "${OUTPUT_DIR}/${asset}"
done

echo ""
echo "Downloaded ${#assets[@]} assets into ${OUTPUT_DIR}"
