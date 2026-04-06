#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-}"
NOTES="${2:-"Release $VERSION"}"

if [ -z "$VERSION" ]; then
  echo "Usage: ./scripts/release.sh <version> [notes]"
  echo "Example: ./scripts/release.sh v0.3.0 \"Five-story training demo\""
  exit 1
fi

PLATFORMS=(
  "linux/amd64"
  "linux/arm64"
  "darwin/amd64"
  "darwin/arm64"
  "windows/amd64"
)

BUILD_DIR="dist"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "Building $VERSION for ${#PLATFORMS[@]} platforms..."

for platform in "${PLATFORMS[@]}"; do
  GOOS="${platform%/*}"
  GOARCH="${platform#*/}"
  output="mscli-${GOOS}-${GOARCH}"
  if [ "$GOOS" = "windows" ]; then
    output="${output}.exe"
  fi
  echo "  -> $output"
  GOOS="$GOOS" GOARCH="$GOARCH" go build -ldflags "-X github.com/mindspore-lab/mindspore-cli/internal/version.Version=${VERSION#v}" -o "${BUILD_DIR}/${output}" ./cmd/mscli/
done

SERVER_GOOS="$(go env GOOS)"
SERVER_GOARCH="$(go env GOARCH)"
SERVER_OUTPUT="mscli-server-${SERVER_GOOS}-${SERVER_GOARCH}"
if [ "${SERVER_GOOS}" = "windows" ]; then
  SERVER_OUTPUT="${SERVER_OUTPUT}.exe"
fi

echo "  -> ${SERVER_OUTPUT}"
CGO_ENABLED=1 GOOS="${SERVER_GOOS}" GOARCH="${SERVER_GOARCH}" go build -ldflags "-X github.com/mindspore-lab/mindspore-cli/internal/version.Version=${VERSION#v}" -o "${BUILD_DIR}/${SERVER_OUTPUT}" ./cmd/mscli-server/

echo ""
echo "Built binaries:"
ls -lh "$BUILD_DIR"

# Generate manifest.json
PLAIN_VERSION="${VERSION#v}"
cat > "${BUILD_DIR}/manifest.json" <<MANIFEST
{
  "latest": "${PLAIN_VERSION}",
  "min_allowed": "",
  "download_base": "https://github.com/mindspore-lab/mindspore-cli/releases/download"
}
MANIFEST
echo "Generated manifest.json"

echo ""
echo "Creating GitHub release $VERSION..."
gh release create "$VERSION" "$BUILD_DIR"/* \
  --title "$VERSION" \
  --notes "$NOTES"

echo ""
echo "Done! https://github.com/mindspore-lab/mindspore-cli/releases/tag/$VERSION"
