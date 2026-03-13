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
  output="ms-cli-${GOOS}-${GOARCH}"
  if [ "$GOOS" = "windows" ]; then
    output="${output}.exe"
  fi
  echo "  -> $output"
  GOOS="$GOOS" GOARCH="$GOARCH" go build -o "${BUILD_DIR}/${output}" ./cmd/ms-cli/
done

echo ""
echo "Built binaries:"
ls -lh "$BUILD_DIR"

echo ""
echo "Creating GitHub release $VERSION..."
gh release create "$VERSION" "$BUILD_DIR"/* \
  --title "$VERSION" \
  --notes "$NOTES"

echo ""
echo "Done! https://github.com/vigo999/ms-cli/releases/tag/$VERSION"
