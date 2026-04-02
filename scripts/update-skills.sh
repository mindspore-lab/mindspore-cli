#!/usr/bin/env bash
set -euo pipefail

# Configuration — change these to switch skills source.
SKILLS_REPO="https://github.com/vigo999/mindspore-skills"
SKILLS_BRANCH="main"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILTIN_DIR="$PROJECT_ROOT/integrations/skills/builtin"
TMP_DIR="$(mktemp -d)"

trap 'rm -rf "$TMP_DIR"' EXIT

echo "Pulling skills from $SKILLS_REPO@$SKILLS_BRANCH..."
git clone --depth 1 -b "$SKILLS_BRANCH" "$SKILLS_REPO" "$TMP_DIR/repo"

echo "Updating $BUILTIN_DIR..."
rm -rf "$BUILTIN_DIR"
mkdir -p "$BUILTIN_DIR"
cp -r "$TMP_DIR/repo/skills/"* "$BUILTIN_DIR/"

count=$(find "$BUILTIN_DIR" -name "SKILL.md" | wc -l)
echo "Done. $count skills copied."
