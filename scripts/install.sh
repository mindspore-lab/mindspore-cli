#!/usr/bin/env bash
set -euo pipefail

REPO="vigo999/ms-cli"
INSTALL_DIR="$HOME/.ms-cli/bin"
BINARY_NAME="mscli"

# Detect OS.
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "$OS" in
  linux)  OS="linux" ;;
  darwin) OS="darwin" ;;
  *)
    echo "Error: unsupported OS: $OS" >&2
    exit 1
    ;;
esac

# Detect architecture.
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) ARCH="amd64" ;;
  aarch64|arm64) ARCH="arm64" ;;
  *)
    echo "Error: unsupported architecture: $ARCH" >&2
    exit 1
    ;;
esac

echo "Detected: ${OS}/${ARCH}"

# Fetch latest release tag.
echo "Fetching latest release..."
LATEST="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" </dev/null | tr -d ' ,' | grep '"tag_name"' | cut -d'"' -f4)"

if [ -z "$LATEST" ]; then
  echo "Error: could not determine latest release" >&2
  exit 1
fi

echo "Latest release: ${LATEST}"

# Build download URL.
ASSET="ms-cli-${OS}-${ARCH}"
URL="https://github.com/${REPO}/releases/download/${LATEST}/${ASSET}"

# Download binary.
echo "Downloading ${URL}..."
mkdir -p "$INSTALL_DIR"
curl -fSL -o "${INSTALL_DIR}/${BINARY_NAME}" "$URL" </dev/null
chmod +x "${INSTALL_DIR}/${BINARY_NAME}"

echo ""
echo "Installed ms-cli ${LATEST} to ${INSTALL_DIR}/${BINARY_NAME}"

# Auto-add to PATH.
PATH_LINE="export PATH=\"${INSTALL_DIR}:\$PATH\""

# Detect shell profile.
CURRENT_SHELL="$(basename "${SHELL:-bash}")"
case "$CURRENT_SHELL" in
  zsh)  PROFILE="$HOME/.zshrc" ;;
  bash)
    if [ -f "$HOME/.bash_profile" ]; then
      PROFILE="$HOME/.bash_profile"
    else
      PROFILE="$HOME/.bashrc"
    fi
    ;;
  *)    PROFILE="$HOME/.profile" ;;
esac

# Add to profile if not already there.
if [ -f "$PROFILE" ] && grep -qF "$INSTALL_DIR" "$PROFILE" 2>/dev/null; then
  echo ""
  echo "PATH already configured in ${PROFILE}"
else
  echo "$PATH_LINE" >> "$PROFILE"
  echo ""
  echo "Added ms-cli to PATH in ${PROFILE}"
fi
echo ""
echo "Run: source ${PROFILE} && mscli"
