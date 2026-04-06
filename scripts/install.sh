#!/usr/bin/env bash
set -euo pipefail

GITHUB_REPO="${MSCLI_GITHUB_REPO:-mindspore-lab/mindspore-cli}"
GITCODE_REPO="${MSCLI_GITCODE_REPO:-mindspore-lab/mindspore-cli}"
MIRROR_BASE_URL="${MSCLI_MIRROR_BASE_URL:-https://mscli.dev/mscli/releases}"
MIRROR_MANIFEST_URL="${MSCLI_MIRROR_MANIFEST_URL:-${MIRROR_BASE_URL%/}/latest/manifest.json}"
GITCODE_BASE_URL="https://gitcode.com/api/v5/repos/${GITCODE_REPO}/releases"
INSTALL_DIR="$HOME/.mscli/bin"
BINARY_NAME="mscli"
INSTALL_SOURCE="${MSCLI_INSTALL_SOURCE:-auto}"
REQUESTED_VERSION="${MSCLI_VERSION:-}"
PROBE_BYTES="${MSCLI_INSTALL_PROBE_BYTES:-262144}"
PROBE_TIMEOUT="${MSCLI_INSTALL_PROBE_TIMEOUT:-8}"
CONNECT_TIMEOUT="${MSCLI_INSTALL_CONNECT_TIMEOUT:-5}"
GITHUB_API="https://api.github.com/repos/${GITHUB_REPO}/releases/latest"

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

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command not found: $1" >&2
    exit 1
  fi
}

need_cmd curl
need_cmd perl

fetch_json() {
  local url="$1"
  curl -fsSL \
    --connect-timeout "$CONNECT_TIMEOUT" \
    --retry 2 \
    --retry-delay 1 \
    -H "Accept: application/json" \
    "$url" </dev/null
}

json_release_tag() {
  perl -MJSON::PP -e '
    use strict;
    use warnings;
    local $/;
    my $payload = <STDIN>;
    my $json = decode_json($payload);
    print($json->{tag_name} // q());
  '
}

json_manifest_version() {
  perl -MJSON::PP -e '
    use strict;
    use warnings;
    local $/;
    my $payload = <STDIN>;
    my $json = decode_json($payload);
    print($json->{latest} // q());
  '
}

normalize_tag() {
  local version="$1"
  case "$version" in
    v*) printf '%s\n' "$version" ;;
    *) printf 'v%s\n' "$version" ;;
  esac
}

latest_from_manifest() {
  local url="$1"
  local version

  version="$(fetch_json "$url" | json_manifest_version)"
  if [ -z "$version" ]; then
    return 1
  fi
  normalize_tag "$version"
}

latest_from_github() {
  local tag

  tag="$(fetch_json "$GITHUB_API" | json_release_tag)"
  if [ -z "$tag" ]; then
    return 1
  fi
  normalize_tag "$tag"
}

latest_from_gitcode() {
  local tag

  tag="$(fetch_json "${GITCODE_BASE_URL}/latest" | json_release_tag)"
  if [ -z "$tag" ]; then
    return 1
  fi
  normalize_tag "$tag"
}

resolve_latest() {
  local latest=""

  case "$INSTALL_SOURCE" in
    auto)
      echo "Resolving latest release from mirror..." >&2
      latest="$(latest_from_manifest "$MIRROR_MANIFEST_URL" 2>/dev/null || true)"
      if [ -n "$latest" ]; then
        printf '%s\n' "$latest"
        return 0
      fi
      echo "Mirror manifest unavailable, trying GitCode..." >&2
      latest="$(latest_from_gitcode 2>/dev/null || true)"
      if [ -n "$latest" ]; then
        printf '%s\n' "$latest"
        return 0
      fi
      echo "GitCode unavailable, falling back to GitHub..." >&2
      latest="$(latest_from_github 2>/dev/null || true)"
      ;;
    mirror)
      echo "Resolving latest release from mirror..." >&2
      latest="$(latest_from_manifest "$MIRROR_MANIFEST_URL" 2>/dev/null || true)"
      ;;
    gitcode)
      echo "Resolving latest release from GitCode..." >&2
      latest="$(latest_from_gitcode 2>/dev/null || true)"
      ;;
    github)
      echo "Resolving latest release from GitHub..." >&2
      latest="$(latest_from_github 2>/dev/null || true)"
      ;;
    *)
      echo "Error: unsupported MSCLI_INSTALL_SOURCE=${INSTALL_SOURCE} (expected auto|github|mirror|gitcode)" >&2
      exit 1
      ;;
  esac

  if [ -z "$latest" ]; then
    return 1
  fi
  printf '%s\n' "$latest"
}

probe_speed() {
  local url="$1"
  local bytes_end
  local speed

  bytes_end=$((PROBE_BYTES - 1))
  speed="$(
    curl -fsSL \
      --connect-timeout "$CONNECT_TIMEOUT" \
      --max-time "$PROBE_TIMEOUT" \
      --range "0-${bytes_end}" \
      --output /dev/null \
      --write-out '%{speed_download}' \
      "$url" 2>/dev/null || true
  )"

  case "$speed" in
    ""|0|0.0|0.000)
      return 1
      ;;
  esac

  printf '%s\n' "$speed"
}

format_speed() {
  local raw="$1"
  awk -v value="$raw" '
    BEGIN {
      split("B/s KiB/s MiB/s GiB/s", units, " ");
      unit = 1;
      while (value >= 1024 && unit < 4) {
        value /= 1024;
        unit++;
      }
      printf "%.1f %s", value, units[unit];
    }
  '
}

pick_source() {
  local best_name=""
  local best_url=""
  local best_speed=""
  local speed=""
  local provider=""
  local url=""

  for provider in "$@"; do
    case "$provider" in
      github)
        url="$GITHUB_URL"
        ;;
      gitcode)
        url="$GITCODE_URL"
        ;;
      mirror)
        url="$MIRROR_URL"
        ;;
      *)
        continue
        ;;
    esac

    if [ -z "$url" ]; then
      continue
    fi

    echo "Probing ${provider}..." >&2
    if ! speed="$(probe_speed "$url")"; then
      echo "  ${provider}: unavailable" >&2
      continue
    fi

    echo "  ${provider}: $(format_speed "$speed")" >&2
    if [ -z "$best_speed" ] || awk -v left="$speed" -v right="$best_speed" 'BEGIN { exit !(left > right) }'; then
      best_name="$provider"
      best_url="$url"
      best_speed="$speed"
    fi
  done

  if [ -n "$best_name" ]; then
    printf '%s\n%s\n%s\n' "$best_name" "$best_url" "$best_speed"
    return 0
  fi
  return 1
}

if [ -n "$REQUESTED_VERSION" ]; then
  LATEST="$(normalize_tag "$REQUESTED_VERSION")"
  echo "Using requested release: ${LATEST}"
else
  if ! LATEST="$(resolve_latest)"; then
    echo "Error: could not determine latest release" >&2
    exit 1
  fi
fi

echo "Latest release: ${LATEST}"

ASSET="mscli-${OS}-${ARCH}"
GITHUB_URL="https://github.com/${GITHUB_REPO}/releases/download/${LATEST}/${ASSET}"
GITCODE_URL="https://gitcode.com/${GITCODE_REPO}/releases/download/${LATEST}/${ASSET}"
MIRROR_URL="${MIRROR_BASE_URL%/}/${LATEST}/${ASSET}"

case "$INSTALL_SOURCE" in
  auto)
    if ! selection="$(pick_source mirror gitcode github)"; then
      echo "Error: could not reach any release download source" >&2
      exit 1
    fi
    ;;
  github)
    if ! selection="$(pick_source github)"; then
      echo "Error: GitHub release download is unavailable" >&2
      exit 1
    fi
    ;;
  gitcode)
    if ! selection="$(pick_source gitcode)"; then
      echo "Error: GitCode release download is unavailable" >&2
      exit 1
    fi
    ;;
  mirror)
    if ! selection="$(pick_source mirror)"; then
      echo "Error: mirror release download is unavailable for tag ${LATEST}" >&2
      exit 1
    fi
    ;;
esac

SELECTED_SOURCE="$(printf '%s' "$selection" | sed -n '1p')"
URL="$(printf '%s' "$selection" | sed -n '2p')"
SELECTED_SPEED="$(printf '%s' "$selection" | sed -n '3p')"

# Download binary.
echo "Downloading ${ASSET} from ${SELECTED_SOURCE} ($(format_speed "$SELECTED_SPEED"))..."
mkdir -p "$INSTALL_DIR"
curl -fSL -o "${INSTALL_DIR}/${BINARY_NAME}" "$URL" </dev/null
chmod +x "${INSTALL_DIR}/${BINARY_NAME}"

echo ""
echo "Installed mscli ${LATEST} to ${INSTALL_DIR}/${BINARY_NAME}"

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
  echo "Added mscli to PATH in ${PROFILE}"
fi
echo ""
echo "Run: source ${PROFILE} && mscli"
