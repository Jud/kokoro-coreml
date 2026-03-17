#!/bin/bash
set -euo pipefail

REPO="Jud/kokoro-tts-swift"
ASSET="kokoro-models.tar.gz"
DEFAULT_DIR="$HOME/Library/Application Support/kokoro-tts/models/kokoro"
DEST="${1:-$DEFAULT_DIR}"

if [ -d "$DEST/voices" ]; then
    echo "Models already present at: $DEST"
    exit 0
fi

# Find latest models-* release
TAG=$(gh api "repos/$REPO/releases" --jq '[.[] | select(.tag_name | startswith("models-"))][0].tag_name' 2>/dev/null || echo "")
if [ -z "$TAG" ]; then
    echo "Could not fetch latest release tag, falling back to models-v1"
    TAG="models-v1"
fi

URL="https://github.com/$REPO/releases/download/$TAG/$ASSET"

echo "Downloading KokoroTTS models ($TAG)..."
echo "  from: $URL"
echo "  to:   $DEST"

mkdir -p "$DEST"
curl -L --progress-bar "$URL" | tar xz -C "$DEST"

echo "Done. Models installed to: $DEST"
