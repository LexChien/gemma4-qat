#!/usr/bin/env bash
# Clone google-deepmind/gemma at the revision used by this project.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/src/gemma"
COMMIT="9e1f74131ba9d45e5fe6ae7bb3941d9f2abb39c3"
URL="https://github.com/google-deepmind/gemma.git"

if [ -d "$DEST/.git" ]; then
  git -C "$DEST" fetch --tags origin
  git -C "$DEST" checkout "$COMMIT"
  echo "Updated $DEST to $COMMIT"
  exit 0
fi

if [ -e "$DEST/README.md" ] && [ ! -d "$DEST/.git" ]; then
  # Publish tree only has our pointer README; clone into a temp then replace others.
  tmp="$(mktemp -d)"
  git clone "$URL" "$tmp/gemma"
  git -C "$tmp/gemma" checkout "$COMMIT"
  # Keep our pointer README, copy the rest beside it.
  rsync -a --exclude 'README.md' "$tmp/gemma/" "$DEST/"
  # Also keep a hidden .git so later updates work; place it under DEST.
  rm -rf "$DEST/.git"
  mv "$tmp/gemma/.git" "$DEST/.git"
  rm -rf "$tmp"
  # Restore our pointer file name if upstream README overwrote nothing (we excluded it).
  echo "Cloned $URL @ $COMMIT into $DEST (project README.md kept)."
  exit 0
fi

git clone "$URL" "$DEST"
git -C "$DEST" checkout "$COMMIT"
echo "Cloned $URL @ $COMMIT into $DEST"
