#!/bin/sh
# Refresh installed images, or rebuild source with explicit --build.
set -eu
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"
case "${1:-}" in
  --summary|--speech) exec ./scripts/docker-install.sh "$@" ;;
  *) echo 'Usage: ./scripts/docker-update.sh --summary|--speech [--build]' >&2; exit 2 ;;
esac
