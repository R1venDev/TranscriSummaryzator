#!/bin/sh
# Fresh isolated deployment. Idempotent: never overwrites stored keys/config.
set -eu
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"
MODE=--speech
BUILD=0
ENABLE_SPEECH=0
for option in "$@"; do
  case "$option" in
    --summary|--speech) MODE=$option ;;
    --build) BUILD=1 ;;
    --enable-speech) ENABLE_SPEECH=1 ;;
    *) echo 'Usage: ./scripts/docker-install.sh [--summary|--speech] [--build] [--enable-speech]' >&2; exit 2 ;;
  esac
done
case "$MODE" in
  --summary) set -- -f compose.yaml ;;
  --speech) set -- -f compose.yaml -f compose.speech.yaml ;;
  *) echo 'Usage: ./scripts/docker-install.sh [--summary|--speech]' >&2; exit 2 ;;
esac
if test "$ENABLE_SPEECH" = 1 && test "$MODE" != --speech; then
  echo '--enable-speech requires --speech.' >&2
  exit 2
fi
TRANSCRI_IMAGE=${TRANSCRI_IMAGE:-ghcr.io/r1vendev/transcrisummaryzator:summary}
TRANSCRI_SPEECH_IMAGE=${TRANSCRI_SPEECH_IMAGE:-ghcr.io/r1vendev/transcrisummaryzator:speech}
export TRANSCRI_IMAGE TRANSCRI_SPEECH_IMAGE
command -v docker >/dev/null 2>&1 || { echo 'Install Docker Engine/Desktop and Docker Compose first.' >&2; exit 1; }
docker compose version >/dev/null
if test -z "${TRANSCRI_SOURCE_REVISION:-}"; then
  TRANSCRI_SOURCE_REVISION=$(git rev-parse HEAD 2>/dev/null || echo unknown)
  if test -n "$(git status --porcelain 2>/dev/null)"; then
    TRANSCRI_SOURCE_REVISION="${TRANSCRI_SOURCE_REVISION}-dirty"
  fi
  export TRANSCRI_SOURCE_REVISION
fi
docker compose "$@" config --quiet
if test "$BUILD" = 1; then
  # Local edits get local tags, never overwrite the registry image name locally.
  TRANSCRI_IMAGE=${TRANSCRI_BUILD_SUMMARY_IMAGE:-transcrisummaryzator:local-summary}
  TRANSCRI_SPEECH_IMAGE=${TRANSCRI_BUILD_SPEECH_IMAGE:-transcrisummaryzator:local-speech}
  export TRANSCRI_IMAGE TRANSCRI_SPEECH_IMAGE
  docker compose "$@" build app scheduler
else
  docker compose "$@" pull app scheduler
fi
if test "$MODE" = --speech; then
  summary_revision=$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$TRANSCRI_IMAGE")
  speech_revision=$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$TRANSCRI_SPEECH_IMAGE")
  if test -z "$summary_revision" || test "$summary_revision" = '<no value>' || test "$summary_revision" != "$speech_revision"; then
    echo 'App and scheduler image revisions differ. Retry after publication completes, or pin matching SHA/version tags.' >&2
    exit 1
  fi
fi
if test "$ENABLE_SPEECH" = 1; then
  docker compose "$@" run --rm --no-deps --pull never bootstrap bootstrap --enable-speech
else
  docker compose "$@" run --rm --no-deps --pull never bootstrap
fi
docker compose "$@" up -d --no-build --pull never --wait app scheduler
echo "Open ${TRANSCRI_PUBLIC_ORIGIN:-http://127.0.0.1:${TRANSCRI_PORT:-8765}}"
if test "$MODE" = --summary; then
  echo 'Summary profile: web UI and summary queue only; speech processing needs --speech.'
fi
