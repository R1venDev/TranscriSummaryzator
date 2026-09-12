#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST="$PLIST_DIR/local.meeting-transcript.watcher.plist"
mkdir -p "$PLIST_DIR"

sed "s|__ROOT__|$ROOT_DIR|g" "$ROOT_DIR/launchd/local.meeting-transcript.watcher.plist.template" > "$PLIST"
launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl enable "gui/$(id -u)/local.meeting-transcript.watcher"
echo "Автозапуск установлен: $PLIST"

