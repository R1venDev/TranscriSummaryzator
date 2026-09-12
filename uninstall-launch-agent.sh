#!/bin/sh
set -eu
PLIST="$HOME/Library/LaunchAgents/local.meeting-transcript.watcher.plist"
launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
if [ -f "$PLIST" ]; then
  mv "$PLIST" "$HOME/.Trash/local.meeting-transcript.watcher.plist"
fi
echo "Автозапуск отключён."

