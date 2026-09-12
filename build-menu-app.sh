#!/bin/sh
set -eu
ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
APP="$ROOT_DIR/Meeting Transcript.app"
mkdir -p "$APP/Contents/MacOS"
mkdir -p "$ROOT_DIR/work/swift-module-cache"
cp "$ROOT_DIR/macos/Info.plist" "$APP/Contents/Info.plist"
xcrun swiftc -O -module-cache-path "$ROOT_DIR/work/swift-module-cache" -framework AppKit "$ROOT_DIR/macos/MeetingTranscriptStatus.swift" -o "$APP/Contents/MacOS/MeetingTranscriptStatus"
codesign --force --sign - "$APP" >/dev/null
echo "Создано: $APP"
