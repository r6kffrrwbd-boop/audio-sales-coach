#!/usr/bin/env bash
# Remove audio-sales-coach.
#   ./uninstall.sh               remove venvs, models, skill and command; KEEP voice profiles and feedback
#   ./uninstall.sh --purge-data  also delete voice profiles, feedback and the config file
# Reports you wrote elsewhere (--output folders) are not touched; delete them yourself.
set -euo pipefail
DATA_DIR="${ASC_DATA_DIR:-$HOME/.local/share/audio-sales-coach}"
SKILL_DIR="${ASC_SKILL_DIR:-$HOME/.claude/skills/audio-sales-coach}"
BIN_DIR="${ASC_BIN_DIR:-$HOME/.local/bin}"
CONFIG="${ASC_CONFIG:-$HOME/.config/audio-sales-coach/config.json}"
PURGE=0
for a in "$@"; do case "$a" in --purge-data) PURGE=1 ;; -h|--help) sed -n '2,5p' "$0"; exit 0 ;; *) echo "unknown option $a"; exit 2 ;; esac; done

LINK="$BIN_DIR/audio-sales-coach"
[ -L "$LINK" ] && rm -f "$LINK" && echo "removed $LINK"
if [ -f "$SKILL_DIR/.installed-by-audio-sales-coach" ]; then rm -rf "$SKILL_DIR"; echo "removed $SKILL_DIR"; fi
for d in venv tf-venv models eval-cache; do [ -e "$DATA_DIR/$d" ] && rm -rf "${DATA_DIR:?}/$d" && echo "removed $DATA_DIR/$d"; done
if [ "$PURGE" = 1 ]; then
  for d in voice-profiles feedback; do [ -e "$DATA_DIR/$d" ] && rm -rf "${DATA_DIR:?}/$d" && echo "deleted $DATA_DIR/$d"; done
  [ -f "$CONFIG" ] && rm -f "$CONFIG" && echo "deleted $CONFIG"
  rmdir "$DATA_DIR" 2>/dev/null || true
else
  [ -d "$DATA_DIR/voice-profiles" ] && echo "kept voice profiles in $DATA_DIR/voice-profiles (use --purge-data to delete)"
  [ -d "$DATA_DIR/feedback" ] && echo "kept feedback in $DATA_DIR/feedback (use --purge-data to delete)"
fi
echo "done"
