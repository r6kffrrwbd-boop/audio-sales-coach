#!/usr/bin/env bash
# Idempotent installer for audio-sales-coach.
#
#   ./install.sh                         default CPU install + models + Claude Code skill + self-test
#   ./install.sh --with-transcription    also install faster-whisper (optional words; model downloads on first use)
#   ./install.sh --with-ced              also fetch the optional CED-mini audio tagger (48 MB) for cross-checks
#   ./install.sh --with-tensorflow       separate TF 2.16.2 venv (~1.3 GB) only for scripts/compare_yamnet_tf.py
#   ./install.sh --no-skill              CLI only (no ~/.claude/skills entry)
#   ./install.sh --no-selftest           skip the self-test
#
# Creates (nothing else is touched):
#   ~/.local/share/audio-sales-coach/venv           isolated Python environment (pinned deps, no TensorFlow)
#   ~/.local/share/audio-sales-coach/models         verified model files (~44 MB by default)
#   ~/.local/share/audio-sales-coach/voice-profiles (created on first enrolment, 0700)
#   ~/.claude/skills/audio-sales-coach/             Claude Code skill (SKILL.md, references/, bin/)
#   ~/.local/bin/audio-sales-coach                  symlink to the launcher
# Override with ASC_DATA_DIR, ASC_SKILL_DIR, ASC_BIN_DIR, ASC_PYTHON.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${ASC_DATA_DIR:-$HOME/.local/share/audio-sales-coach}"
VENV="$DATA_DIR/venv"
TF_VENV="$DATA_DIR/tf-venv"
SKILL_DIR="${ASC_SKILL_DIR:-$HOME/.claude/skills/audio-sales-coach}"
BIN_DIR="${ASC_BIN_DIR:-$HOME/.local/bin}"
MARKER=".installed-by-audio-sales-coach"
SKILL=1; SELFTEST=1; TX=0; CED=0; TF=0
for a in "$@"; do
  case "$a" in
    --with-transcription) TX=1 ;;
    --with-ced) CED=1 ;;
    --with-tensorflow) TF=1 ;;
    --no-skill) SKILL=0 ;;
    --no-selftest) SELFTEST=0 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "error: unknown option $a"; echo "hint: ./install.sh --help"; exit 2 ;;
  esac
done

say() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$1" >&2; [ -n "${2:-}" ] && printf 'hint: %s\n' "$2" >&2; exit "${3:-1}"; }

# 0. system prerequisites
command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1 || die "ffmpeg/ffprobe not found" "brew install ffmpeg" 3
command -v curl >/dev/null 2>&1 || die "curl not found" "install curl" 3
say "machine: $(uname -s) $(uname -m); ffmpeg $(ffmpeg -version | head -1 | awk '{print $3}')"

# 1. Python 3.10-3.12 (the pinned wheels are verified on 3.12)
PY="${ASC_PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' 2>/dev/null; then
      PY="$(command -v "$c")"; break
    fi
  done
fi
[ -n "$PY" ] || die "Python 3.10-3.12 not found" "brew install python@3.12, or set ASC_PYTHON=/path/to/python3.12"
say "Python: $PY ($("$PY" -c 'import platform; print(platform.python_version())'))"

# 2. isolated environment
mkdir -p "$DATA_DIR"; chmod 700 "$DATA_DIR"
if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c 'import sys' >/dev/null 2>&1; then
  say "creating virtual environment at $VENV"
  rm -rf "$VENV"
  "$PY" -m venv "$VENV"
else
  say "reusing virtual environment at $VENV"
fi
PIP=("$VENV/bin/python" -m pip install --quiet --disable-pip-version-check)
"${PIP[@]}" --upgrade "pip>=25.1" >/dev/null
RETRY=(--retries 10 --timeout 60)
"$VENV/bin/python" -m pip help install 2>/dev/null | grep -q -- --resume-retries && RETRY+=(--resume-retries 20)
say "installing pinned dependencies (requirements/base.lock)"
"${PIP[@]}" "${RETRY[@]}" -r "$REPO/requirements/base.lock"
say "installing audio-sales-coach from $REPO"
# build from a staged copy so the repository is never written to (no stale build/ dirs)
STAGE="$DATA_DIR/.build-src"; BTMP="$DATA_DIR/.tmp"
rm -rf "$STAGE" "$BTMP"; mkdir -p "$STAGE" "$BTMP"
cp -R "$REPO/pyproject.toml" "$REPO/README.md" "$REPO/LICENSE" "$REPO/src" "$STAGE/"
TMPDIR="$BTMP" "${PIP[@]}" --no-deps --force-reinstall "$STAGE" >/dev/null
rm -rf "$STAGE" "$BTMP"
if [ "$TX" = 1 ]; then
  say "installing optional transcription (requirements/transcription.lock)"
  "${PIP[@]}" "${RETRY[@]}" -r "$REPO/requirements/transcription.lock"
fi
CLI="$VENV/bin/audio-sales-coach"

# 3. models (verified by size and SHA-256; resumable)
say "fetching default models (YAMNet 15 MB, Silero VAD 0.6 MB, CAM++ speaker embeddings 28 MB)"
EXTRA=(); [ "$CED" = 1 ] && EXTRA=(--with ced_mini)
ASC_DATA_DIR="$DATA_DIR" "$CLI" models fetch ${EXTRA[@]+"${EXTRA[@]}"}
if [ "$TX" = 1 ]; then
  say "fetching faster-whisper '${ASC_WHISPER:-small}' model (resumable; ~480 MB for small, set ASC_WHISPER=base for ~145 MB)"
  ASC_DATA_DIR="$DATA_DIR" "$CLI" models fetch --whisper "${ASC_WHISPER:-small}" --only-whisper
fi

# 4. optional TensorFlow cross-check environment (never used by the analysis itself)
if [ "$TF" = 1 ]; then
  say "creating separate TensorFlow 2.16.2 environment at $TF_VENV (large download)"
  [ -x "$TF_VENV/bin/python" ] || "$PY" -m venv "$TF_VENV"
  "$TF_VENV/bin/python" -m pip install --quiet --disable-pip-version-check --upgrade "pip>=25.1" >/dev/null
  "$TF_VENV/bin/python" -m pip install --quiet --disable-pip-version-check "${RETRY[@]}" -r "$REPO/requirements/tensorflow.lock"
  mkdir -p "$STAGE" "$BTMP"; cp -R "$REPO/pyproject.toml" "$REPO/README.md" "$REPO/LICENSE" "$REPO/src" "$STAGE/"
  TMPDIR="$BTMP" "$TF_VENV/bin/python" -m pip install --quiet --disable-pip-version-check --no-deps "$STAGE" >/dev/null
  rm -rf "$STAGE" "$BTMP"
fi

# 5. Claude Code skill (only its own directory)
LAUNCHER="$CLI"
if [ "$SKILL" = 1 ]; then
  if [ -e "$SKILL_DIR" ] && [ ! -f "$SKILL_DIR/$MARKER" ]; then
    die "$SKILL_DIR exists and was not created by this installer" "move it aside, or set ASC_SKILL_DIR=/other/path"
  fi
  say "installing skill to $SKILL_DIR"
  mkdir -p "$SKILL_DIR/references" "$SKILL_DIR/bin" "$SKILL_DIR/examples"
  cp "$REPO/skill/SKILL.md" "$SKILL_DIR/SKILL.md"
  cp "$REPO"/docs/*.md "$SKILL_DIR/references/"
  cp "$REPO"/examples/*.json "$REPO"/examples/*.md "$SKILL_DIR/examples/" 2>/dev/null || true
  cat > "$SKILL_DIR/bin/audio-sales-coach" <<EOF
#!/usr/bin/env bash
export ASC_DATA_DIR="\${ASC_DATA_DIR:-$DATA_DIR}"
exec "$CLI" "\$@"
EOF
  chmod +x "$SKILL_DIR/bin/audio-sales-coach"
  echo "$REPO" > "$SKILL_DIR/$MARKER"
  LAUNCHER="$SKILL_DIR/bin/audio-sales-coach"
fi

# 6. command on PATH (never overwrite someone else's file)
mkdir -p "$BIN_DIR"
LINK="$BIN_DIR/audio-sales-coach"
if [ -L "$LINK" ] || [ ! -e "$LINK" ]; then
  ln -sfn "$LAUNCHER" "$LINK"
  say "linked $LINK -> $LAUNCHER"
else
  say "WARNING: $LINK exists and is not a symlink; left untouched. Use $LAUNCHER directly."
fi
case ":$PATH:" in *":$BIN_DIR:"*) ;; *) say "note: $BIN_DIR is not on PATH; add it or call $LAUNCHER" ;; esac

# 7. evidence-council (optional, separate program)
if command -v evidence-council >/dev/null 2>&1; then
  say "evidence-council found: $(evidence-council --version 2>/dev/null) - draft notes can be reviewed (local checks by default)"
else
  say "evidence-council not found - analysis works; council review will be reported as skipped"
fi

# 8. self-test
if [ "$SELFTEST" = 1 ]; then
  say "self-test (offline, synthetic audio):"
  ASC_DATA_DIR="$DATA_DIR" "$CLI" selftest
fi
say "done. Try: audio-sales-coach analyze /path/to/call.m4a --output ~/coach-reports/call-1 --consent all-parties-consented"
