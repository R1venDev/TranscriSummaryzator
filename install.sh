#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
TOOLS_DIR="$ROOT_DIR/work/tools"
PYTHON_DIR="$ROOT_DIR/work/pythons"
CACHE_DIR="$ROOT_DIR/work/cache/uv"

mkdir -p "$TOOLS_DIR" "$PYTHON_DIR" "$CACHE_DIR" "$ROOT_DIR/inbox" "$ROOT_DIR/outputs" "$ROOT_DIR/state"

if [ ! -d "$ROOT_DIR/work/vendor/GigaAM/.git" ]; then
  mkdir -p "$ROOT_DIR/work/vendor"
  git clone --depth 1 https://github.com/salute-developers/GigaAM.git "$ROOT_DIR/work/vendor/GigaAM"
fi
if [ ! -d "$ROOT_DIR/work/vendor/DiariZen/.git" ]; then
  mkdir -p "$ROOT_DIR/work/vendor"
  git clone --depth 1 https://github.com/BUTSpeechFIT/DiariZen.git "$ROOT_DIR/work/vendor/DiariZen"
fi

if [ ! -x "$TOOLS_DIR/bin/uv" ]; then
  /usr/bin/python3 -m pip install --disable-pip-version-check --target "$TOOLS_DIR" uv
fi

UV="$TOOLS_DIR/bin/uv"
UV_PYTHON_INSTALL_DIR="$PYTHON_DIR" UV_CACHE_DIR="$CACHE_DIR" "$UV" python install --no-bin 3.10

UV_PYTHON_INSTALL_DIR="$PYTHON_DIR" UV_CACHE_DIR="$CACHE_DIR" "$UV" venv --clear --python 3.10 "$ROOT_DIR/.venv-gigaam"
UV_PYTHON_INSTALL_DIR="$PYTHON_DIR" UV_CACHE_DIR="$CACHE_DIR" "$UV" pip install --python "$ROOT_DIR/.venv-gigaam/bin/python" -r "$ROOT_DIR/requirements-gigaam.txt"

UV_PYTHON_INSTALL_DIR="$PYTHON_DIR" UV_CACHE_DIR="$CACHE_DIR" "$UV" venv --clear --python 3.10 "$ROOT_DIR/.venv-diarizen"
UV_PYTHON_INSTALL_DIR="$PYTHON_DIR" UV_CACHE_DIR="$CACHE_DIR" "$UV" pip install --python "$ROOT_DIR/.venv-diarizen/bin/python" "setuptools<81" wheel
UV_PYTHON_INSTALL_DIR="$PYTHON_DIR" UV_CACHE_DIR="$CACHE_DIR" "$UV" pip install --python "$ROOT_DIR/.venv-diarizen/bin/python" torch==2.1.1 torchaudio==2.1.1 torchvision==0.16.1 numpy==1.26.4
UV_PYTHON_INSTALL_DIR="$PYTHON_DIR" UV_CACHE_DIR="$CACHE_DIR" "$UV" pip install --python "$ROOT_DIR/.venv-diarizen/bin/python" -r "$ROOT_DIR/requirements-diarizen-macos.txt"
UV_PYTHON_INSTALL_DIR="$PYTHON_DIR" UV_CACHE_DIR="$CACHE_DIR" "$UV" pip install --python "$ROOT_DIR/.venv-diarizen/bin/python" -e "$ROOT_DIR/work/vendor/DiariZen"
UV_PYTHON_INSTALL_DIR="$PYTHON_DIR" UV_CACHE_DIR="$CACHE_DIR" "$UV" pip install --python "$ROOT_DIR/.venv-diarizen/bin/python" -e "$ROOT_DIR/work/vendor/DiariZen/pyannote-audio"
# Some transitive packages accept any Torch and would otherwise upgrade the
# old Pyannote-compatible pair. Reassert the official DiariZen pins last.
UV_PYTHON_INSTALL_DIR="$PYTHON_DIR" UV_CACHE_DIR="$CACHE_DIR" "$UV" pip install --python "$ROOT_DIR/.venv-diarizen/bin/python" torch==2.1.1 torchaudio==2.1.1 torchvision==0.16.1 numpy==1.26.4

chmod +x "$ROOT_DIR/meeting-transcript" "$ROOT_DIR/pipeline.py" "$ROOT_DIR/scripts/asr_worker.py" "$ROOT_DIR/scripts/diarize_worker.py"
"$ROOT_DIR/meeting-transcript" doctor
