#!/bin/bash
# Enhanced launch script: automatically adds project root to PYTHONPATH so that
# `import svox2` works even if the editable install was not performed yet or VS Code
# picked a different interpreter search path. For best performance still run
# `pip install -e .` in the repo root. This merely mitigates path issues.
set -e

echo Launching experiment "$1"
echo GPU "$2"
echo EXTRA "${@:3}"

# Resolve project root (parent of this opt/ directory)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

# Inject PYTHONPATH if not already containing root
case ":$PYTHONPATH:" in
	*":$ROOT_DIR:"*) ;; # already present
	*) export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
echo Using PYTHONPATH: "$PYTHONPATH"

# Diagnostics: show python interpreter & version
PYTHON_BIN=$(which python || true)
echo Python interpreter: "$PYTHON_BIN"
python -c 'import sys; print("Python version:", sys.version.split()[0]); print("sys.path first entries:", sys.path[:3])' || echo "(Python quick check failed)"

CKPT_DIR=ckpt/$1
mkdir -p "$CKPT_DIR"
NOHUP_FILE="$CKPT_DIR/log"
echo CKPT "$CKPT_DIR"
echo LOGFILE "$NOHUP_FILE"

CUDA_VISIBLE_DEVICES=$2 nohup python -u opt.py -t "$CKPT_DIR" "${@:3}" > "$NOHUP_FILE" 2>&1 &
echo DETACH
