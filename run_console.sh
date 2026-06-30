#!/usr/bin/env bash
# Launch the AGLM Cognitive Console.
#
# AGLM is an augmentation layer, not a model — this UI loads with zero heavy ML
# dependencies. A language model is optional (Ollama, local or :cloud).
set -e
cd "$(dirname "$0")"

PORT="${PORT:-5000}"

# Flask is the only hard requirement for the UI to load.
python3 - <<'PY' || { echo "Installing Flask…"; pip install --quiet flask; }
import flask  # noqa
PY

# Optional: report Ollama status (the console still works without it).
if command -v ollama >/dev/null 2>&1; then
  echo "Ollama detected — local + cloud models will be available in the UI."
else
  echo "Ollama not found — console runs in augmentation-only mode."
  echo "  (install from https://ollama.com to attach local/cloud models)"
fi

echo "Starting AGLM Cognitive Console on http://localhost:${PORT}"
PORT="$PORT" exec python3 aglm_app.py
