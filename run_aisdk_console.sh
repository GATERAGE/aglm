#!/usr/bin/env bash
# Launch the full AGLM participant stack:
#   - Flask AGLM brain (faculties + augmentation) on :5000
#   - Next.js + Vercel AI SDK console on :3000  (open this one)
#
# AGLM augments a model; the AI SDK streams it. Ollama serves the model
# (local for modest hardware, :cloud for larger reasoning).
set -e
cd "$(dirname "$0")"

# 1. AGLM brain (Flask). Only Flask is required for it to load.
python3 - <<'PY' >/dev/null 2>&1 || pip install --quiet flask
import flask
PY
echo "▸ Starting AGLM brain (Flask) on http://localhost:5000 …"
PORT=5000 python3 aglm_app.py >/tmp/aglm_brain.log 2>&1 &
BRAIN_PID=$!

# 2. AI SDK console (Next.js). Install deps once.
cd aglm-console
if [ ! -d node_modules ]; then
  echo "▸ Installing console dependencies (first run) …"
  npm install --no-audit --no-fund
fi
echo "▸ Starting AI SDK console on http://localhost:3000 …"
echo "  (Ollama should be running: 'ollama serve'; default model qwen3:0.6b)"
trap "kill $BRAIN_PID 2>/dev/null" EXIT
npm run dev
