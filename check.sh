#!/usr/bin/env bash
# Gate canónico del proyecto tono-politico.
# Ejecuta ruff + ty + pytest (excluyendo tests que cargan modelos pesados).
#
# Uso:
#   bash check.sh           # gate rápido (default)
#   RUN_SLOW=1 bash check.sh # gate + tests slow (carga modelos reales)

set -euo pipefail

cd "$(dirname "$0")"

echo "── ruff check ──────────────────────────────────────────"
uv run ruff check src/ tests/ main.py
echo "── ruff format --check ────────────────────────────────"
uv run ruff format --check src/ tests/ main.py
echo "── ty check ───────────────────────────────────────────"
uv run ty check
echo "── pytest ─────────────────────────────────────────────"
if [[ "${RUN_SLOW:-0}" == "1" ]]; then
    RUN_SLOW_MODELS=1 uv run pytest tests/ -v --tb=short
else
    uv run pytest tests/ -m "not slow" --tb=short
fi
echo ""
echo "✅ Gate completo: ruff + ty + pytest OK"
