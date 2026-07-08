#!/usr/bin/env bash
# Limpieza de output/, data/ y caches Python del proyecto tono-politico.
#
# Uso:
#   bash clean.sh                # limpiar todo (con confirmación)
#   bash clean.sh --dry-run      # previsualizar qué se borraría
#   bash clean.sh -y             # sin confirmación (scripting/CI)
#   bash clean.sh --output       # solo output/
#   bash clean.sh --data         # solo data/
#   bash clean.sh --caches       # solo caches Python
#
# No borra: .venv/, *.pt (modelos Whisper), config/, src/, tests/, docs/

set -euo pipefail
cd "$(dirname "$0")"

DRY_RUN=0
ASSUME_YES=0
CLEAN_OUTPUT=1
CLEAN_DATA=1
CLEAN_CACHES=1

for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN=1 ;;
        -y|--yes)   ASSUME_YES=1 ;;
        --output)   CLEAN_OUTPUT=1; CLEAN_DATA=0; CLEAN_CACHES=0 ;;
        --data)     CLEAN_DATA=1; CLEAN_OUTPUT=0; CLEAN_CACHES=0 ;;
        --caches)   CLEAN_CACHES=1; CLEAN_OUTPUT=0; CLEAN_DATA=0 ;;
        -h|--help)
            sed -n '2,12p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Opción no reconocida: $arg (usa --help)"
            exit 1
            ;;
    esac
done

# ── Resumen de lo que se va a limpiar ──────────────────────

print_target() {
    local label="$1"
    local path="$2"
    local count="—"
    if [[ -d "$path" ]]; then
        count=$(find "$path" -type f 2>/dev/null | wc -l | tr -d ' ')
    elif [[ -f "$path" ]]; then
        count=1
    else
        count="(no existe)"
    fi
    echo "  $label: $path ($count archivos)"
}

echo "━ Limpieza de tono-politico ━━━━━━━━━━━━━━━━━━━━━━━━━━"
[[ "$DRY_RUN" == "1" ]] && echo "  ⚠ DRY-RUN: no se borrará nada"
echo ""

if [[ "$CLEAN_OUTPUT" == "1" ]]; then
    print_target "output/" "output"
fi
if [[ "$CLEAN_DATA" == "1" ]]; then
    print_target "data/" "data"
fi
if [[ "$CLEAN_CACHES" == "1" ]]; then
    print_target "__pycache__" "__pycache__"
    print_target ".pytest_cache/" ".pytest_cache"
    print_target ".ruff_cache/" ".ruff_cache"
    # subdirectorios __pycache__
    local_pycache_count=0
    if [[ -d "src" ]]; then
        local_pycache_count=$(find src/ tests/ -type d -name "__pycache__" 2>/dev/null | wc -l | tr -d ' ')
    fi
    echo "  __pycache__ en src/tests/: $local_pycache_count dirs"
fi

# ── Confirmación ───────────────────────────────────────────

if [[ "$DRY_RUN" == "1" ]]; then
    echo ""
    echo "Fin del dry-run. Ejecuta sin --dry-run para borrar."
    exit 0
fi

if [[ "$ASSUME_YES" != "1" ]]; then
    echo ""
    read -rp "¿Borrar todo lo anterior? [s/N] " confirm
    if [[ "$confirm" != "s" && "$confirm" != "S" ]]; then
        echo "Abortado."
        exit 0
    fi
fi

# ── Ejecución ──────────────────────────────────────────────

rm_target() {
    local label="$1"
    local path="$2"
    if [[ -d "$path" ]]; then
        echo "  🗑  $label"
        rm -rf "$path"
    elif [[ -f "$path" ]]; then
        echo "  🗑  $label"
        rm -f "$path"
    fi
}

echo ""

if [[ "$CLEAN_OUTPUT" == "1" ]]; then
    rm_target "output/" "output"
fi

if [[ "$CLEAN_DATA" == "1" ]]; then
    rm_target "data/" "data"
fi

if [[ "$CLEAN_CACHES" == "1" ]]; then
    rm_target ".pytest_cache/" ".pytest_cache"
    rm_target ".ruff_cache/" ".ruff_cache"
    # __pycache__ en raíz y subdirectorios
    find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
    echo "  🗑  __pycache__/ (raíz + src/tests/)"
fi

echo ""
echo "✅ Limpieza completa."
