#!/usr/bin/env bash
set -euo pipefail

# Output SIF path (override with first arg)
OUTPUT="${1:-alphafast.sif}"

# Set cache dir — change if your home quota is small
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$HOME/.apptainer/cache}"
export SINGULARITY_CACHEDIR="${APPTAINER_CACHEDIR}"

mkdir -p "$APPTAINER_CACHEDIR"

echo "==> Building $OUTPUT from docker://romerolabduke/alphafast:latest"
echo "    Cache: $APPTAINER_CACHEDIR"
echo ""

# Prefer apptainer, fall back to singularity
if command -v apptainer &>/dev/null; then
    apptainer pull --force "$OUTPUT" docker://romerolabduke/alphafast:latest
elif command -v singularity &>/dev/null; then
    singularity pull --force "$OUTPUT" docker://romerolabduke/alphafast:latest
else
    echo "ERROR: neither apptainer nor singularity found in PATH" >&2
    exit 1
fi

echo ""
echo "==> Done: $OUTPUT ($(du -sh "$OUTPUT" | cut -f1))"
