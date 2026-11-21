#!/bin/bash
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=112G
#SBATCH --time=01:00:00
#SBATCH --job-name=ais-northsea-align
#SBATCH --output=logs/northsea_align_%j.out
#SBATCH --error=logs/northsea_align_%j.err

set -euo pipefail
mkdir -p logs

OUT_ROOT="/scratch-shared/${USER:-fbaart}/ais_northsea_7m"
ALIGNED_DIR="${OUT_ROOT}/aligned"

echo "Controller Node: $(hostname -f)"
echo "==> Running AlignedUMAP across months 01..07..."

uv run --no-sync ais-shader behavior align \
    "${OUT_ROOT}/m01_windows.parquet" \
    "${OUT_ROOT}/m02_windows.parquet" \
    "${OUT_ROOT}/m03_windows.parquet" \
    "${OUT_ROOT}/m04_windows.parquet" \
    "${OUT_ROOT}/m05_windows.parquet" \
    "${OUT_ROOT}/m06_windows.parquet" \
    "${OUT_ROOT}/m07_windows.parquet" \
    -o "${ALIGNED_DIR}" \
    --sample-per-slice 50000 \
    --alignment-regularisation 0.01 \
    --alignment-window-size 3

echo "AlignedUMAP (months 01..07) complete!"
ls -lh "${ALIGNED_DIR}"
