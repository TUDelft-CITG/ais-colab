#!/bin/bash
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=224G
#SBATCH --time=00:45:00
#SBATCH --array=1-7
#SBATCH --job-name=ais-northsea-m
#SBATCH --output=logs/northsea_m%a_%A_%a.out
#SBATCH --error=logs/northsea_m%a_%A_%a.err

set -euo pipefail
mkdir -p logs

module load 2025 CGAL/6.0.1-GCCcore-14.2.0 Boost/1.88.0-GCC-14.2.0 GMP/6.3.0-GCCcore-14.2.0 MPFR/4.2.2-GCCcore-14.2.0

MONTH=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
RAW_INPUT="/projects/prjs2131/data/rws/ais/northsea/year=2025/month=${MONTH}"
OUT_ROOT="/scratch-shared/${USER:-fbaart}/ais_northsea_7m"
TRAJ_FILE="${OUT_ROOT}/m${MONTH}_trajectorized.parquet"
WINDOWS_FILE="${OUT_ROOT}/m${MONTH}_windows.parquet"

mkdir -p "${OUT_ROOT}"

SCHEDULER_HOST=$(hostname -f)
PORT=$((8800 + SLURM_ARRAY_TASK_ID))
SCHEDULER_URL="tcp://${SCHEDULER_HOST}:${PORT}"

echo "Node: ${SCHEDULER_HOST} | Month: ${MONTH}"
uv run --no-sync dask-scheduler --host "${SCHEDULER_HOST}" --port "${PORT}" --no-dashboard &
SCHEDULER_PID=$!
sleep 4

uv run --no-sync dask-worker "${SCHEDULER_URL}" \
    --nthreads 16 \
    --nworkers 8 \
    --memory-limit 24GB \
    --no-dashboard &
WORKERS_PID=$!
sleep 6

cleanup() {
    kill "${WORKERS_PID}" "${SCHEDULER_PID}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "==> [m${MONTH}] Step 1: trajectory compute..."
uv run --no-sync ais-shader trajectory compute "${RAW_INPUT}" \
    -o "${TRAJ_FILE}" \
    --scheduler "${SCHEDULER_URL}" \
    --vessel-id-col mmsi_hash \
    --time-col timestamp_recv \
    --shuffle-backend disk \
    --n-partitions 64

echo "==> [m${MONTH}] Step 2: behavior extract-windows..."
uv run --no-sync ais-shader behavior extract-windows "${TRAJ_FILE}" \
    -o "${WINDOWS_FILE}" \
    --scheduler "${SCHEDULER_URL}" \
    --vessel-id-col mmsi_hash \
    --time-col timestamp_recv

echo "==> [m${MONTH}] Done: ${WINDOWS_FILE}"
ls -lh "${WINDOWS_FILE}"
