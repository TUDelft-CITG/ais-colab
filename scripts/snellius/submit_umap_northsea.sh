#!/bin/bash
#SBATCH --partition=rome
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=224G
#SBATCH --time=02:00:00
#SBATCH --job-name=ais-umap-northsea-m01
#SBATCH --output=logs/umap_northsea_m01_%j.out
#SBATCH --error=logs/umap_northsea_m01_%j.err

set -euo pipefail
mkdir -p logs

module load 2025 CGAL/6.0.1-GCCcore-14.2.0 Boost/1.88.0-GCC-14.2.0 GMP/6.3.0-GCCcore-14.2.0 MPFR/4.2.2-GCCcore-14.2.0

RAW_INPUT="/projects/prjs2131/data/rws/ais/northsea/year=2025/month=01"
OUT_ROOT="/scratch-shared/${USER:-fbaart}/ais_northsea_m01"
TRAJ_FILE="${OUT_ROOT}/trajectorized.parquet"
WINDOWS_FILE="${OUT_ROOT}/behavior_windows_10m.parquet"
KNN_FILE="${OUT_ROOT}/knn_50.npz"
UMAP_DIR="${OUT_ROOT}/umap"

mkdir -p "${OUT_ROOT}"

SCHEDULER_HOST=$(hostname -f)
SCHEDULER_URL="tcp://${SCHEDULER_HOST}:8786"

echo "Controller Node: ${SCHEDULER_HOST}"
echo "Starting Dask Scheduler at ${SCHEDULER_URL}..."
uv run --no-sync dask-scheduler --host "${SCHEDULER_HOST}" --port 8786 --dashboard-address :8787 &
SCHEDULER_PID=$!
sleep 5

echo "Starting Dask Workers across 2 allocated nodes..."
srun uv run --no-sync dask-worker "${SCHEDULER_URL}" \
    --nthreads 16 \
    --nworkers 8 \
    --memory-limit 26GB \
    --no-dashboard &
WORKERS_PID=$!
sleep 10

cleanup() {
    echo "Tearing down Dask cluster..."
    kill "${WORKERS_PID}" "${SCHEDULER_PID}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "==> Step 1: Computing trajectories & kinematics (2025-01)..."
uv run --no-sync ais-shader trajectory compute "${RAW_INPUT}" \
    -o "${TRAJ_FILE}" \
    --scheduler "${SCHEDULER_URL}" \
    --vessel-id-col mmsi_hash \
    --time-col timestamp_recv \
    --shuffle-backend disk \
    --n-partitions 128

echo "==> Step 2: Extracting 10-minute behavioral windows..."
uv run --no-sync ais-shader behavior extract-windows "${TRAJ_FILE}" \
    -o "${WINDOWS_FILE}" \
    --scheduler "${SCHEDULER_URL}" \
    --vessel-id-col mmsi_hash \
    --time-col timestamp_recv

echo "==> Step 3: Precomputing 50-NN graph..."
uv run --no-sync ais-shader behavior compute-knn "${WINDOWS_FILE}" \
    -o "${KNN_FILE}" \
    --n-neighbors 50

echo "==> Step 4: Running UMAP (0.6+ pipeline), clustering & anomaly detection..."
uv run --no-sync ais-shader behavior umap "${WINDOWS_FILE}" \
    -o "${UMAP_DIR}" \
    --knn-file "${KNN_FILE}" \
    --save-model "${UMAP_DIR}/umap_model.joblib" \
    --cluster-method hdbscan \
    --fit-sample-size 100000

echo "UMAP North Sea (2025-01) multi-step workflow complete!"
ls -lh "${OUT_ROOT}" "${UMAP_DIR}"
