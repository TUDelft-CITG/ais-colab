#!/bin/bash
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --job-name=ais-umap-northsea-d02
#SBATCH --output=logs/umap_northsea_d02_%j.out
#SBATCH --error=logs/umap_northsea_d02_%j.err

set -euo pipefail
mkdir -p logs

module load 2025 CGAL/6.0.1-GCCcore-14.2.0 Boost/1.88.0-GCC-14.2.0 GMP/6.3.0-GCCcore-14.2.0 MPFR/4.2.2-GCCcore-14.2.0

RAW_INPUT="/projects/prjs2131/data/rws/ais/northsea/year=2025/month=01/day=02"
OUT_ROOT="/projects/prjs2131/data/rws/ais/northsea_d02_test"
TRAJ_FILE="${OUT_ROOT}/trajectorized.parquet"
WINDOWS_FILE="${OUT_ROOT}/behavior_windows_10m.parquet"
KNN_FILE="${OUT_ROOT}/knn_50.npz"
UMAP_DIR="${OUT_ROOT}/umap"

rm -rf "${OUT_ROOT}"
mkdir -p "${OUT_ROOT}"

echo "Controller Node: $(hostname -f)"

echo "==> Step 1: Computing trajectories & kinematics (2025-01-02)..."
uv run --no-sync ais-shader trajectory compute "${RAW_INPUT}" \
    -o "${TRAJ_FILE}" \
    --vessel-id-col mmsi_hash \
    --time-col timestamp_recv \
    --shuffle-backend disk \
    --n-partitions 16

echo "==> Step 2: Extracting 10-minute behavioral windows..."
uv run --no-sync ais-shader behavior extract-windows "${TRAJ_FILE}" \
    -o "${WINDOWS_FILE}" \
    --cluster-type local \
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

echo "UMAP North Sea (2025-01-02) 1-day test complete!"
ls -lh "${OUT_ROOT}" "${UMAP_DIR}"
