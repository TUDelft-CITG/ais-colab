# Behavioral Window Extraction, UMAP Embedding & Anomaly Detection

The `ais_shader.behavior` module implements an explicit, multi-step pipeline for direction-agnostic vessel behavior analysis, 2D UMAP embedding, hierarchical density clustering, and local/global anomaly detection.

Based on methodology by Jessica van den Heuvel (MSc Thesis, TU Delft, 2024), Van Engelen (2023), and Rong et al. (2024).

## Workflow Overview

Each step fails fast if required upstream columns are missing—no implicit fallback is performed:

1. **Trajectory kinematics (`ais-shader trajectory compute`)**
   Computes per-ping `acceleration_mps2`, `turn_rate_from_cog`, and CGAL `rolling_area_m2` on **trajectorized point data** (not whole-trip `to-linestring` geometries).
2. **Behavioral time windows (`ais-shader behavior extract-windows`)**
   Aggregates trajectorized points per vessel into 10-minute windows (`window_10m`) with 16 standardized motion and shape features (`sog_*`, `rot_*`, `accel_mean`, `drift_angle_*`, `length_to_beam`, `tortuosity`, `log_rolling_area`, `in_anchorage_area`).
3. **Optional KNN precomputation (`ais-shader behavior compute-knn`)**
   Computes and stores the k-nearest-neighbor graph (`knn_indices`, `knn_dists`) in an `.npz` file via `umap.umap_.nearest_neighbors` (`low_memory=True`, `compatibility_layout=False`). This avoids recomputing the neighbor graph when tuning UMAP or clustering parameters.
4. **UMAP, Clustering & Anomaly Detection (`ais-shader behavior umap`)**
   Runs UMAP 0.6+ (`compatibility_layout=False` by default), HDBSCAN or Ward agglomerative clustering, Local Outlier Factor (LOF), and Isolation Forest, and exports a GeoParquet (`LineString` per 10-minute window) plus a 4-panel dashboard PNG.
5. **Time-Dependent Multi-Month Embedding (`ais-shader behavior align`)**
   Jointly embeds time-ordered monthly window files using `umap.AlignedUMAP`, linking shared vessels (`mmsi_hash`) across adjacent months so embeddings remain comparable over time.

## Parallelism & Scaling Architecture

| Step | Execution Scope | Parallelism Mechanism |
|---|---|---|
| `trajectory compute` | **Multi-node** | Dask Distributed (`--scheduler` or `LocalCluster`) |
| `behavior extract-windows` | **Multi-node** | Dask Distributed (`--scheduler`, `--cluster-type slurm` via `SLURMCluster`, or `local`) |
| `behavior compute-knn` | **Single node** | PyNNDescent / Numba multi-core over all CPU cores |
| `behavior umap` | **Single node** | UMAP 0.6+ (Numba), `fast-hdbscan` (Numba Boruvka KD-tree) or `sklearn.cluster.HDBSCAN(n_jobs=-1)` (joblib `loky`), LOF & IsolationForest (`n_jobs=-1`) |
| `behavior align` | **Single node** | `umap.AlignedUMAP` across sampled time slices |

### Scaling Notes

- **Subsampled fit + batched transform (`--fit-sample-size`):** When a window dataset exceeds `--fit-sample-size` (default `100,000`) and `--knn-file` is not used, UMAP, clustering, `LocalOutlierFactor(novelty=True)`, and `IsolationForest` are fitted on the subsample and applied to the full dataset in batches of `100,000` rows.
- **Cross-month model reuse (`--save-model` / `--model-file`):** Fit once on a reference month (saving `umap_model.joblib`) and transform subsequent months into the same 2D UMAP, cluster, and anomaly space without re-fitting.
- **`fast-hdbscan` vs `sklearn.cluster.HDBSCAN`:** `fast-hdbscan` runs the full Boruvka KD-tree in parallel via Numba and completes 1M+ 2D points in seconds; `sklearn.cluster.HDBSCAN(n_jobs=-1)` parallelizes only the neighbor/core-distance step via `joblib` (`loky`).

## CLI Examples

```bash
# 1. Extract 10-minute behavioral windows (requires trajectorized points)
uv run --no-sync ais-shader behavior extract-windows /path/to/trajectorized.parquet \
    -o /path/to/behavior_windows_10m.parquet \
    --cluster-type slurm --n-workers 4 --cores-per-worker 16 --memory-per-worker 28GB \
    --vessel-id-col mmsi_hash --time-col timestamp_recv

# 2. Precompute 50-NN graph (optional)
uv run --no-sync ais-shader behavior compute-knn /path/to/behavior_windows_10m.parquet \
    -o /path/to/knn_50.npz --n-neighbors 50

# 3. Fit UMAP + HDBSCAN + LOF/IsolationForest and save reusable model
uv run --no-sync ais-shader behavior umap /path/to/behavior_windows_10m.parquet \
    -o /path/to/umap_out --knn-file /path/to/knn_50.npz \
    --save-model /path/to/umap_out/umap_model.joblib

# 4. Map another month into the saved UMAP space
uv run --no-sync ais-shader behavior umap /path/to/m02_windows.parquet \
    -o /path/to/m02_umap --model-file /path/to/umap_out/umap_model.joblib

# 5. Time-dependent AlignedUMAP across multiple months
uv run --no-sync ais-shader behavior align \
    /path/to/m01_windows.parquet /path/to/m02_windows.parquet \
    -o /path/to/aligned_out --sample-per-slice 50000
```

## Snellius Slurm Scripts

- [submit_umap_northsea_day.sh](../scripts/snellius/submit_umap_northsea_day.sh): 1-day end-to-end test (`trajectory compute` → `extract-windows` → `compute-knn` → `umap`).
- [submit_umap_northsea.sh](../scripts/snellius/submit_umap_northsea.sh): 1-month 2-node Dask + UMAP pipeline writing to `/scratch-shared/$USER`.
- [submit_northsea_7months_windows.sh](../scripts/snellius/submit_northsea_7months_windows.sh) & [submit_northsea_7months_align.sh](../scripts/snellius/submit_northsea_7months_align.sh): 7-month Slurm array job for window extraction followed by `AlignedUMAP`.
