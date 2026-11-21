"""AIS 10-minute direction-agnostic behavior, UMAP embedding & anomaly detection.

Based on methodology by Jessica van den Heuvel (MSc Thesis, TU Delft, 2024),
Van Engelen (2023), and Rong et al. (2024).

Workflow (explicit multi-step pipeline):
1. ``ais-shader trajectory compute``: compute per-point kinematics
   (``acceleration_mps2``, ``turn_rate_from_cog``, ``rolling_area_m2``).
2. ``ais-shader behavior extract-windows``: aggregate trajectorized points into
   10-minute direction-agnostic behavioral window features.
3. ``ais-shader behavior umap``: fit/transform 2D UMAP, cluster with HDBSCAN or
   Ward agglomerative clustering, detect local (LOF) and global
   (IsolationForest) anomalies, and generate the dashboard.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import geopandas as gpd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as pds
from shapely.geometry import LineString, Point
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor, kneighbors_graph
from sklearn.preprocessing import StandardScaler
import umap
from umap.umap_ import nearest_neighbors

from .preprocessing import build_vessel_mapping, get_vessel_group

logger = logging.getLogger(__name__)

BEHAVIOR_FEATURE_COLS = [
    "sog_mean", "sog_std", "sog_min", "sog_max", "sog_median",
    "rot_mean", "rot_std", "rot_max", "rot_median",
    "accel_mean",
    "drift_angle_mean", "drift_angle_max",
    "length_to_beam",
    "tortuosity",
    "log_rolling_area",
    "in_anchorage_area",
]

REQUIRED_WINDOW_COORD_COLS = ["lon_start", "lat_start", "lon_end", "lat_end"]

_ANOMALY_PLOT_SPECS = (
    ("local_outlier", "#ff9800", "Lokaal afwijkend (LOF)", "Lokaal afwijkend"),
    ("global_outlier", "#e91e63", "Globaal afwijkend (IsoForest)", "Globaal afwijkend"),
    ("extreme_both", "#d32f2f", "Zowel lokaal als globaal", "Extreem (beide)"),
)


VESSEL_GROUPS = (
    "Cargo",
    "Tanker",
    "Fishing",
    "Passenger",
    "Tug, Towing",
    "Support",
    "High-Speed Craft (HSC)",
    "Dredging",
    "Military",
    "Pleasure Craft/Sailing",
    "Other",
    "Not Available",
)

_DEFAULT_VESSEL_GROUPS_JSON = Path(__file__).resolve().parents[2] / "resources" / "vessel_groups.json"


def map_shiptype_to_category(shiptype_series: pd.Series) -> pd.Series:
    """Map AIS ship type codes to Marine Cadastre vessel groups using ``resources/vessel_groups.json``."""
    mapping = build_vessel_mapping(_DEFAULT_VESSEL_GROUPS_JSON) if _DEFAULT_VESSEL_GROUPS_JSON.exists() else {}
    return shiptype_series.map(lambda st: get_vessel_group(st, mapping))


def _build_feature_matrix(df: pd.DataFrame, include_vessel_type: bool) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix from motion/shape features and optional one-hot Marine Cadastre vessel groups."""
    cols = list(BEHAVIOR_FEATURE_COLS)
    X_base = np.nan_to_num(df[BEHAVIOR_FEATURE_COLS].to_numpy(copy=True), nan=0.0)
    if not include_vessel_type or "shiptypeais" not in df.columns:
        return X_base, cols

    groups = df["vessel_group"] if "vessel_group" in df.columns else map_shiptype_to_category(df["shiptypeais"])
    one_hot = np.column_stack([(groups == g).to_numpy(dtype=np.float64) for g in VESSEL_GROUPS])
    cols.extend(f"vessel_group_{g}" for g in VESSEL_GROUPS)
    return np.hstack([X_base, one_hot]), cols


def haversine_np(lon1, lat1, lon2, lat2):
    """Vectorized haversine distance in meters."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return 6367000.0 * c


def _create_dask_client(
    scheduler: Optional[str] = None,
    cluster_type: str = "local",
    n_workers: int = 4,
    cores_per_worker: int = 16,
    memory_per_worker: str = "28GB",
    walltime: str = "01:00:00",
    queue: str = "rome",
    interface: str = "ib0",
) -> tuple[Optional[object], Client]:
    """Create a Dask client from a scheduler URL, SLURMCluster, or LocalCluster."""
    if scheduler:
        return None, Client(scheduler)
    if cluster_type == "slurm":
        try:
            from dask_jobqueue import SLURMCluster
        except ImportError as exc:
            raise ImportError(
                "dask-jobqueue is required for cluster_type='slurm'. "
                "Install it with `pip install ais-shader[slurm]` or `uv sync --extra slurm`."
            ) from exc

        os.makedirs("logs/dask_slurm", exist_ok=True)
        logger.info(
            "Launching SLURMCluster (queue=%s, workers=%d, cores=%d, mem=%s, walltime=%s)...",
            queue, n_workers, cores_per_worker, memory_per_worker, walltime,
        )
        cluster = SLURMCluster(
            queue=queue,
            cores=cores_per_worker,
            memory=memory_per_worker,
            walltime=walltime,
            interface=interface,
            log_directory="logs/dask_slurm",
            job_extra_directives=[
                "--output=logs/dask_slurm/dask_worker_%j.out",
                "--error=logs/dask_slurm/dask_worker_%j.err",
            ],
        )
        cluster.scale(jobs=n_workers)
        client = Client(cluster)
        logger.info("Waiting for at least 1 Slurm worker to register...")
        client.wait_for_workers(1, timeout=180)
        logger.info("Connected Slurm workers: %d", len(client.scheduler_info()["workers"]))
        return cluster, client
    if cluster_type == "local":
        cluster = LocalCluster()
        return cluster, Client(cluster)
    raise ValueError(f"Unsupported cluster_type: {cluster_type!r}. Expected 'local' or 'slurm'.")


def extract_behavior_windows(
    input_file: Path | str,
    polygons_file: Optional[Path | str] = None,
    window: str = "10min",
    min_pings: int = 5,
    vessel_id_col: str = "mmsi_hash",
    time_col: str = "timestamp_recv",
) -> pd.DataFrame:
    """Aggregate trajectorized AIS points into direction-agnostic behavioral windows.

    Requires precomputed trajectory columns (``acceleration_mps2``,
    ``turn_rate_from_cog``, ``rolling_area_m2``) produced by
    ``ais-shader trajectory compute``. Raises ``ValueError`` if any required
    column is missing.
    """
    input_path = Path(input_file)
    required_cols = [
        time_col,
        vessel_id_col,
        "longitude",
        "latitude",
        "sog",
        "cog",
        "heading",
        "length",
        "beam",
        "acceleration_mps2",
        "turn_rate_from_cog",
        "rolling_area_m2",
        "shiptypeais",
    ]

    dataset = pds.dataset(str(input_path), format="parquet", partitioning="hive")
    available_cols = set(dataset.schema.names)
    missing_cols = [c for c in required_cols if c not in available_cols]
    if missing_cols:
        raise ValueError(
            f"Input {input_path} is missing required trajectorized column(s): {missing_cols}. "
            "Run `ais-shader trajectory compute` first."
        )

    logger.info("Reading trajectorized AIS data with Dask from %s...", input_path)
    ddf = dd.read_parquet(str(input_path), columns=required_cols)

    logger.info("Discretizing into %s windows & computing kinematics...", window)
    ddf[time_col] = dd.to_datetime(ddf[time_col])
    ddf["window_10m"] = ddf[time_col].dt.floor(window)
    ddf["abs_rot"] = ddf["turn_rate_from_cog"].abs().clip(upper=10.0)
    ddf["abs_accel"] = ddf["acceleration_mps2"].abs().clip(upper=2.0)

    heading_diff = (ddf["heading"] - ddf["cog"] + 180.0) % 360.0 - 180.0
    ddf["drift_angle"] = heading_diff.abs()

    logger.info("Computing distributed groupby aggregations...")
    agg = ddf.groupby([vessel_id_col, "window_10m"]).agg({
        "sog": ["count", "mean", "std", "min", "max", "median"],
        "abs_rot": ["mean", "std", "max", "median"],
        "abs_accel": ["mean", "std"],
        "drift_angle": ["mean", "max"],
        "rolling_area_m2": ["max"],
        "length": ["first"],
        "beam": ["first"],
        "longitude": ["first", "last"],
        "latitude": ["first", "last"],
        "shiptypeais": ["first"],
    })

    t0 = time.time()
    df = agg.compute()
    logger.info("Dask aggregation finished in %.2fs! Raw windows: %s", time.time() - t0, f"{len(df):,}")

    df.columns = [
        "n_pings",
        "sog_mean", "sog_std", "sog_min", "sog_max", "sog_median",
        "rot_mean", "rot_std", "rot_max", "rot_median",
        "accel_mean", "accel_std",
        "drift_angle_mean", "drift_angle_max",
        "rolling_area_max",
        "length", "beam",
        "lon_start", "lon_end",
        "lat_start", "lat_end",
        "shiptypeais",
    ]
    df = df.reset_index()
    df = df[df["n_pings"] >= min_pings].copy()

    for col in (
        "sog_std", "rot_mean", "rot_std", "rot_max", "rot_median",
        "accel_mean", "accel_std", "drift_angle_mean", "drift_angle_max", "rolling_area_max",
    ):
        df[col] = df[col].fillna(0.0)

    df["length_to_beam"] = np.where(df["beam"] > 0, df["length"] / df["beam"], np.nan)
    median_ltb = df["length_to_beam"].median()
    fallback_ltb = 5.0 if np.isnan(median_ltb) else median_ltb
    df["length_to_beam"] = df["length_to_beam"].fillna(fallback_ltb).clip(1.0, 15.0)

    disp_m = haversine_np(
        df["lon_start"].values, df["lat_start"].values,
        df["lon_end"].values, df["lat_end"].values,
    )
    df["displacement_m"] = disp_m
    df["displacement_km"] = disp_m / 1000.0

    expected_dist_m = df["sog_mean"] * 0.514444 * 600.0
    df["tortuosity"] = np.where(
        expected_dist_m > 50.0,
        np.clip(df["displacement_m"] / expected_dist_m, 0.0, 1.0),
        0.0,
    )
    df["log_rolling_area"] = np.log1p(df["rolling_area_max"])

    df["in_anchorage_area"] = 0
    if polygons_file is not None:
        poly_path = Path(polygons_file)
        if poly_path.exists():
            try:
                poly_gdf = gpd.read_file(poly_path)
                points_geom = [Point(xy) for xy in zip(df["lon_start"], df["lat_start"])]
                points_gdf = gpd.GeoDataFrame(df[[vessel_id_col, "window_10m"]], geometry=points_geom, crs="EPSG:4326")
                joined = gpd.sjoin(points_gdf, poly_gdf, how="inner", predicate="within")
                df.loc[joined.index, "in_anchorage_area"] = 1
                logger.info("Windows inside anchorage areas: %s", f"{df['in_anchorage_area'].sum():,}")
            except Exception as exc:
                logger.warning("Anchorage check failed: %s", exc)

    logger.info("Valid behavioral windows ready: %s", f"{len(df):,}")
    return df


def run_extract_windows(
    input_file: Path | str,
    output_file: Path | str,
    scheduler: Optional[str] = None,
    cluster_type: str = "local",
    n_workers: int = 4,
    cores_per_worker: int = 16,
    memory_per_worker: str = "28GB",
    walltime: str = "01:00:00",
    queue: str = "rome",
    interface: str = "ib0",
    polygons_file: Optional[Path | str] = None,
    window: str = "10min",
    min_pings: int = 5,
    vessel_id_col: str = "mmsi_hash",
    time_col: str = "timestamp_recv",
) -> pd.DataFrame:
    """Extract 10-minute behavioral windows using Dask and write them to Parquet."""
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cluster, client = _create_dask_client(
        scheduler=scheduler,
        cluster_type=cluster_type,
        n_workers=n_workers,
        cores_per_worker=cores_per_worker,
        memory_per_worker=memory_per_worker,
        walltime=walltime,
        queue=queue,
        interface=interface,
    )
    try:
        df = extract_behavior_windows(
            input_file=input_file,
            polygons_file=polygons_file,
            window=window,
            min_pings=min_pings,
            vessel_id_col=vessel_id_col,
            time_col=time_col,
        )
        df.to_parquet(out_path, index=False)
        logger.info("Behavioral windows saved to %s", out_path)
        return df
    finally:
        client.close()
        if cluster is not None:
            cluster.close()


def _validate_window_df(df: pd.DataFrame) -> None:
    """Validate that a behavioral window DataFrame has all required feature & coordinate columns."""
    required = BEHAVIOR_FEATURE_COLS + REQUIRED_WINDOW_COORD_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required window feature column(s): {missing}. "
            "Run `ais-shader behavior extract-windows` first."
        )


def compute_and_save_knn(
    windows_file: Path | str,
    output_file: Path | str,
    n_neighbors: int = 50,
    compatibility_layout: bool = False,
    include_vessel_type: bool = True,
    random_state: Optional[int] = 42,
) -> Path:
    """Precompute and store UMAP k-nearest neighbors (.npz) for a prepared windows Parquet."""
    df = pd.read_parquet(windows_file)
    _validate_window_df(df)

    X, _ = _build_feature_matrix(df, include_vessel_type=include_vessel_type)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    effective_k = min(n_neighbors, max(2, len(X_scaled) - 1))
    logger.info("Computing %d-NN graph for %s windows...", effective_k, f"{len(X_scaled):,}")
    t0 = time.time()
    knn_indices, knn_dists, _ = nearest_neighbors(
        X_scaled,
        n_neighbors=effective_k,
        metric="euclidean",
        metric_kwds={},
        angular=False,
        random_state=random_state,
        low_memory=True,
        compatibility_layout=compatibility_layout,
    )
    logger.info("KNN computation finished in %.2fs.", time.time() - t0)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        n_rows=np.array([len(df)], dtype=np.int64),
    )
    logger.info("Precomputed KNN saved to %s", out_path)
    return out_path


def load_knn(knn_file: Path | str, expected_rows: int) -> tuple[np.ndarray, np.ndarray, None]:
    """Load precomputed KNN indices and distances from an .npz file."""
    data = np.load(knn_file)
    knn_indices = np.ascontiguousarray(data["knn_indices"])
    knn_dists = np.ascontiguousarray(data["knn_dists"], dtype=np.float32)
    if len(knn_indices) != expected_rows:
        raise ValueError(
            f"Precomputed KNN row count ({len(knn_indices)}) does not match windows row count ({expected_rows})."
        )
    return (knn_indices, knn_dists, None)


def _batched_predict_and_score(model, X: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Run .predict() and .score_samples() in batches to bound memory usage."""
    n = len(X)
    preds = np.empty(n, dtype=bool)
    scores = np.empty(n, dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = X[start:end]
        preds[start:end] = model.predict(chunk) == -1
        scores[start:end] = -model.score_samples(chunk)
    return preds, scores


def run_umap_and_anomalies(
    df: pd.DataFrame,
    cluster_method: str = "hdbscan",
    n_clusters: int = 8,
    contamination: float = 0.015,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    min_cluster_size: int = 150,
    min_samples: int = 20,
    fit_sample_size: Optional[int] = 100_000,
    batch_size: int = 100_000,
    compatibility_layout: bool = False,
    by_vessel_category: bool = True,
    include_vessel_type: bool = True,
    precomputed_knn: Optional[tuple] = None,
    return_model: bool = False,
    random_state: int = 42,
):
    """Run UMAP embedding, hierarchical clustering, and LOF/IsolationForest anomaly detection.

    Expects a prepared behavioral window table containing ``BEHAVIOR_FEATURE_COLS``
    and ``REQUIRED_WINDOW_COORD_COLS``. Raises ``ValueError`` if any required
    column is missing.
    """
    _validate_window_df(df)

    X, used_feature_cols = _build_feature_matrix(df, include_vessel_type=include_vessel_type)
    logger.info("Standardizing motion, shape & vessel-type features (%d features)...", len(used_feature_cols))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_rows = len(df)
    use_subsample = precomputed_knn is None and fit_sample_size is not None and n_rows > fit_sample_size
    if use_subsample:
        rng = np.random.default_rng(random_state)
        sample_idx = np.sort(rng.choice(n_rows, size=fit_sample_size, replace=False))
        X_fit = X_scaled[sample_idx]
        logger.info("Fitting UMAP/clustering/anomaly models on %s sampled windows (out of %s)...", f"{fit_sample_size:,}", f"{n_rows:,}")
    else:
        X_fit = X_scaled

    logger.info("Fitting 2D UMAP embedding (compatibility_layout=%s)...", compatibility_layout)
    t0 = time.time()
    effective_neighbors = min(n_neighbors, max(2, len(X_fit) - 1))
    umap_kwargs = dict(
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        n_components=2,
        compatibility_layout=compatibility_layout,
        random_state=random_state,
    )
    if precomputed_knn is not None:
        knn_idx, knn_dst, knn_search = precomputed_knn
        if knn_idx.shape[1] > effective_neighbors:
            knn_idx = np.ascontiguousarray(knn_idx[:, :effective_neighbors])
            knn_dst = np.ascontiguousarray(knn_dst[:, :effective_neighbors], dtype=np.float32)
        else:
            knn_idx = np.ascontiguousarray(knn_idx)
            knn_dst = np.ascontiguousarray(knn_dst, dtype=np.float32)
        umap_kwargs["precomputed_knn"] = (knn_idx, knn_dst, knn_search)
    reducer = umap.UMAP(**umap_kwargs)
    embedding_fit = reducer.fit_transform(X_fit)
    if use_subsample:
        embedding = np.empty((n_rows, 2), dtype=np.float32)
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            embedding[start:end] = reducer.transform(X_scaled[start:end])
    else:
        embedding = embedding_fit
    logger.info("UMAP finished in %.2fs!", time.time() - t0)

    out = df.copy()
    out["umap_1"] = embedding[:, 0]
    out["umap_2"] = embedding[:, 1]

    fit_coords_64 = np.ascontiguousarray(embedding_fit, dtype=np.float64)
    if cluster_method == "hdbscan":
        eff_min_cluster_size = max(min_cluster_size, int(0.005 * len(fit_coords_64))) if len(fit_coords_64) > 50_000 else min_cluster_size
        eff_min_samples = max(min_samples, int(0.0002 * len(fit_coords_64))) if len(fit_coords_64) > 50_000 else min_samples
        selection_method = "leaf" if len(fit_coords_64) > 50_000 else "eom"
        try:
            import fast_hdbscan
            logger.info(
                "Running parallel fast_hdbscan (mcs=%d, ms=%d, method=%s)...",
                eff_min_cluster_size, eff_min_samples, selection_method,
            )
            hdb = fast_hdbscan.HDBSCAN(
                min_cluster_size=eff_min_cluster_size,
                min_samples=eff_min_samples,
                cluster_selection_method=selection_method,
            )
            fit_clusters = hdb.fit_predict(fit_coords_64)
        except ImportError:
            logger.info(
                "Running Hierarchical Density-Based Clustering (sklearn HDBSCAN, mcs=%d, ms=%d, method=%s, n_jobs=-1)...",
                eff_min_cluster_size, eff_min_samples, selection_method,
            )
            hdb = HDBSCAN(
                min_cluster_size=eff_min_cluster_size,
                min_samples=eff_min_samples,
                cluster_selection_method=selection_method,
                n_jobs=-1,
                copy=True,
            )
            fit_clusters = hdb.fit_predict(fit_coords_64)
        # Assign remaining noise (-1) points to nearest dense cluster core when leaf selection leaves unclustered points
        non_noise = fit_clusters >= 0
        if non_noise.any() and (~non_noise).any() and len(fit_coords_64) > 50_000:
            noise_knn = KNeighborsClassifier(n_neighbors=min(15, int(non_noise.sum())))
            noise_knn.fit(fit_coords_64[non_noise], fit_clusters[non_noise])
            fit_clusters = fit_clusters.copy()
            fit_clusters[~non_noise] = noise_knn.predict(fit_coords_64[~non_noise])
    elif cluster_method == "agglomerative":
        logger.info("Running Hierarchical Agglomerative Clustering (Ward, k=%d)...", n_clusters)
        k_conn = min(15, max(1, len(fit_coords_64) - 1))
        connectivity = kneighbors_graph(fit_coords_64, n_neighbors=k_conn, include_self=False)
        agg = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity, linkage="ward")
        fit_clusters = agg.fit_predict(fit_coords_64)
    else:
        raise ValueError(f"Unsupported cluster_method: {cluster_method!r}. Expected 'hdbscan' or 'agglomerative'.")

    knn_clf = KNeighborsClassifier(n_neighbors=min(15, len(fit_coords_64)))
    knn_clf.fit(fit_coords_64, fit_clusters)
    if use_subsample:
        clusters = np.empty(n_rows, dtype=fit_clusters.dtype)
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            clusters[start:end] = knn_clf.predict(np.ascontiguousarray(embedding[start:end], dtype=np.float64))
        out["cluster"] = clusters
    else:
        out["cluster"] = fit_clusters

    lof_neighbors = min(100 if len(fit_coords_64) > 50_000 else 20, max(1, len(fit_coords_64) - 1))
    lof_contamination = min(contamination, 0.002) if len(fit_coords_64) > 50_000 else contamination
    logger.info("Running Local Outlier Factor on 2D UMAP space (n_neighbors=%d, contamination=%s, n_jobs=-1)...", lof_neighbors, lof_contamination)
    lof_novelty = LocalOutlierFactor(n_neighbors=lof_neighbors, contamination=lof_contamination, novelty=True, n_jobs=-1)
    lof_novelty.fit(fit_coords_64)
    if use_subsample:
        coords_full_64 = np.ascontiguousarray(embedding, dtype=np.float64)
        out["is_local_anomaly"], out["local_anomaly_score"] = _batched_predict_and_score(lof_novelty, coords_full_64, batch_size)
    else:
        lof = LocalOutlierFactor(n_neighbors=lof_neighbors, contamination=lof_contamination, n_jobs=-1)
        out["is_local_anomaly"] = lof.fit_predict(fit_coords_64) == -1
        out["local_anomaly_score"] = -lof.negative_outlier_factor_

    logger.info("Running Global Anomaly Detection (Isolation Forest, contamination=%s, by_vessel_category=%s, n_jobs=-1)...", contamination, by_vessel_category)
    iso_global = IsolationForest(n_estimators=100, contamination=contamination, n_jobs=-1, random_state=random_state)
    iso_global.fit(X_fit)
    iso_by_category: dict[str, IsolationForest] = {}

    if by_vessel_category and "shiptypeais" in out.columns:
        out["vessel_category"] = map_shiptype_to_category(out["shiptypeais"])
        global_anom = np.zeros(n_rows, dtype=bool)
        global_score = np.zeros(n_rows, dtype=np.float64)
        for cat, idx in out.groupby("vessel_category").groups.items():
            pos = out.index.get_indexer(idx)
            if len(pos) >= 100:
                cat_X = X_scaled[pos]
                if fit_sample_size is not None and len(cat_X) > fit_sample_size:
                    cat_fit_idx = np.random.default_rng(random_state).choice(len(cat_X), size=fit_sample_size, replace=False)
                    cat_X_fit = cat_X[cat_fit_idx]
                else:
                    cat_X_fit = cat_X
                cat_iso = IsolationForest(n_estimators=100, contamination=contamination, n_jobs=-1, random_state=random_state)
                cat_iso.fit(cat_X_fit)
                iso_by_category[cat] = cat_iso
                global_anom[pos], global_score[pos] = _batched_predict_and_score(cat_iso, cat_X, batch_size)
            else:
                global_anom[pos], global_score[pos] = _batched_predict_and_score(iso_global, X_scaled[pos], batch_size)
        out["is_global_anomaly"] = global_anom
        out["global_anomaly_score"] = global_score
    else:
        out["is_global_anomaly"], out["global_anomaly_score"] = _batched_predict_and_score(iso_global, X_scaled, batch_size)

    conditions = [
        out["is_local_anomaly"] & out["is_global_anomaly"],
        out["is_global_anomaly"] & (~out["is_local_anomaly"]),
        out["is_local_anomaly"] & (~out["is_global_anomaly"]),
    ]
    choices = ["extreme_both", "global_outlier", "local_outlier"]
    out["anomaly_type"] = np.select(conditions, choices, default="normal")

    geometries = [
        LineString([(lon1, lat1), (lon2, lat2)])
        for lon1, lat1, lon2, lat2 in zip(out["lon_start"], out["lat_start"], out["lon_end"], out["lat_end"])
    ]
    gdf = gpd.GeoDataFrame(out, geometry=geometries, crs="EPSG:4326")
    if return_model:
        model_bundle = {
            "scaler": scaler,
            "reducer": reducer,
            "cluster_classifier": knn_clf,
            "lof": lof_novelty,
            "iso": iso_global,
            "iso_by_category": iso_by_category,
            "by_vessel_category": by_vessel_category,
            "include_vessel_type": include_vessel_type,
            "feature_cols": used_feature_cols,
            "cluster_method": cluster_method,
        }
        return gdf, model_bundle
    return gdf


def transform_with_behavior_model(
    df: pd.DataFrame,
    model_bundle: dict,
    batch_size: int = 100_000,
) -> gpd.GeoDataFrame:
    """Map new behavioral windows into a previously fitted UMAP/clustering/anomaly model."""
    _validate_window_df(df)
    X, _ = _build_feature_matrix(df, include_vessel_type=model_bundle.get("include_vessel_type", False))
    X_scaled = model_bundle["scaler"].transform(X)

    n_rows = len(df)
    embedding = np.empty((n_rows, 2), dtype=np.float32)
    clusters = np.empty(n_rows, dtype=np.int64)
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        emb_chunk = model_bundle["reducer"].transform(X_scaled[start:end])
        embedding[start:end] = emb_chunk
        clusters[start:end] = model_bundle["cluster_classifier"].predict(
            np.ascontiguousarray(emb_chunk, dtype=np.float64)
        )

    out = df.copy()
    out["umap_1"] = embedding[:, 0]
    out["umap_2"] = embedding[:, 1]
    out["cluster"] = clusters
    out["is_local_anomaly"], out["local_anomaly_score"] = _batched_predict_and_score(
        model_bundle["lof"], np.ascontiguousarray(embedding, dtype=np.float64), batch_size
    )
    iso_by_cat = model_bundle.get("iso_by_category") or {}
    if model_bundle.get("by_vessel_category") and iso_by_cat and "shiptypeais" in out.columns:
        out["vessel_category"] = map_shiptype_to_category(out["shiptypeais"])
        global_anom = np.zeros(n_rows, dtype=bool)
        global_score = np.zeros(n_rows, dtype=np.float64)
        for cat, idx in out.groupby("vessel_category").groups.items():
            pos = out.index.get_indexer(idx)
            model = iso_by_cat.get(cat, model_bundle["iso"])
            global_anom[pos], global_score[pos] = _batched_predict_and_score(model, X_scaled[pos], batch_size)
        out["is_global_anomaly"] = global_anom
        out["global_anomaly_score"] = global_score
    else:
        out["is_global_anomaly"], out["global_anomaly_score"] = _batched_predict_and_score(
            model_bundle["iso"], X_scaled, batch_size
        )

    conditions = [
        out["is_local_anomaly"] & out["is_global_anomaly"],
        out["is_global_anomaly"] & (~out["is_local_anomaly"]),
        out["is_local_anomaly"] & (~out["is_global_anomaly"]),
    ]
    choices = ["extreme_both", "global_outlier", "local_outlier"]
    out["anomaly_type"] = np.select(conditions, choices, default="normal")

    geometries = [
        LineString([(lon1, lat1), (lon2, lat2)])
        for lon1, lat1, lon2, lat2 in zip(out["lon_start"], out["lat_start"], out["lon_end"], out["lat_end"])
    ]
    return gpd.GeoDataFrame(out, geometry=geometries, crs="EPSG:4326")


def plot_behavior_and_anomalies(
    gdf: gpd.GeoDataFrame,
    output_path: Path | str,
    cluster_method: str = "hdbscan",
    polygons_file: Optional[Path | str] = None,
    max_plot_points: int = 100_000,
) -> Path:
    """Generate a 4-panel visual dashboard: UMAP, anomalies, spatial clusters, spatial anomalies."""
    png_path = Path(output_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    plot_gdf = gdf
    if len(gdf) > max_plot_points:
        logger.info("Subsampling %s windows to %s for plotting...", f"{len(gdf):,}", f"{max_plot_points:,}")
        plot_gdf = gdf.sample(n=max_plot_points, random_state=42)

    logger.info("Generating 4-panel visual dashboard at %s...", png_path)
    fig, axes = plt.subplots(2, 2, figsize=(22, 16), dpi=300)

    unique_clusters = sorted(c for c in plot_gdf["cluster"].unique() if c >= 0)
    cmap_base = matplotlib.colormaps["tab20"].resampled(max(len(unique_clusters), 20))
    cluster_color_map = {c: cmap_base(i % 20)[:3] for i, c in enumerate(unique_clusters)}
    cluster_color_map[-1] = (0.45, 0.45, 0.45)
    point_colors = [cluster_color_map.get(c, (0.5, 0.5, 0.5)) for c in plot_gdf["cluster"]]

    ax0 = axes[0, 0]
    ax0.scatter(plot_gdf["umap_1"], plot_gdf["umap_2"], c=point_colors, s=4, alpha=0.25, edgecolors="none")
    ax0.set_title(
        f"A. UMAP Gedragsruimte ({len(unique_clusters)} Hiërarchische Clusters - {cluster_method.upper()})",
        fontsize=13,
        fontweight="bold",
    )
    ax0.set_xlabel("UMAP 1", fontsize=11)
    ax0.set_ylabel("UMAP 2", fontsize=11)
    ax0.grid(True, linestyle="--", alpha=0.3)

    ax1 = axes[0, 1]
    norm_mask = plot_gdf["anomaly_type"] == "normal"
    ax1.scatter(plot_gdf.loc[norm_mask, "umap_1"], plot_gdf.loc[norm_mask, "umap_2"], c="#d0d0d0", s=3, alpha=0.4, label="Normaal gedrag")
    for atype, color, umap_label, _ in _ANOMALY_PLOT_SPECS:
        mask = plot_gdf["anomaly_type"] == atype
        if mask.any():
            ax1.scatter(
                plot_gdf.loc[mask, "umap_1"], plot_gdf.loc[mask, "umap_2"],
                c=color, s=25, alpha=0.9, edgecolors="black", linewidths=0.5,
                label=f"{umap_label} (N={int((gdf['anomaly_type'] == atype).sum()):,})",
            )
    ax1.set_title("B. Lokale & Globale Anomalieën in UMAP-ruimte", fontsize=13, fontweight="bold")
    ax1.set_xlabel("UMAP 1", fontsize=11)
    ax1.set_ylabel("UMAP 2", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper right", frameon=True, fontsize=10)

    poly_gdf = None
    if polygons_file is not None and Path(polygons_file).exists():
        try:
            poly_gdf = gpd.read_file(polygons_file)
        except Exception as exc:
            logger.warning("Could not load polygons for plotting: %s", exc)

    ax2 = axes[1, 0]
    if poly_gdf is not None:
        poly_gdf.boundary.plot(ax=ax2, color="blue", linewidth=1.0, linestyle="--", label="Ankergebieden (PDOK)")
    plot_gdf.plot(color=point_colors, linewidth=0.5, alpha=0.15, ax=ax2)
    ax2.set_title("C. Geografische Spreiding per Gedragscluster", fontsize=13, fontweight="bold")
    ax2.set_xlim(2.5, 7.2)
    ax2.set_ylim(51.2, 55.6)
    ax2.set_xlabel("Lengtegraad (°E)", fontsize=11)
    ax2.set_ylabel("Breedtegraad (°N)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.3)

    ax3 = axes[1, 1]
    if poly_gdf is not None:
        poly_gdf.boundary.plot(ax=ax3, color="blue", linewidth=1.0, linestyle="--", label="Ankergebieden")
    if norm_mask.any():
        plot_gdf.loc[norm_mask].plot(ax=ax3, color="#cccccc", linewidth=0.4, alpha=0.08)
    for atype, color, _, map_label in _ANOMALY_PLOT_SPECS:
        sub_anom = plot_gdf[plot_gdf["anomaly_type"] == atype]
        if not sub_anom.empty:
            sub_anom.plot(
                ax=ax3, color=color, linewidth=1.4, alpha=0.55,
                label=f"{map_label} (N={int((gdf['anomaly_type'] == atype).sum()):,})",
            )
    ax3.set_title("D. Geografische Kaart van Afwijkend Vaargedrag (Anomalieën)", fontsize=13, fontweight="bold")
    ax3.set_xlim(2.5, 7.2)
    ax3.set_ylim(51.2, 55.6)
    ax3.set_xlabel("Lengtegraad (°E)", fontsize=11)
    ax3.set_ylabel("Breedtegraad (°N)", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.3)
    ax3.legend(loc="upper right", frameon=True, fontsize=10)

    plt.suptitle(
        "AIS 10-Minuten Vaargedrag & Anomaliedetectie (Thesis Van den Heuvel Methode)",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Dashboard map saved: %s", png_path)
    return png_path


def run_umap_behavior(
    windows_file: Path | str,
    output_dir: Path | str,
    polygons_file: Optional[Path | str] = None,
    cluster_method: str = "hdbscan",
    n_clusters: int = 8,
    contamination: float = 0.015,
    fit_sample_size: Optional[int] = 100_000,
    compatibility_layout: bool = False,
    by_vessel_category: bool = True,
    include_vessel_type: bool = True,
    knn_file: Optional[Path | str] = None,
    save_model: Optional[Path | str] = None,
    model_file: Optional[Path | str] = None,
    plot: bool = True,
) -> gpd.GeoDataFrame:
    """Run UMAP clustering and anomaly detection on a prepared behavioral windows Parquet file."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading prepared behavioral windows from %s...", windows_file)
    df = pd.read_parquet(windows_file)

    if model_file is not None:
        logger.info("Loading fitted UMAP/anomaly model from %s...", model_file)
        model_bundle = joblib.load(model_file)
        gdf = transform_with_behavior_model(df, model_bundle)
        effective_cluster_method = model_bundle.get("cluster_method", cluster_method)
    else:
        precomputed_knn = load_knn(knn_file, expected_rows=len(df)) if knn_file is not None else None
        result = run_umap_and_anomalies(
            df,
            cluster_method=cluster_method,
            n_clusters=n_clusters,
            contamination=contamination,
            fit_sample_size=fit_sample_size,
            compatibility_layout=compatibility_layout,
            by_vessel_category=by_vessel_category,
            include_vessel_type=include_vessel_type,
            precomputed_knn=precomputed_knn,
            return_model=(save_model is not None),
        )
        if save_model is not None:
            gdf, model_bundle = result
            model_path = Path(save_model)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model_bundle, model_path)
            logger.info("Fitted UMAP/anomaly model saved to %s", model_path)
        else:
            gdf = result
        effective_cluster_method = cluster_method

    parquet_path = out_dir / "umap_behavior.geoparquet"
    gdf.to_parquet(parquet_path)
    logger.info("Updated GeoParquet saved: %s", parquet_path)

    if plot:
        plot_behavior_and_anomalies(
            gdf,
            output_path=out_dir / "umap_behavior_map.png",
            cluster_method=effective_cluster_method,
            polygons_file=polygons_file,
        )
    return gdf


def run_aligned_umap(
    windows_files: list[Path | str],
    output_dir: Path | str,
    vessel_id_col: str = "mmsi_hash",
    sample_per_slice: Optional[int] = 50_000,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    alignment_regularisation: float = 0.01,
    alignment_window_size: int = 3,
    compatibility_layout: bool = False,
    include_vessel_type: bool = True,
    random_state: int = 42,
) -> list[Path]:
    """Jointly embed time-ordered behavioral window slices using ``umap.AlignedUMAP``."""
    if len(windows_files) < 2:
        raise ValueError("AlignedUMAP requires at least 2 time-ordered windows files.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)

    dfs = []
    for f in windows_files:
        df = pd.read_parquet(f)
        _validate_window_df(df)
        if sample_per_slice is not None and len(df) > sample_per_slice:
            idx = np.sort(rng.choice(len(df), size=sample_per_slice, replace=False))
            df = df.iloc[idx].reset_index(drop=True)
        dfs.append(df)

    # Fit shared StandardScaler across all slices
    raw_slices = [_build_feature_matrix(d, include_vessel_type=include_vessel_type)[0] for d in dfs]
    all_X = np.vstack(raw_slices)
    scaler = StandardScaler().fit(all_X)
    slices = [scaler.transform(x) for x in raw_slices]

    # Build adjacent slice relations by matching shared vessels (first occurrence per vessel in each slice)
    relations = []
    for i, (df_a, df_b) in enumerate(zip(dfs[:-1], dfs[1:])):
        if vessel_id_col in df_a.columns and vessel_id_col in df_b.columns:
            map_a = df_a.reset_index().groupby(vessel_id_col)["index"].first().to_dict()
            map_b = df_b.reset_index().groupby(vessel_id_col)["index"].first().to_dict()
            shared = set(map_a).intersection(map_b)
            rel = {int(map_a[v]): int(map_b[v]) for v in shared}
        else:
            rel = {}
        if not rel:
            # Fallback when vessel IDs do not overlap across a boundary: link nearest behavioral neighbors
            from sklearn.neighbors import NearestNeighbors
            n_anchor = min(500, len(slices[i]), len(slices[i + 1]))
            anchor_idx = rng.choice(len(slices[i]), size=n_anchor, replace=False)
            nn = NearestNeighbors(n_neighbors=1).fit(slices[i + 1])
            _, indices = nn.kneighbors(slices[i][anchor_idx])
            rel = {int(a): int(b[0]) for a, b in zip(anchor_idx, indices)}
            logger.info("AlignedUMAP fallback feature-anchor links between slices %d->%d: %d", i, i + 1, len(rel))
        else:
            logger.info("AlignedUMAP relation links between adjacent slices %d->%d: %d", i, i + 1, len(rel))
        relations.append(rel)

    min_slice_len = min(len(s) for s in slices)
    effective_neighbors = min(n_neighbors, max(2, min_slice_len - 1))
    mapper = umap.AlignedUMAP(
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        alignment_regularisation=alignment_regularisation,
        alignment_window_size=alignment_window_size,
        compatibility_layout=compatibility_layout,
        random_state=random_state,
    )
    mapper.fit(slices, relations=relations)

    saved_paths = []
    for i, (src_file, df, emb) in enumerate(zip(windows_files, dfs, mapper.embeddings_)):
        out = df.copy()
        out["umap_1"] = emb[:, 0]
        out["umap_2"] = emb[:, 1]
        geometries = [
            LineString([(lon1, lat1), (lon2, lat2)])
            for lon1, lat1, lon2, lat2 in zip(out["lon_start"], out["lat_start"], out["lon_end"], out["lat_end"])
        ]
        gdf = gpd.GeoDataFrame(out, geometry=geometries, crs="EPSG:4326")
        out_path = out_dir / f"aligned_{i:02d}_{Path(src_file).stem}.geoparquet"
        gdf.to_parquet(out_path)
        saved_paths.append(out_path)
        logger.info("Saved aligned embedding slice %d to %s", i, out_path)

    return saved_paths
