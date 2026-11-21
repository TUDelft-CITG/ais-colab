from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from ais_shader.behavior import (
    BEHAVIOR_FEATURE_COLS,
    compute_and_save_knn,
    extract_behavior_windows,
    haversine_np,
    plot_behavior_and_anomalies,
    run_aligned_umap,
    run_umap_and_anomalies,
    run_umap_behavior,
)


def _make_synthetic_ais_points(n_vessels: int = 8, pings_per_vessel: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    base_time = pd.Timestamp("2024-05-01T10:00:00Z")
    for v in range(n_vessels):
        lon0 = 4.0 + v * 0.05
        lat0 = 52.0 + v * 0.02
        for p in range(pings_per_vessel):
            rows.append(
                {
                    "timestamp_recv": base_time + pd.Timedelta(seconds=p * 60),
                    "mmsi_hash": f"vessel_{v}",
                    "longitude": lon0 + p * 0.001,
                    "latitude": lat0 + p * 0.0005,
                    "sog": float(10.0 + (v % 3) * 2.0 + rng.normal(0, 0.2)),
                    "cog": float((45.0 + v * 15.0) % 360.0),
                    "heading": float((47.0 + v * 15.0) % 360.0),
                    "length": 100.0 + v * 5.0,
                    "beam": 16.0 + (v % 3),
                    "acceleration_mps2": float(rng.normal(0, 0.02)),
                    "turn_rate_from_cog": float(rng.normal(0, 0.5)),
                    "rolling_area_m2": float(500.0 + p * 50.0),
                    "shiptypeais": 70,
                }
            )
    return pd.DataFrame(rows)


def test_haversine_np():
    assert haversine_np(4.0, 52.0, 4.0, 52.0) == pytest.approx(0.0, abs=1e-6)
    dist = haversine_np(4.0, 52.0, 4.0, 53.0)
    assert 110_000.0 < dist < 112_000.0


def test_extract_behavior_windows_fails_on_untrajectorized_input(tmp_path: Path):
    raw_pts = _make_synthetic_ais_points(n_vessels=3, pings_per_vessel=6).drop(
        columns=["acceleration_mps2", "turn_rate_from_cog", "rolling_area_m2"]
    )
    parquet_path = tmp_path / "raw_ais.parquet"
    raw_pts.to_parquet(parquet_path, index=False)

    with pytest.raises(ValueError, match="trajectory compute"):
        extract_behavior_windows(parquet_path)


def test_umap_fails_on_unprepared_input(tmp_path: Path):
    raw_pts = _make_synthetic_ais_points(n_vessels=3, pings_per_vessel=6)
    with pytest.raises(ValueError, match="extract-windows"):
        run_umap_and_anomalies(raw_pts)


def test_two_step_extract_windows_and_umap_pipeline(tmp_path: Path):
    df_pts = _make_synthetic_ais_points(n_vessels=15, pings_per_vessel=7)
    traj_path = tmp_path / "trajectorized.parquet"
    df_pts.to_parquet(traj_path, index=False)

    windows_df = extract_behavior_windows(traj_path, min_pings=5)
    assert len(windows_df) == 15
    for col in BEHAVIOR_FEATURE_COLS:
        assert col in windows_df.columns
    assert (windows_df["n_pings"] >= 5).all()
    assert (windows_df["in_anchorage_area"] == 0).all()

    windows_path = tmp_path / "windows.parquet"
    windows_df.to_parquet(windows_path, index=False)

    gdf = run_umap_behavior(
        windows_file=windows_path,
        output_dir=tmp_path / "umap_out",
        cluster_method="agglomerative",
        n_clusters=3,
        contamination=0.1,
        fit_sample_size=10,
        compatibility_layout=False,
        plot=True,
    )
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs.to_epsg() == 4326
    assert {"umap_1", "umap_2", "cluster", "anomaly_type"}.issubset(gdf.columns)
    assert set(gdf["anomaly_type"]).issubset({"normal", "local_outlier", "global_outlier", "extreme_both"})
    assert (tmp_path / "umap_out" / "umap_behavior.geoparquet").exists()
    assert (tmp_path / "umap_out" / "umap_behavior_map.png").exists()


def test_precomputed_knn_and_model_reuse_and_aligned_umap(tmp_path: Path):
    df_pts = _make_synthetic_ais_points(n_vessels=15, pings_per_vessel=7)
    traj_path = tmp_path / "trajectorized.parquet"
    df_pts.to_parquet(traj_path, index=False)

    windows_df = extract_behavior_windows(traj_path, min_pings=5)
    windows_m1 = tmp_path / "m01_windows.parquet"
    windows_m2 = tmp_path / "m02_windows.parquet"
    windows_df.to_parquet(windows_m1, index=False)
    windows_df.to_parquet(windows_m2, index=False)

    # 1. Precomputed KNN + save model (k=16 > default n_neighbors=15 exercises slicing & contiguity)
    knn_path = compute_and_save_knn(windows_m1, tmp_path / "m01_knn.npz", n_neighbors=16)
    assert knn_path.exists()

    model_path = tmp_path / "umap_model.joblib"
    gdf_m1 = run_umap_behavior(
        windows_file=windows_m1,
        output_dir=tmp_path / "m01_umap",
        cluster_method="agglomerative",
        n_clusters=3,
        contamination=0.1,
        knn_file=knn_path,
        save_model=model_path,
        plot=False,
    )
    assert model_path.exists()
    assert len(gdf_m1) == 15

    # 2. Transform month 2 using saved model
    gdf_m2 = run_umap_behavior(
        windows_file=windows_m2,
        output_dir=tmp_path / "m02_umap",
        model_file=model_path,
        plot=False,
    )
    assert len(gdf_m2) == 15
    assert {"umap_1", "umap_2", "cluster", "anomaly_type"}.issubset(gdf_m2.columns)

    # 3. AlignedUMAP across time slices
    aligned_paths = run_aligned_umap(
        [windows_m1, windows_m2],
        output_dir=tmp_path / "aligned",
        n_neighbors=5,
    )
    assert len(aligned_paths) == 2
    assert all(p.exists() for p in aligned_paths)
