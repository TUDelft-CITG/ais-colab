"""Export AIS behavioral UMAP & anomaly GeoParquet to LAS 1.2 and 3D Gaussian Splat PLY."""

import logging
import struct
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib
import numpy as np

logger = logging.getLogger(__name__)

_ANOMALY_COLOR_MAP = {
    "normal": np.array([0.45, 0.55, 0.60], dtype=np.float32),
    "local_outlier": np.array([1.0, 0.60, 0.0], dtype=np.float32),
    "global_outlier": np.array([0.95, 0.15, 0.50], dtype=np.float32),
    "extreme_both": np.array([1.0, 0.10, 0.10], dtype=np.float32),
}

_ANOMALY_CLASS_MAP = {
    "normal": 1,
    "local_outlier": 2,
    "global_outlier": 3,
    "extreme_both": 4,
}


def write_gaussian_splat_ply(
    filepath: Path | str,
    positions: np.ndarray,
    rgb_colors: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    opacities: np.ndarray,
) -> Path:
    """Write standard 3D Gaussian Splatting binary PLY format (Blender 5.3+ Point Cloud compatible)."""
    n_points = positions.shape[0]
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    c0 = 0.28209479177387814
    f_dc = ((rgb_colors - 0.5) / c0).astype(np.float32)

    scales = np.clip(scales, 1e-6, 1e6)
    log_scales = np.log(scales).astype(np.float32)

    clipped_opacities = np.clip(opacities, 1e-4, 1.0 - 1e-4)
    logit_opacities = np.log(clipped_opacities / (1.0 - clipped_opacities)).astype(np.float32)

    rot_norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    rot_norms[rot_norms == 0] = 1.0
    rotations = (rotations / rot_norms).astype(np.float32)

    vertex_dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    for i in range(45):
        vertex_dtype.append((f"f_rest_{i}", "f4"))
    vertex_dtype.extend([
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ])

    vertex_data = np.zeros(n_points, dtype=vertex_dtype)
    vertex_data["x"] = positions[:, 0].astype(np.float32)
    vertex_data["y"] = positions[:, 1].astype(np.float32)
    vertex_data["z"] = positions[:, 2].astype(np.float32)
    vertex_data["f_dc_0"] = f_dc[:, 0]
    vertex_data["f_dc_1"] = f_dc[:, 1]
    vertex_data["f_dc_2"] = f_dc[:, 2]
    vertex_data["opacity"] = logit_opacities
    vertex_data["scale_0"] = log_scales[:, 0]
    vertex_data["scale_1"] = log_scales[:, 1]
    vertex_data["scale_2"] = log_scales[:, 2]
    vertex_data["rot_0"] = rotations[:, 0]
    vertex_data["rot_1"] = rotations[:, 1]
    vertex_data["rot_2"] = rotations[:, 2]
    vertex_data["rot_3"] = rotations[:, 3]

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n_points}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    for i in range(45):
        header_lines.append(f"property float f_rest_{i}")
    header_lines.extend([
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header\n",
    ])
    header_bytes = "\n".join(header_lines).encode("ascii")

    logger.info("Writing %s Gaussian splats to %s...", f"{n_points:,}", out_path)
    with open(out_path, "wb") as f:
        f.write(header_bytes)
        f.write(vertex_data.tobytes())
    return out_path


def write_las(
    filepath: Path | str,
    positions: np.ndarray,
    rgb_colors: np.ndarray,
    classifications: Optional[np.ndarray] = None,
) -> Path:
    """Write a LAS 1.2 binary point cloud (Point Data Record Format 2: XYZ + RGB)."""
    n_points = positions.shape[0]
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pos_64 = np.asarray(positions, dtype=np.float64)
    min_xyz = pos_64.min(axis=0) if n_points > 0 else np.zeros(3, dtype=np.float64)
    max_xyz = pos_64.max(axis=0) if n_points > 0 else np.zeros(3, dtype=np.float64)
    scale_xyz = np.array([1e-4, 1e-4, 1e-4], dtype=np.float64)
    offset_xyz = min_xyz

    header_size = 227
    record_length = 26  # LAS 1.2 Point Data Format 2

    header = bytearray(header_size)
    struct.pack_into("<4sHH", header, 0, b"LASF", 0, 0)
    struct.pack_into("<BB", header, 24, 1, 2)  # Version 1.2
    struct.pack_into("<32s32s", header, 26, b"ais-shader", b"ais-shader pointcloud")
    struct.pack_into("<HH", header, 90, 1, 2025)
    struct.pack_into("<HI", header, 94, header_size, header_size)
    struct.pack_into("<IBHI", header, 98, 0, 2, record_length, n_points)
    struct.pack_into("<5I", header, 111, n_points, 0, 0, 0, 0)
    struct.pack_into("<3d", header, 131, *scale_xyz)
    struct.pack_into("<3d", header, 155, *offset_xyz)
    struct.pack_into("<6d", header, 179, max_xyz[0], min_xyz[0], max_xyz[1], min_xyz[1], max_xyz[2], min_xyz[2])

    point_dtype = np.dtype([
        ("x", "<i4"),
        ("y", "<i4"),
        ("z", "<i4"),
        ("intensity", "<u2"),
        ("flags", "u1"),
        ("classification", "u1"),
        ("scan_angle", "i1"),
        ("user_data", "u1"),
        ("point_source_id", "<u2"),
        ("red", "<u2"),
        ("green", "<u2"),
        ("blue", "<u2"),
    ])
    records = np.zeros(n_points, dtype=point_dtype)
    if n_points > 0:
        scaled = np.round((pos_64 - offset_xyz) / scale_xyz).astype(np.int32)
        records["x"] = scaled[:, 0]
        records["y"] = scaled[:, 1]
        records["z"] = scaled[:, 2]
        rgb_16 = (np.clip(rgb_colors, 0.0, 1.0) * 65535.0).astype(np.uint16)
        records["red"] = rgb_16[:, 0]
        records["green"] = rgb_16[:, 1]
        records["blue"] = rgb_16[:, 2]
        records["flags"] = 0b00001001  # return 1 of 1
        records["point_source_id"] = 1
        if classifications is not None:
            records["classification"] = np.asarray(classifications, dtype=np.uint8)

    logger.info("Writing %s LAS 1.2 points to %s...", f"{n_points:,}", out_path)
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(records.tobytes())
    return out_path


def export_behavior_pointcloud(
    input_file: Path | str,
    output_file: Path | str,
    space: str = "geo",
    color_by: str = "anomaly",
    fmt: Optional[str] = None,
) -> Path:
    """Export UMAP behavioral GeoParquet to LAS 1.2 or 3D Gaussian Splat PLY."""
    out_path = Path(output_file)
    resolved_fmt = fmt or out_path.suffix.lstrip(".").lower()
    if resolved_fmt not in {"las", "ply"}:
        raise ValueError(f"Unsupported pointcloud format: {resolved_fmt!r}. Expected 'las' or 'ply'.")

    gdf = gpd.read_parquet(input_file)
    n = len(gdf)

    if space == "geo":
        lon_mid = 0.5 * (gdf["lon_start"].to_numpy() + gdf["lon_end"].to_numpy())
        lat_mid = 0.5 * (gdf["lat_start"].to_numpy() + gdf["lat_end"].to_numpy())
        sog = gdf["sog_mean"].to_numpy()
        center_lon = float(np.median(lon_mid)) if n > 0 else 4.5
        center_lat = float(np.median(lat_mid)) if n > 0 else 53.0
        geo_x = ((lon_mid - center_lon) * 67.0).astype(np.float32)
        geo_y = ((lat_mid - center_lat) * 111.0).astype(np.float32)
        geo_z = (sog * 1.5).astype(np.float32)
        positions = np.stack([geo_x, geo_y, geo_z], axis=1)
    elif space == "umap":
        u1 = gdf["umap_1"].to_numpy()
        u2 = gdf["umap_2"].to_numpy()
        sog = gdf["sog_mean"].to_numpy()
        raw_pos = np.stack([u1, u2, sog], axis=1)
        if n > 0:
            center = raw_pos.mean(axis=0)
            span = max(float(np.ptp(raw_pos, axis=0).max()), 1e-6)
            positions = ((raw_pos - center) / span * 20.0).astype(np.float32)
        else:
            positions = raw_pos.astype(np.float32)
    else:
        raise ValueError(f"Unsupported space: {space!r}. Expected 'geo' or 'umap'.")

    if color_by == "anomaly":
        rgb = np.array(
            [_ANOMALY_COLOR_MAP.get(t, _ANOMALY_COLOR_MAP["normal"]) for t in gdf["anomaly_type"]],
            dtype=np.float32,
        )
    elif color_by == "cluster":
        cmap = matplotlib.colormaps["tab20"].resampled(20)
        clusters = gdf["cluster"].to_numpy()
        rgb = np.array(
            [(0.45, 0.45, 0.45) if c < 0 else cmap(int(c) % 20)[:3] for c in clusters],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Unsupported color_by: {color_by!r}. Expected 'anomaly' or 'cluster'.")

    if resolved_fmt == "las":
        classifications = np.array(
            [_ANOMALY_CLASS_MAP.get(t, 1) for t in gdf["anomaly_type"]],
            dtype=np.uint8,
        )
        return write_las(out_path, positions=positions, rgb_colors=rgb, classifications=classifications)

    is_anom = (gdf["anomaly_type"] != "normal").to_numpy()
    radius = np.where(is_anom, 0.35 if space == "umap" else 0.8, 0.12 if space == "umap" else 0.25).astype(np.float32)
    scales = np.stack([radius, radius, radius], axis=1)
    rotations = np.zeros((n, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    opacities = np.where(is_anom, 0.99, 0.90).astype(np.float32)
    return write_gaussian_splat_ply(
        out_path,
        positions=positions,
        rgb_colors=rgb,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
    )
