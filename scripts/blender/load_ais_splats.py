"""Standalone Blender 5.3+ script to import AIS Gaussian Splat PLY point clouds.

Usage:
    blender --python scripts/blender/load_ais_splats.py -- /path/to/ais_splat.ply
"""

import sys
from pathlib import Path

import bpy


def clear_default_scene() -> None:
    """Remove default cube/light/camera if present."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_gaussian_splat_ply(ply_path: Path) -> bpy.types.Object:
    """Import a 3D Gaussian Splat .ply using Blender 5.3+ native PLY importer."""
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    bpy.ops.wm.ply_import(filepath=str(ply_path))
    obj = bpy.context.active_object
    if obj is None:
        raise RuntimeError(f"Blender failed to import {ply_path}")
    obj.name = ply_path.stem
    return obj


def setup_camera_and_render(target_obj: bpy.types.Object) -> None:
    """Frame the imported point cloud with a top-angled camera and EEVEE/Cycles settings."""
    cam_data = bpy.data.cameras.new(name="AIS_Camera")
    cam_obj = bpy.data.objects.new("AIS_Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = (0.0, -35.0, 28.0)

    constraint = cam_obj.constraints.new(type="TRACK_TO")
    constraint.target = target_obj
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    bpy.context.scene.camera = cam_obj
    world = bpy.context.scene.world
    if world and world.use_nodes:
        bg = world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs[0].default_value = (0.02, 0.03, 0.05, 1.0)


def main() -> None:
    argv = sys.argv
    if "--" not in argv or len(argv[argv.index("--") + 1 :]) < 1:
        raise SystemExit("Usage: blender --python scripts/blender/load_ais_splats.py -- <path_to_splat.ply>")

    ply_path = Path(argv[argv.index("--") + 1]).resolve()
    clear_default_scene()
    obj = import_gaussian_splat_ply(ply_path)
    setup_camera_and_render(obj)
    print(f"Loaded AIS Gaussian Splat '{obj.name}' from {ply_path}")


if __name__ == "__main__":
    main()
