"""Standalone Blender 5.3+ script to import and optionally animate AIS Gaussian Splat PLY point clouds.

Usage:
    # Single PLY:
    blender --python scripts/blender/load_ais_splats.py -- /path/to/ais_splat.ply

    # Time-series directory of PLYs (e.g. aligned_00.ply .. aligned_06.ply):
    blender --python scripts/blender/load_ais_splats.py -- /path/to/aligned_ply_dir/
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


def animate_ply_sequence(objects: list[bpy.types.Object], frames_per_slice: int = 24) -> None:
    """Keyframe viewport/render visibility across a time-ordered sequence of PLY objects."""
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = max(1, len(objects) * frames_per_slice)

    for idx, obj in enumerate(objects):
        start_f = idx * frames_per_slice + 1
        end_f = (idx + 1) * frames_per_slice

        # Hidden before slice starts
        if start_f > 1:
            obj.hide_viewport = True
            obj.hide_render = True
            obj.keyframe_insert(data_path="hide_viewport", frame=1)
            obj.keyframe_insert(data_path="hide_render", frame=1)

        # Visible during slice
        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert(data_path="hide_viewport", frame=start_f)
        obj.keyframe_insert(data_path="hide_render", frame=start_f)

        # Hidden after slice ends (except last slice stays visible at frame_end)
        if idx < len(objects) - 1:
            obj.hide_viewport = True
            obj.hide_render = True
            obj.keyframe_insert(data_path="hide_viewport", frame=end_f + 1)
            obj.keyframe_insert(data_path="hide_render", frame=end_f + 1)


def setup_camera_and_render(target_obj: bpy.types.Object) -> None:
    """Frame the imported point cloud with a top-angled camera and dark background."""
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
        raise SystemExit("Usage: blender --python scripts/blender/load_ais_splats.py -- <path_to_splat.ply_or_dir>")

    target = Path(argv[argv.index("--") + 1]).resolve()
    clear_default_scene()

    if target.is_dir():
        ply_files = sorted(target.glob("*.ply"))
        if not ply_files:
            raise FileNotFoundError(f"No .ply files found in directory: {target}")
        objects = [import_gaussian_splat_ply(p) for p in ply_files]
        animate_ply_sequence(objects, frames_per_slice=24)
        setup_camera_and_render(objects[0])
        print(f"Loaded and animated {len(objects)} AIS Gaussian Splat slices from {target}")
    else:
        obj = import_gaussian_splat_ply(target)
        setup_camera_and_render(obj)
        print(f"Loaded AIS Gaussian Splat '{obj.name}' from {target}")


if __name__ == "__main__":
    main()
