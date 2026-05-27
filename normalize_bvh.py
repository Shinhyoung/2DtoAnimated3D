"""Normalize Mixamo-standard BVH files so they retarget cleanly with the
existing pipeline.

Why this exists
---------------
Mixamo's FBX-to-BVH conversion produces files that don't quite match the
shape of BVHs already in this project (walk.bvh, breakdance.bvh,
sillydancing.bvh, flair.bvh). The visible symptom is that motion plays
but limbs swing in subtly wrong directions on our generated rigs.

Inspecting the headers shows three differences:
  * Mixamo bones carry a `mixamorig:` prefix (e.g. `mixamorig:Hips`),
    whereas the working BVHs use plain names (`Hips`).
  * Mixamo uses centimeter scale — Hips at world Y~100, character
    ~160 cm tall. Working BVHs are at meter-ish scale (~0.5 unit tall).
  * Bone rolls baked into the Mixamo FBX don't match the auto-computed
    rolls Blender produces from offset-only BVH input.

This script normalizes each Mixamo BVH so it looks like the working set:
  1. Import the source BVH into a clean Blender scene.
  2. Strip the `mixamorig:` prefix from every bone name.
  3. Uniformly scale the armature so its rest height equals --height.
  4. Apply the object transform destructively (so bone rolls/offsets are
     baked at the new scale; the next BVH export reflects this).
  5. Re-export as BVH at the same FPS and frame range.

The result is a BVH file with plain bone names, meter-ish scale, and
bone roll values recomputed by Blender from the scaled rest pose — i.e.
the same shape the working BVHs already have.

Run inside Blender headless:
  blender --background --python normalize_bvh.py -- --input X.bvh --output Y.bvh
  blender --background --python normalize_bvh.py -- --input-dir IN --output-dir OUT
Optional flags:
  --height FLOAT   Target rest height in mesh units (default 0.5, matches
                   the existing walk.bvh-style scale).
  --overwrite      Replace existing files in --output-dir (default skip).
"""
import argparse
import sys
from pathlib import Path

import bpy


MIXAMO_PREFIX = "mixamorig:"


def _parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--input", help="Single input BVH path")
    p.add_argument("--output", help="Single output BVH path")
    p.add_argument("--input-dir", help="Batch: directory of input BVHs")
    p.add_argument("--output-dir", help="Batch: directory to write outputs")
    p.add_argument("--height", type=float, default=0.5,
                   help="Target armature rest height in mesh units "
                        "(default 0.5; matches walk.bvh-style scale).")
    p.add_argument("--overwrite", action="store_true",
                   help="In batch mode, overwrite existing output files.")
    return p.parse_args(argv)


def _clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _import_bvh(path: Path):
    """Returns the imported armature object and (frame_start, frame_end)."""
    bpy.ops.import_anim.bvh(
        filepath=str(path),
        target="ARMATURE",
        global_scale=1.0,
        rotate_mode="NATIVE",
        update_scene_fps=True,
        update_scene_duration=True,
    )
    arm = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)
    if arm is None:
        raise RuntimeError(f"No armature produced when importing {path}")
    frame_start = int(bpy.context.scene.frame_start)
    frame_end = int(bpy.context.scene.frame_end)
    return arm, frame_start, frame_end


def _strip_prefix(arm):
    """Remove `mixamorig:` prefix from every bone in the armature.

    Renaming on the armature.bones collection is safe — it propagates
    automatically to pose bones, action fcurve data_paths, and any
    constraints referring to these bones."""
    n_renamed = 0
    for b in arm.data.bones:
        if b.name.startswith(MIXAMO_PREFIX):
            b.name = b.name[len(MIXAMO_PREFIX):]
            n_renamed += 1
    return n_renamed


def _armature_rest_height(arm) -> float:
    """Vertical span of the armature in its rest pose, world space."""
    mw = arm.matrix_world
    pts = [mw @ b.head_local for b in arm.data.bones]
    pts += [mw @ b.tail_local for b in arm.data.bones]
    if not pts:
        return 0.0
    zs = [p.z for p in pts]
    return max(zs) - min(zs)


def _scale_and_apply(arm, target_height: float):
    """Uniformly scale the armature so its rest-pose vertical span equals
    target_height, then bake the scale into bone offsets via Apply Scale.

    Apply Scale on an armature with an action rescales the action's
    position keyframes proportionally too, so the Hips translation
    channels of the motion data come out in the new scale automatically."""
    cur = _armature_rest_height(arm)
    if cur < 1e-6:
        print(f"[normalize_bvh] skipped scale: degenerate armature height {cur}")
        return cur, cur
    s = target_height / cur
    arm.scale = (s, s, s)

    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    new = _armature_rest_height(arm)
    return cur, new


def _export_bvh(arm, out_path: Path, frame_start: int, frame_end: int):
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.export_anim.bvh(
        filepath=str(out_path),
        global_scale=1.0,
        frame_start=frame_start,
        frame_end=frame_end,
        rotate_mode="NATIVE",
        root_transform_only=False,
    )


def normalize_one(in_path: Path, out_path: Path, target_height: float):
    _clear_scene()
    arm, fs, fe = _import_bvh(in_path)
    renamed = _strip_prefix(arm)
    h0, h1 = _scale_and_apply(arm, target_height)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _export_bvh(arm, out_path, fs, fe)
    print(f"[normalize_bvh] {in_path.name}: renamed {renamed} bones, "
          f"height {h0:.3f} → {h1:.3f}, frames {fs}-{fe} → {out_path}")


def main():
    args = _parse_args()

    if args.input and args.output:
        normalize_one(Path(args.input), Path(args.output), args.height)
        return

    if args.input_dir and args.output_dir:
        in_dir = Path(args.input_dir)
        out_dir = Path(args.output_dir)
        bvhs = sorted(in_dir.glob("*.bvh"))
        if not bvhs:
            print(f"[normalize_bvh] no .bvh files in {in_dir}")
            return
        for src in bvhs:
            dst = out_dir / src.name
            if dst.is_file() and not args.overwrite:
                print(f"[normalize_bvh] skip (exists): {dst.name}")
                continue
            try:
                normalize_one(src, dst, args.height)
            except Exception as e:
                print(f"[normalize_bvh] FAILED {src.name}: {e}")
        return

    raise SystemExit("Provide either --input AND --output, OR "
                     "--input-dir AND --output-dir.")


if __name__ == "__main__":
    main()
