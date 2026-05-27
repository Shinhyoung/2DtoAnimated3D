"""Run inside Blender headless: blender --background --python retarget_in_blender.py -- ..."""
import argparse
import json
import sys
from pathlib import Path

import bpy


def _parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--fbx", required=True, help="Rigged FBX (target).")
    p.add_argument("--bvh", required=True, help="Source BVH motion.")
    p.add_argument("--output", required=True, help="Output FBX with animation.")
    p.add_argument("--config", required=True, help="retarget_config.json.")
    return p.parse_args(argv)


def _clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _import_fbx(path: Path):
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.fbx(filepath=str(path))
    new = set(bpy.context.scene.objects) - before
    arm = next((o for o in new if o.type == "ARMATURE"), None)
    meshes = [o for o in new if o.type == "MESH"]
    if arm is None:
        raise RuntimeError(f"No armature in {path}")
    return arm, meshes


def _import_bvh(path: Path):
    before = set(bpy.context.scene.objects)
    bpy.ops.import_anim.bvh(
        filepath=str(path),
        target="ARMATURE",
        global_scale=1.0,
        rotate_mode="NATIVE",
        update_scene_fps=False,
        update_scene_duration=True,
    )
    new = set(bpy.context.scene.objects) - before
    arm = next((o for o in new if o.type == "ARMATURE"), None)
    if arm is None:
        raise RuntimeError(f"BVH import produced no armature: {path}")
    return arm


def _frame_range(arm):
    if not (arm.animation_data and arm.animation_data.action):
        return 1, 1
    fcurves = arm.animation_data.action.fcurves
    if not fcurves:
        return 1, 1
    starts, ends = [], []
    for fc in fcurves:
        if not fc.keyframe_points:
            continue
        starts.append(fc.keyframe_points[0].co[0])
        ends.append(fc.keyframe_points[-1].co[0])
    return int(min(starts)), int(max(ends))


def _setup_constraints(target_arm, source_arm, mapping):
    bpy.ops.object.select_all(action="DESELECT")
    target_arm.select_set(True)
    bpy.context.view_layer.objects.active = target_arm
    bpy.ops.object.mode_set(mode="POSE")

    src_names = set(source_arm.pose.bones.keys())
    tgt_names = set(target_arm.pose.bones.keys())

    # Mixamo BVHs come in two flavors: bones named plainly ("Hips", "Spine"
    # ...) and bones with the FBX-style "mixamorig:" prefix ("mixamorig:Hips"
    # ...). retarget_config.json uses plain names; resolve to whichever form
    # the imported source skeleton actually uses so both flavors retarget.
    def resolve_src(cfg_name):
        if cfg_name in src_names:
            return cfg_name
        prefixed = f"mixamorig:{cfg_name}"
        if prefixed in src_names:
            return prefixed
        return None

    matched = 0
    for src_name, tgt_name in mapping.items():
        actual_src = resolve_src(src_name)
        if actual_src is None or tgt_name not in tgt_names:
            continue
        bone = target_arm.pose.bones[tgt_name]
        c = bone.constraints.new("COPY_ROTATION")
        c.target = source_arm
        c.subtarget = actual_src
        c.target_space = "LOCAL"
        c.owner_space = "LOCAL"
        matched += 1
    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"[retarget] {matched}/{len(mapping)} bone constraints set")


def _bake(target_arm, start_frame, end_frame):
    # Pre-create an action so nla.bake has a place to write keyframes that
    # survives as the armature's active action (otherwise bake output gets
    # orphaned and FBX export sees no animation_data).
    if target_arm.animation_data is None:
        target_arm.animation_data_create()
    action = bpy.data.actions.new(name=f"{target_arm.name}_Retargeted")
    target_arm.animation_data.action = action

    bpy.ops.object.select_all(action="DESELECT")
    target_arm.select_set(True)
    bpy.context.view_layer.objects.active = target_arm
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame
    bpy.ops.nla.bake(
        frame_start=start_frame,
        frame_end=end_frame,
        only_selected=False,
        visual_keying=True,
        clear_constraints=True,
        clear_parents=False,
        use_current_action=True,
        bake_types={"POSE"},
    )
    bpy.ops.object.mode_set(mode="OBJECT")
    # Ensure action stays assigned (bake may reassign).
    target_arm.animation_data.action = action
    fcurve_count = len(action.fcurves)
    print(f"[retarget] baked frames {start_frame}-{end_frame}, action='{action.name}' fcurves={fcurve_count}")


def _delete(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.delete()


def _export_fbx(out_path: Path, target_arm, meshes):
    bpy.ops.object.select_all(action="DESELECT")
    target_arm.select_set(True)
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = target_arm
    bpy.ops.export_scene.fbx(
        filepath=str(out_path),
        use_selection=True,
        object_types={"ARMATURE", "MESH"},
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        axis_forward="-Z",
        axis_up="Y",
        use_armature_deform_only=True,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        mesh_smooth_type="FACE",
        path_mode="COPY",
        embed_textures=True,
    )


def main():
    args = _parse_args()
    fbx_path = Path(args.fbx)
    bvh_path = Path(args.bvh)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    _clear_scene()
    target_arm, meshes = _import_fbx(fbx_path)
    print(f"[retarget] target armature: {target_arm.name}, {len(meshes)} mesh(es)")
    source_arm = _import_bvh(bvh_path)
    print(f"[retarget] source armature: {source_arm.name}")

    start_frame, end_frame = _frame_range(source_arm)
    _setup_constraints(target_arm, source_arm, config["mapping"])
    _bake(target_arm, start_frame, end_frame)
    _delete(source_arm)
    _export_fbx(out_path, target_arm, meshes)
    print(f"[retarget] exported: {out_path}")


if __name__ == "__main__":
    main()
