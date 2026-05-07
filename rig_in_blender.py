"""Run inside Blender headless: blender --background --python rig_in_blender.py -- ..."""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import mathutils


def _parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--skeleton-type", default="mixamo")
    return p.parse_args(argv)


def _clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _import_mesh(path: Path):
    suf = path.suffix.lower()
    if suf in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suf == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    else:
        raise ValueError(f"Unsupported mesh format: {suf}")
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh found after import")
    return meshes[-1]


def _world_bbox(obj):
    # obj.bound_box is cached and not always refreshed after data.transform();
    # iterate vertices directly for a reliable post-rotation bbox.
    mw = obj.matrix_world
    if obj.type == "MESH" and obj.data.vertices:
        pts = [mw @ v.co for v in obj.data.vertices]
    else:
        pts = [mw @ mathutils.Vector(c) for c in obj.bound_box]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _orient_upright(mesh_obj):
    """Rotate mesh so its longest bbox axis aligns with world Z (Blender's up).
    Applies rotation directly to mesh vertex data so it persists regardless of
    object/parent transforms or operator context."""
    # Bake matrix_world (glTF Y-up→Z-up rotation lives here) into vertex data
    # so subsequent rotations compose correctly in world space.
    mw = mesh_obj.matrix_world.copy()
    mesh_obj.data.transform(mw)
    mesh_obj.matrix_world = mathutils.Matrix.Identity(4)
    if mesh_obj.parent:
        mesh_obj.parent = None

    bmin, bmax = _world_bbox(mesh_obj)
    dims = [bmax[i] - bmin[i] for i in range(3)]
    longest = dims.index(max(dims))
    if longest == 2:
        print("[rig] mesh already Z-up, no orient needed")
        return
    if longest == 0:
        rot = mathutils.Matrix.Rotation(math.radians(90), 4, "Y")
    else:
        rot = mathutils.Matrix.Rotation(math.radians(-90), 4, "X")
    mesh_obj.data.transform(rot)
    mesh_obj.data.update()
    print(f"[rig] auto-oriented: longest axis was {'XYZ'[longest]}, rotated to Z")


def _map_coord(norm, bmin, bmax):
    """Map normalized skeleton coords (Y=up) to Blender world (Z=up)."""
    nx, ny, nz = norm
    cx = (bmin[0] + bmax[0]) * 0.5
    cy = (bmin[1] + bmax[1]) * 0.5
    sx = (bmax[0] - bmin[0]) * 0.5
    sy = (bmax[1] - bmin[1]) * 0.5
    x = cx + nx * sx                              # skeleton X → world X (right)
    y = cy + nz * sy                              # skeleton Z → world Y (depth)
    z = bmin[2] + ny * (bmax[2] - bmin[2])        # skeleton Y → world Z (up)
    return mathutils.Vector((x, y, z))


def _build_armature(config, bmin, bmax):
    bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
    arm_obj = bpy.context.object
    arm_obj.name = "Armature"
    arm = arm_obj.data
    arm.name = "Armature"
    # Remove the default 'Bone' added by armature_add
    for eb in list(arm.edit_bones):
        arm.edit_bones.remove(eb)

    created = {}
    for b in config["bones"]:
        eb = arm.edit_bones.new(b["name"])
        eb.head = _map_coord(b["head"], bmin, bmax)
        eb.tail = _map_coord(b["tail"], bmin, bmax)
        created[b["name"]] = eb
    for b in config["bones"]:
        if b["parent"]:
            created[b["name"]].parent = created[b["parent"]]
            # Connect only when head matches parent's tail to avoid disjoint connection.
            ph = created[b["parent"]].tail
            ch = created[b["name"]].head
            created[b["name"]].use_connect = (ph - ch).length < 1e-5

    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def _parent_with_auto_weights(mesh_obj, arm_obj):
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")


def _bake_vertex_colors_to_texture(mesh_obj, image_path, image_size=1024):
    """Bake the mesh's vertex color attribute to a UV-mapped PNG and rewire
    the material to use that texture. FBX viewers seldom honour vertex colors
    via shader nodes, so this is the only reliable way to make colors visible
    everywhere."""
    color_attrs = mesh_obj.data.color_attributes
    if not color_attrs:
        print("[rig] mesh has no vertex_colors; skipping texture bake")
        return None
    color_name = color_attrs[0].name

    scene = bpy.context.scene
    prev_engine = scene.render.engine
    scene.render.engine = "CYCLES"
    try:
        scene.cycles.device = "GPU"
    except Exception:
        pass
    scene.cycles.samples = 1
    scene.render.bake.use_pass_direct = False
    scene.render.bake.use_pass_indirect = False
    scene.render.bake.use_pass_color = True
    scene.render.bake.margin = 4

    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.15192, island_margin=0.01)
    bpy.ops.object.mode_set(mode="OBJECT")

    img = bpy.data.images.new(f"{mesh_obj.name}_baked",
                              width=image_size, height=image_size, alpha=False)

    if not mesh_obj.data.materials:
        mesh_obj.data.materials.append(bpy.data.materials.new(name="VertexColorMaterial"))
    mat = mesh_obj.data.materials[0]
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    vc_node = nodes.new("ShaderNodeVertexColor")
    vc_node.layer_name = color_name
    vc_node.location = (-500, 200)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    bsdf.inputs["Roughness"].default_value = 0.9
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (400, 0)
    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = img
    tex_node.location = (-500, -200)

    links.new(vc_node.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    nodes.active = tex_node

    bpy.ops.object.bake(type="DIFFUSE")

    # Swap material to read from the baked texture instead of vertex colors.
    for link in list(links):
        if link.to_node == bsdf and link.to_socket.name == "Base Color":
            links.remove(link)
    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

    img.filepath_raw = str(image_path)
    img.file_format = "PNG"
    img.save()
    print(f"[rig] baked vertex colors -> {image_path}")

    scene.render.engine = prev_engine
    return str(image_path)


def _export_fbx(out_path: Path, mesh_obj, arm_obj):
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.export_scene.fbx(
        filepath=str(out_path),
        use_selection=True,
        object_types={"ARMATURE", "MESH"},
        add_leaf_bones=False,
        bake_anim=False,
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
    mesh_path = Path(args.mesh)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    _clear_scene()
    mesh_obj = _import_mesh(mesh_path)
    # Mesh is expected to arrive Z-up (generate_mesh.py applies the +90°-around-Z
    # correction). _orient_upright remains as a safety net for non-TripoSR meshes
    # but is intentionally not called by default.
    bmin, bmax = _world_bbox(mesh_obj)
    print(f"[rig] mesh bbox: min={bmin} max={bmax}")

    arm_obj = _build_armature(config, bmin, bmax)
    _parent_with_auto_weights(mesh_obj, arm_obj)

    texture_path = out_path.parent / f"{out_path.stem}_color.png"
    _bake_vertex_colors_to_texture(mesh_obj, texture_path)

    _export_fbx(out_path, mesh_obj, arm_obj)
    print(f"[rig] exported: {out_path}")


if __name__ == "__main__":
    main()
