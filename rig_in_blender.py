"""Run inside Blender headless: blender --background --python rig_in_blender.py -- ..."""
import argparse
import json
import math
import sys
from pathlib import Path

import bmesh
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
    p.add_argument("--voxel-size", type=float, default=0.0065,
                   help="Voxel remesh edge size in mesh units. Smaller preserves "
                        "narrow gaps (e.g. armpit) but increases poly count.")
    p.add_argument("--skip-voxel-remesh", action="store_true",
                   help="Skip the voxel remesh + bake-from-original step entirely "
                        "(use original mesh topology as-is, even when UV-textured).")
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


def _has_uv_textured_material(mesh_obj) -> bool:
    """Detect TRELLIS-style mesh: UV map + material with image texture, no vertex colors."""
    if not mesh_obj.data.uv_layers:
        return False
    if mesh_obj.data.color_attributes:
        return False  # TripoSR-style — handled by vertex-color bake path
    for mat in mesh_obj.data.materials:
        if mat is None or not mat.use_nodes:
            continue
        for n in mat.node_tree.nodes:
            if n.type == "TEX_IMAGE" and n.image is not None:
                return True
    return False


def _voxel_remesh_and_rebake(mesh_obj, image_path, voxel_size=0.008, image_size=2048):
    """For TRELLIS-style meshes: keep a hidden copy of the original (with UV
    + texture), voxel-remesh the working mesh into a clean watertight
    manifold (which loses the UV map), unwrap the remesh, and bake the
    original's diffuse pass onto the new UV via selected-to-active baking."""
    ref = mesh_obj.copy()
    ref.data = mesh_obj.data.copy()
    ref.name = mesh_obj.name + "_ref"
    bpy.context.collection.objects.link(ref)
    n0 = len(mesh_obj.data.vertices)

    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    mod = mesh_obj.modifiers.new(name="VoxelRemesh", type="REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = voxel_size
    mod.adaptivity = 0.0
    mod.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier=mod.name)
    n1 = len(mesh_obj.data.vertices)
    print(f"[rig] voxel remesh: {n0} -> {n1} verts (voxel_size={voxel_size})")

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.15192, island_margin=0.01)
    bpy.ops.object.mode_set(mode="OBJECT")

    img = bpy.data.images.new(f"{mesh_obj.name}_baked",
                              width=image_size, height=image_size, alpha=False)

    new_mat = bpy.data.materials.new(name="VoxelBakeMat")
    if mesh_obj.data.materials:
        mesh_obj.data.materials[0] = new_mat
    else:
        mesh_obj.data.materials.append(new_mat)
    new_mat.use_nodes = True
    nodes = new_mat.node_tree.nodes
    links = new_mat.node_tree.links
    nodes.clear()
    bsdf = nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (0, 0)
    bsdf.inputs["Roughness"].default_value = 0.9
    output = nodes.new("ShaderNodeOutputMaterial"); output.location = (300, 0)
    tex_node = nodes.new("ShaderNodeTexImage"); tex_node.image = img
    tex_node.location = (-300, 0)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    nodes.active = tex_node

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
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.cage_extrusion = max(0.01, voxel_size * 2)
    scene.render.bake.margin = 8

    bpy.ops.object.select_all(action="DESELECT")
    ref.select_set(True)            # source (selected, not active)
    mesh_obj.select_set(True)       # target (active)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.bake(type="DIFFUSE")

    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    img.filepath_raw = str(image_path)
    img.file_format = "PNG"
    img.save()
    print(f"[rig] baked original texture onto remeshed UV -> {image_path}")

    bpy.data.objects.remove(ref, do_unlink=True)
    scene.render.engine = prev_engine
    return str(image_path)


def _recalc_normals_outward(mesh_obj):
    """Final cleanup pass: recalculate outward normals (helps TripoSR meshes;
    voxel-remeshed meshes already have clean outward normals) and enable
    backface culling on materials so the GLB preview path renders correctly."""
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    for mat in mesh_obj.data.materials:
        if mat is not None:
            mat.use_backface_culling = True


def _strip_inward_polys(mesh_obj):
    """Centroid heuristic: remove polygons whose normal points back toward the
    mesh centroid. Cleans up TRELLIS double-shell artifacts when voxel remesh
    is skipped — handles the easy 'pure inverted shell' case while leaving
    concave regions (armpit, between fingers) alone."""
    me = mesh_obj.data
    if not (me.vertices and me.polygons):
        return
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table()
    centroid = mathutils.Vector((0.0, 0.0, 0.0))
    for v in bm.verts:
        centroid += v.co
    centroid /= len(bm.verts)
    inward = []
    n_total = len(bm.faces)
    for f in bm.faces:
        fc = mathutils.Vector((0.0, 0.0, 0.0))
        for v in f.verts:
            fc += v.co
        fc /= len(f.verts)
        outward_dir = fc - centroid
        if outward_dir.length_squared < 1e-12:
            continue
        if f.normal.dot(outward_dir.normalized()) < -0.05:
            inward.append(f)
    if 0 < len(inward) < n_total:
        bmesh.ops.delete(bm, geom=inward, context="FACES")
        bm.to_mesh(me)
        print(f"[rig] stripped {len(inward)}/{n_total} inward-facing polys (skip-remesh cleanup)")
    else:
        print(f"[rig] inward-facing polys: {len(inward)}/{n_total} — no strip")
    bm.free()
    me.update()


def _maybe_decimate(mesh_obj, max_verts=50000):
    """Bone-Heat auto-weighting fails on dense meshes (TRELLIS ~200k verts);
    decimate down to a level the heat algorithm can solve. Lower poly is
    actually preferred for skinned animation anyway."""
    n = len(mesh_obj.data.vertices)
    if n <= max_verts:
        return n
    ratio = max_verts / n
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    mod = mesh_obj.modifiers.new(name="DecimateForRig", type="DECIMATE")
    mod.ratio = ratio
    mod.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=mod.name)
    new_n = len(mesh_obj.data.vertices)
    print(f"[rig] decimated for rigging: {n} -> {new_n} verts (ratio {ratio:.3f})")
    return new_n


def _has_meaningful_weights(mesh_obj):
    """True if any vertex has a non-zero weight in any vertex group."""
    for v in mesh_obj.data.vertices:
        for g in v.groups:
            if g.weight > 0.001:
                return True
    return False


def _parent_with_auto_weights(mesh_obj, arm_obj):
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    print(f"[rig] AUTO weights: {len(mesh_obj.vertex_groups)} groups, "
          f"meaningful={_has_meaningful_weights(mesh_obj)}")
    if _has_meaningful_weights(mesh_obj):
        return

    # Heat weighting failed silently (common on TRELLIS / non-manifold meshes).
    # Drop the empty groups, expand bone envelope coverage, and use envelope
    # weights instead so the FBX exporter has real numbers to write.
    print("[rig] heat weighting produced empty weights — falling back to ENVELOPE")
    for vg in list(mesh_obj.vertex_groups):
        mesh_obj.vertex_groups.remove(vg)
    for b in arm_obj.data.bones:
        blen = (b.tail_local - b.head_local).length
        b.head_radius = max(b.head_radius, blen * 0.4)
        b.tail_radius = max(b.tail_radius, blen * 0.4)
        b.envelope_distance = max(b.envelope_distance, blen * 0.6)

    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")

    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")
    print(f"[rig] ENVELOPE weights: {len(mesh_obj.vertex_groups)} groups, "
          f"meaningful={_has_meaningful_weights(mesh_obj)}")


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
    # Mesh is expected to arrive Z-up (generate_mesh.py applies the rotation
    # for TripoSR; user-supplied GLB/OBJ should already be Z-up).
    is_uv_textured = _has_uv_textured_material(mesh_obj)

    if is_uv_textured and not args.skip_voxel_remesh:
        # TRELLIS-style: voxel-remesh to a clean watertight manifold and bake
        # the original textures onto the new UVs (avoids the inside-visible
        # double-shell problem and keeps texture quality).
        texture_path = out_path.parent / f"{out_path.stem}_color.png"
        _voxel_remesh_and_rebake(mesh_obj, texture_path, voxel_size=args.voxel_size)
    else:
        # TripoSR-style or skip-remesh mode: decimate if huge, no remesh.
        if args.skip_voxel_remesh and is_uv_textured:
            print("[rig] --skip-voxel-remesh: keeping original topology")
            # Mitigate inside-visible by stripping clearly-inward polys + recalc.
            _strip_inward_polys(mesh_obj)
        _maybe_decimate(mesh_obj, max_verts=50000)

    bmin, bmax = _world_bbox(mesh_obj)
    print(f"[rig] mesh bbox: min={bmin} max={bmax}")

    arm_obj = _build_armature(config, bmin, bmax)
    _parent_with_auto_weights(mesh_obj, arm_obj)

    if not is_uv_textured:
        # Vertex-color path (TripoSR): bake VC to a PNG texture.
        texture_path = out_path.parent / f"{out_path.stem}_color.png"
        _bake_vertex_colors_to_texture(mesh_obj, texture_path)

    _recalc_normals_outward(mesh_obj)
    _export_fbx(out_path, mesh_obj, arm_obj)
    print(f"[rig] exported: {out_path}")


if __name__ == "__main__":
    main()
