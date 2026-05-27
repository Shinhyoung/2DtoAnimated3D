"""Convert FBX (with animation) to GLB for in-browser preview.

When --texture <path-to-png> is given, the PNG is bound to each mesh's
material as the Base Color image before glTF export. This works around
Blender's FBX exporter dropping baked textures created via the rig step
(it only reliably exports textures backed by certain image attributes).

Run inside Blender:
  blender --background --python fbx_to_glb.py -- --fbx X.fbx --glb X.glb [--texture X_color.png]
"""
import argparse
import os
import sys
import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
p = argparse.ArgumentParser()
p.add_argument("--fbx", required=True)
p.add_argument("--glb", required=True)
p.add_argument("--texture", default=None,
               help="Optional Base-Color PNG to bind to all mesh materials")
p.add_argument("--floor-grid", action="store_true",
               help="Embed a textured floor grid plane at the character's foot "
                    "level in the exported GLB (off by default).")
args = p.parse_args(argv)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
bpy.ops.import_scene.fbx(filepath=args.fbx)

# Re-enable DQS on the Armature modifier (FBX strips this flag on round-trip).
# DQS preserves volume across joint bends so the armpit/elbow don't open up
# visible holes when the walk animation runs.
for obj in bpy.context.scene.objects:
    if obj.type == "MESH":
        for mod in obj.modifiers:
            if mod.type == "ARMATURE":
                try:
                    mod.use_deform_preserve_volume = True
                except Exception:
                    pass


def _bind_texture(mesh_obj, image):
    """Ensure each material on mesh_obj has an Image Texture node feeding the
    Principled BSDF's Base Color, wired to `image`. Creates the node graph if
    necessary and replaces any existing Base Color link."""
    for mat in mesh_obj.data.materials:
        if mat is None:
            continue
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
            bsdf.location = (0, 0)
            out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
            if out is None:
                out = nt.nodes.new("ShaderNodeOutputMaterial")
                out.location = (300, 0)
            nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        tex_node = next((n for n in nt.nodes
                         if n.type == "TEX_IMAGE" and n.image is image), None)
        if tex_node is None:
            tex_node = nt.nodes.new("ShaderNodeTexImage")
            tex_node.image = image
            tex_node.location = (-300, 0)
        base = bsdf.inputs.get("Base Color")
        if base is not None:
            for link in list(base.links):
                nt.links.remove(link)
            nt.links.new(tex_node.outputs["Color"], base)


if args.texture and os.path.isfile(args.texture):
    img = bpy.data.images.load(args.texture, check_existing=False)
    try:
        img.pack()
    except Exception as e:
        print(f"[fbx_to_glb] could not pack texture: {e}")
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            _bind_texture(obj, img)
    print(f"[fbx_to_glb] bound texture: {args.texture}")

# Make materials double-sided so the resulting glTF has `doubleSided=true`.
# An earlier version enabled backface culling to hide TRELLIS double-shell
# inner walls, but for single-shell meshes (MV-Adapter / Hunyuan / TripoSG
# outputs) it instead culls any face whose normal happens to point inward,
# producing visible holes through the body. Single-shell meshes don't need
# culling — keep both sides visible so flipped normals don't punch holes.
for obj in bpy.context.scene.objects:
    if obj.type == "MESH":
        for mat in obj.data.materials:
            if mat is not None:
                mat.use_backface_culling = False


def _build_floor_grid(divisions=10, tex_size=1024, buffer_ratio=1.0):
    """Add a textured floor plane at the character's foot level for the
    Gradio preview. The plane spans (1 + buffer_ratio)× the character's
    horizontal bbox so animations that swing limbs outward still have a
    floor under them. Tinted center axes (red = world X, blue = world Y
    in Blender / world Z in glTF) make orientation legible at a glance."""
    z_min = float("inf")
    x_min = y_min = float("inf")
    x_max = y_max = float("-inf")
    has_mesh = False
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        has_mesh = True
        mw = obj.matrix_world
        for v in obj.data.vertices:
            p = mw @ v.co
            if p.z < z_min: z_min = p.z
            if p.x < x_min: x_min = p.x
            if p.x > x_max: x_max = p.x
            if p.y < y_min: y_min = p.y
            if p.y > y_max: y_max = p.y
    if not has_mesh:
        return None

    extent = max(x_max - x_min, y_max - y_min, 1.0)
    half = extent * (0.5 + buffer_ratio)
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)

    # --- Grid texture: dark background with bright gridlines + tinted axes
    img = bpy.data.images.new("_floor_grid_tex",
                              width=tex_size, height=tex_size, alpha=True)
    try:
        import numpy as np
        arr = np.full((tex_size, tex_size, 4),
                      [0.09, 0.09, 0.11, 1.0], dtype=np.float32)
        step = tex_size // divisions
        line = [0.48, 0.48, 0.56, 1.0]
        thick = max(1, tex_size // 256)
        for i in range(0, tex_size + 1, step):
            lo = max(0, i - thick // 2)
            hi = min(tex_size, i + (thick - thick // 2) + 1)
            arr[lo:hi, :] = line
            arr[:, lo:hi] = line
        mid = tex_size // 2
        ax_thick = max(2, tex_size // 200)
        arr[mid - ax_thick:mid + ax_thick, :] = [0.85, 0.30, 0.25, 1.0]
        arr[:, mid - ax_thick:mid + ax_thick] = [0.25, 0.50, 0.90, 1.0]
        img.pixels = arr.flatten().tolist()
    except ImportError:
        bg = (0.09, 0.09, 0.11, 1.0)
        ln = (0.48, 0.48, 0.56, 1.0)
        step = tex_size // divisions
        flat = []
        for y in range(tex_size):
            for x in range(tex_size):
                if x % step == 0 or y % step == 0:
                    flat.extend(ln)
                else:
                    flat.extend(bg)
        img.pixels = flat
    img.pack()

    # --- Plane mesh at floor height. Created in OBJECT mode so we don't
    # accidentally clobber any selection state used by later steps.
    bpy.ops.mesh.primitive_plane_add(size=2 * half,
                                     location=(center_x, center_y, z_min))
    plane = bpy.context.active_object
    plane.name = "_floor_grid"

    mat = bpy.data.materials.new(name="_floor_grid_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.location = (-400, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    bsdf.inputs["Roughness"].default_value = 1.0
    # Self-illuminate the gridlines a touch so the floor is visible even
    # if the previewer doesn't supply much ambient light.
    for socket_name in ("Emission Color", "Emission"):
        sock = bsdf.inputs.get(socket_name)
        if sock is not None:
            nt.links.new(tex.outputs["Color"], sock)
            break
    strength = bsdf.inputs.get("Emission Strength")
    if strength is not None:
        try:
            strength.default_value = 0.4
        except Exception:
            pass
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    plane.data.materials.append(mat)
    # Double-sided so the grid is visible from below if the viewer ever
    # ends up under the floor (some default Model3D cameras start low).
    mat.use_backface_culling = False
    print(f"[fbx_to_glb] floor grid: z={z_min:.3f}, half_extent={half:.2f}, "
          f"divisions={divisions}")
    return plane


if args.floor_grid:
    _build_floor_grid()


bpy.ops.export_scene.gltf(
    filepath=args.glb,
    export_format="GLB",
    export_animations=True,
    export_yup=True,
    export_apply=False,
    export_skins=True,
    export_morph=False,
    export_image_format="AUTO",
    export_force_sampling=True,
    export_frame_range=False,
)
print(f"[fbx_to_glb] {args.fbx} -> {args.glb}")
