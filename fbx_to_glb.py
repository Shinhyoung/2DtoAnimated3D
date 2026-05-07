"""Convert FBX (with animation) to GLB for in-browser preview.
Run inside Blender: blender --background --python fbx_to_glb.py -- --fbx X.fbx --glb X.glb"""
import argparse
import sys
import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
p = argparse.ArgumentParser()
p.add_argument("--fbx", required=True)
p.add_argument("--glb", required=True)
args = p.parse_args(argv)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
bpy.ops.import_scene.fbx(filepath=args.fbx)

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
