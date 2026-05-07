import argparse
import sys
import types
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

_PROJ_DIR = Path(__file__).resolve().parent
_TRIPO_DIR = _PROJ_DIR / "TripoSR"
if str(_TRIPO_DIR) not in sys.path:
    sys.path.insert(0, str(_TRIPO_DIR))


def _install_torchmcubes_shim():
    # Use PyMCubes (CPU) as fallback when torchmcubes CUDA build is unavailable.
    try:
        import torchmcubes  # noqa: F401
        return
    except ImportError:
        pass
    import mcubes

    def marching_cubes(volume, threshold):
        vol_np = volume.detach().cpu().numpy() if hasattr(volume, "detach") else np.asarray(volume)
        verts, faces = mcubes.marching_cubes(vol_np, float(threshold))
        # torchmcubes returns vertices in (z, y, x) order; TripoSR's
        # MarchingCubeHelper applies `v_pos[..., [2, 1, 0]]` to convert back
        # to (x, y, z). PyMCubes returns (x, y, z) directly, so we pre-swap
        # to mimic torchmcubes — without this, color lookups query the wrong
        # spatial location and return near-zero gray colors.
        # The swap also inverts face orientation (it's a reflection), which
        # restores the correct winding without an explicit face[:, [0,2,1]] flip.
        verts = verts[:, [2, 1, 0]]
        return (
            torch.from_numpy(np.ascontiguousarray(verts).astype(np.float32)),
            torch.from_numpy(np.ascontiguousarray(faces).astype(np.int64)),
        )

    shim = types.ModuleType("torchmcubes")
    shim.marching_cubes = marching_cubes
    sys.modules["torchmcubes"] = shim


_install_torchmcubes_shim()

from tsr.system import TSR  # noqa: E402
from tsr.utils import resize_foreground  # noqa: E402

_MODEL = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 12GB VRAM safe defaults.
_DEFAULT_CHUNK_SIZE = 8192
_DEFAULT_MC_RESOLUTION = 256
_DEFAULT_FOREGROUND_RATIO = 0.85


def _load_model():
    global _MODEL
    if _MODEL is None:
        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.renderer.set_chunk_size(_DEFAULT_CHUNK_SIZE)
        model.to(_DEVICE)
        _MODEL = model
    return _MODEL


def _prepare_image(image_path: Path, foreground_ratio: float) -> Image.Image:
    img = Image.open(image_path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    img = resize_foreground(img, foreground_ratio)
    arr = np.array(img).astype(np.float32) / 255.0
    rgb = arr[..., :3] * arr[..., 3:4] + (1 - arr[..., 3:4]) * 0.5
    return Image.fromarray((rgb * 255.0).astype(np.uint8))


def generate_mesh(
    image_path: Union[str, Path],
    output_dir: Union[str, Path] = "./output/mesh",
    format: str = "glb",
    mc_resolution: int = _DEFAULT_MC_RESOLUTION,
    foreground_ratio: float = _DEFAULT_FOREGROUND_RATIO,
    use_fp16: bool = False,
) -> str:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = format.lower().lstrip(".")
    if fmt not in {"glb", "obj"}:
        raise ValueError(f"Unsupported format: {format}")

    model = _load_model()
    image = _prepare_image(image_path, foreground_ratio)

    autocast_dtype = torch.float16 if (use_fp16 and _DEVICE == "cuda") else torch.float32
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if _DEVICE == "cuda" else "cpu",
                            dtype=autocast_dtype, enabled=use_fp16):
            scene_codes = model([image], device=_DEVICE)
        meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=mc_resolution)

    # With the PyMCubes axis-swap shim restoring colors, the mesh in trimesh
    # frame has head along +Z and arm span along +Y. Three rotations land it
    # in the right pose for skeleton_config.json's (X=arms, Y=up, Z=depth) and
    # facing -Y (toward Blender's default camera):
    #  1) Rx(-90°) brings head from trimesh +Z to +Y → Blender +Z (standing).
    #  2) Ry(-90°) (= +90° + 180° about-face) puts arm span on Blender X and
    #     turns the character to face -Y (toward camera).
    import trimesh.transformations as _tf
    rot_x = _tf.rotation_matrix(-np.pi / 2, [1.0, 0.0, 0.0])
    rot_y = _tf.rotation_matrix(-np.pi / 2, [0.0, 1.0, 0.0])
    meshes[0].apply_transform(rot_x)
    meshes[0].apply_transform(rot_y)

    out_path = output_dir / f"{image_path.stem}.{fmt}"
    meshes[0].export(str(out_path))
    print(f"[generate_mesh] {image_path.name} -> {out_path}")

    if _DEVICE == "cuda":
        torch.cuda.empty_cache()

    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Generate 3D mesh from a single image (TripoSR).")
    parser.add_argument("--input", required=True, help="Input RGBA image (background-removed).")
    parser.add_argument("--output", default="./output/mesh", help="Output directory.")
    parser.add_argument("--format", default="glb", choices=["glb", "obj"], help="Mesh format.")
    parser.add_argument("--mc-resolution", type=int, default=_DEFAULT_MC_RESOLUTION,
                        help="Marching cubes resolution (lower=less VRAM).")
    parser.add_argument("--foreground-ratio", type=float, default=_DEFAULT_FOREGROUND_RATIO)
    parser.add_argument("--fp16", action="store_true",
                        help="Enable fp16 autocast (TripoSR may emit dtype mismatch errors).")
    args = parser.parse_args()

    generate_mesh(
        args.input,
        args.output,
        args.format,
        mc_resolution=args.mc_resolution,
        foreground_ratio=args.foreground_ratio,
        use_fp16=args.fp16,
    )


if __name__ == "__main__":
    main()
