"""Auto-rig orchestrator.

Backend choice: Blender (bpy) headless with a pre-defined Mixamo-named bone
template fitted to the mesh AABB. Reasons over RigNet:
  * Stable on RTX 5070 / sm_120 (no PyTorch CUDA-extension dependency).
  * Bone names are Mixamo-conventional out of the box, so the FBX is directly
    usable in Blender retargeting; Mixamo upload may also accept it.
  * Rigify is intentionally avoided: its metarig requires manual placement,
    which defeats the "auto" requirement. The simpler template-fit gives
    deterministic results for T-posed humanoids.
Limitation: assumes a roughly T-posed humanoid mesh oriented with Y-up.
Non-humanoid meshes will get a generic skeleton that may need manual cleanup.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Union

_PROJ_DIR = Path(__file__).resolve().parent
_BLENDER_SCRIPT = _PROJ_DIR / "rig_in_blender.py"
_DEFAULT_CONFIG = _PROJ_DIR / "skeleton_config.json"


def _find_blender() -> str:
    env = os.environ.get("BLENDER_PATH")
    if env and Path(env).is_file():
        return env

    found = shutil.which("blender") or shutil.which("blender.exe")
    if found:
        return found

    candidates = []
    candidates += list(Path("C:/Program Files/Blender Foundation").glob("Blender */blender.exe"))
    candidates += list(Path("C:/Program Files (x86)/Blender Foundation").glob("Blender */blender.exe"))
    candidates += [Path("/Applications/Blender.app/Contents/MacOS/Blender")]
    candidates += [Path("/usr/bin/blender"), Path("/usr/local/bin/blender")]

    for c in sorted(candidates, reverse=True):
        if c.is_file():
            return str(c)

    raise FileNotFoundError(
        "Blender executable not found. Install Blender 3.6+ and either:\n"
        "  - add it to PATH, or\n"
        "  - set the BLENDER_PATH environment variable to blender.exe"
    )


def auto_rig(
    mesh_path: Union[str, Path],
    output_dir: Union[str, Path] = "./output/rigged",
    skeleton_type: str = "mixamo",
    config_path: Union[str, Path, None] = None,
) -> str:
    mesh_path = Path(mesh_path).resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{mesh_path.stem}.fbx"

    config_path = Path(config_path).resolve() if config_path else _DEFAULT_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(f"Skeleton config not found: {config_path}")

    blender = _find_blender()
    cmd = [
        blender,
        "--background",
        "--python", str(_BLENDER_SCRIPT),
        "--",
        "--mesh", str(mesh_path),
        "--output", str(out_path),
        "--config", str(config_path),
        "--skeleton-type", skeleton_type,
    ]
    print(f"[auto_rig] blender: {blender}")
    print(f"[auto_rig] {mesh_path.name} -> {out_path}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout:
        sys.stdout.write(res.stdout)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"Blender failed (exit {res.returncode})")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Auto-rig a mesh with a Mixamo-named skeleton via Blender.")
    parser.add_argument("--input", required=True, help="Input mesh (.glb or .obj).")
    parser.add_argument("--output", default="./output/rigged", help="Output directory.")
    parser.add_argument("--skeleton-type", default="mixamo")
    parser.add_argument("--config", default=None, help="Override skeleton config JSON path.")
    args = parser.parse_args()

    auto_rig(args.input, args.output, args.skeleton_type, args.config)


if __name__ == "__main__":
    main()
