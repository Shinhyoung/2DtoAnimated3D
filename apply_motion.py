"""Apply a BVH motion onto a rigged FBX via Blender headless retargeting."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Union

_PROJ_DIR = Path(__file__).resolve().parent
_BLENDER_SCRIPT = _PROJ_DIR / "retarget_in_blender.py"
_DEFAULT_CONFIG = _PROJ_DIR / "retarget_config.json"
_MOTIONS_DIR = _PROJ_DIR / "motions"


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
        "Blender executable not found. Install Blender 3.6+ and either "
        "add it to PATH or set BLENDER_PATH."
    )


def _resolve_motion(motion: Union[str, Path]) -> Path:
    p = Path(motion)
    if p.is_file():
        return p
    candidate = _MOTIONS_DIR / f"{motion}.bvh"
    if candidate.is_file():
        return candidate
    available = sorted(p.stem for p in _MOTIONS_DIR.glob("*.bvh")) if _MOTIONS_DIR.is_dir() else []
    raise FileNotFoundError(
        f"'{motion}.bvh'를 motions/ 폴더에서 찾을 수 없습니다. "
        f"사용 가능한 모션: {', '.join(available) if available else '(none)'}"
    )


def apply_motion(
    rigged_fbx: Union[str, Path],
    motion: Union[str, Path],
    output_dir: Union[str, Path] = "./output/animated",
    motion_name: str = None,
    config_path: Union[str, Path, None] = None,
) -> str:
    rigged_fbx = Path(rigged_fbx).resolve()
    if not rigged_fbx.is_file():
        raise FileNotFoundError(f"Rigged FBX not found: {rigged_fbx}")

    bvh_path = _resolve_motion(motion).resolve()
    if motion_name is None:
        motion_name = bvh_path.stem

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{rigged_fbx.stem}_{motion_name}.fbx"

    config_path = Path(config_path).resolve() if config_path else _DEFAULT_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(f"Retarget config not found: {config_path}")

    blender = _find_blender()
    cmd = [
        blender,
        "--background",
        "--python", str(_BLENDER_SCRIPT),
        "--",
        "--fbx", str(rigged_fbx),
        "--bvh", str(bvh_path),
        "--output", str(out_path),
        "--config", str(config_path),
    ]
    print(f"[apply_motion] {rigged_fbx.name} + {bvh_path.name} -> {out_path}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout:
        sys.stdout.write(res.stdout)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"Blender retarget failed (exit {res.returncode})")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Apply BVH motion onto rigged FBX.")
    parser.add_argument("--input", required=True, help="Rigged FBX path.")
    parser.add_argument("--motion", required=True, help="BVH file path or motion name (e.g. 'walk').")
    parser.add_argument("--output", default="./output/animated")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    apply_motion(args.input, args.motion, args.output, config_path=args.config)


if __name__ == "__main__":
    main()
