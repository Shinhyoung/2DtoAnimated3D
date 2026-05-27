"""img2mesh pipeline: image → background removed → 3D mesh → rigged FBX → animated FBX."""
import argparse
import contextlib
import io
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from remove_bg import remove_background
from generate_mesh import generate_mesh, VALID_BACKENDS, VALID_TEXTURERS
from auto_rig import auto_rig
from apply_motion import apply_motion

PROJ_DIR = Path(__file__).resolve().parent
MOTIONS_DIR = PROJ_DIR / "motions"
OUTPUT_BASE = PROJ_DIR / "output"
LOGS_DIR = OUTPUT_BASE / "logs"
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ============================================================
# Logging + VRAM
# ============================================================
def setup_logger():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = logging.getLogger("pipeline")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    return logger, log_path


def vram_peak_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def reset_vram():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# ============================================================
# Motion lookup
# ============================================================
def list_motions():
    if not MOTIONS_DIR.is_dir():
        return []
    return sorted(p.stem for p in MOTIONS_DIR.glob("*.bvh"))


def resolve_motion(motion_arg):
    if motion_arg is None:
        return None
    p = Path(motion_arg)
    if p.is_file():
        return p
    candidate = MOTIONS_DIR / f"{motion_arg}.bvh"
    if candidate.is_file():
        return candidate
    avail = list_motions()
    raise FileNotFoundError(
        f"'{motion_arg}.bvh'를 motions/ 폴더에서 찾을 수 없습니다. "
        f"사용 가능한 모션: {', '.join(avail) if avail else '(없음)'}"
    )


# ============================================================
# Checkpoint
# ============================================================
def cp_path(input_path):
    return OUTPUT_BASE / f".checkpoint_{Path(input_path).stem}.json"


def load_cp(input_path):
    p = cp_path(input_path)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cp(input_path, completed, backend=None):
    p = cp_path(input_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"input": str(input_path), "stages": sorted(completed)}
    if backend is not None:
        payload["backend"] = backend
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_cp(input_path):
    p = cp_path(input_path)
    if p.is_file():
        try:
            p.unlink()
        except Exception:
            pass


# ============================================================
# Stage runner
# ============================================================
def _line(idx, total, label):
    return f"[{idx}/{total}] {label.ljust(22, '.')}"


def run_stage(idx, total, label, fn, logger):
    print(_line(idx, total, label) + " ", end="", flush=True)
    reset_vram()
    start = time.time()
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            result = fn()
    except Exception as e:
        elapsed = time.time() - start
        print(f"실패 ({elapsed:.1f}s)")
        if captured.getvalue():
            logger.info(f"--- stdout from '{label}' ---\n{captured.getvalue().rstrip()}")
        logger.error(f"Stage [{idx}/{total}] '{label}' failed in {elapsed:.1f}s: {e}")
        raise
    elapsed = time.time() - start
    vram = vram_peak_gb()
    print(f"완료 ({elapsed:.1f}s, VRAM: {vram:.1f}GB)")
    logger.info(f"Stage [{idx}/{total}] '{label}' done in {elapsed:.1f}s, VRAM peak {vram:.2f}GB")
    if captured.getvalue():
        logger.info(f"--- stdout from '{label}' ---\n{captured.getvalue().rstrip()}")
    return result


def skip_stage(idx, total, label, logger):
    print(_line(idx, total, label) + " 건너뜀 (resume)")
    logger.info(f"Stage [{idx}/{total}] '{label}' skipped (resume)")


# ============================================================
# Pipeline
# ============================================================
def run_pipeline(args, image_path, output_path, motion_path, logger):
    image_path = str(Path(image_path).resolve())
    output_path = Path(output_path)
    stem = Path(image_path).stem

    cp = load_cp(image_path) if args.resume else {}
    completed = set(cp.get("stages", []))
    # If the user switches backends between runs, the cached mesh is from the
    # other model and must be re-generated. Downstream stages (rig, motion)
    # consume the mesh so they must also redo.
    cached_backend = cp.get("backend")
    if completed and cached_backend and cached_backend != args.backend:
        for s in ("mesh", "rig", "motion"):
            completed.discard(s)
        logger.info(f"Resume: backend changed ({cached_backend} -> {args.backend}); "
                    "invalidating mesh/rig/motion stages.")

    # Determine stages
    stages = []
    if not args.skip_bg:
        stages.append("bg")
    stages.append("mesh")
    if not args.mesh_only:
        stages.append("rig")
        if not args.rig_only and motion_path is not None:
            stages.append("motion")
    total = len(stages)

    nobg = image_path
    mesh = None
    rigged = None
    final = None
    idx = 0

    if "bg" in stages:
        idx += 1
        out = OUTPUT_BASE / "nobg" / f"{stem}.png"
        if "bg" in completed and out.is_file():
            skip_stage(idx, total, "배경 제거", logger)
            nobg = str(out)
        else:
            nobg = run_stage(idx, total, "배경 제거",
                lambda: remove_background(image_path, OUTPUT_BASE / "nobg"), logger)
            completed.add("bg"); save_cp(image_path, completed, backend=args.backend)

    if "mesh" in stages:
        idx += 1
        out = OUTPUT_BASE / "mesh" / f"{stem}.{args.format}"
        mesh_label = f"메시 생성 ({args.backend})"
        if "mesh" in completed and out.is_file():
            skip_stage(idx, total, mesh_label, logger); mesh = str(out)
        else:
            mesh = run_stage(idx, total, mesh_label,
                lambda: generate_mesh(nobg, OUTPUT_BASE / "mesh",
                                      format=args.format, backend=args.backend,
                                      texturer=args.texturer), logger)
            completed.add("mesh"); save_cp(image_path, completed, backend=args.backend)
        final = mesh

    if "rig" in stages:
        idx += 1
        out = OUTPUT_BASE / "rigged" / f"{stem}.fbx"
        if "rig" in completed and out.is_file():
            skip_stage(idx, total, "자동 리깅", logger); rigged = str(out)
        else:
            rigged = run_stage(idx, total, "자동 리깅",
                lambda: auto_rig(mesh, OUTPUT_BASE / "rigged"), logger)
            completed.add("rig"); save_cp(image_path, completed, backend=args.backend)
        final = rigged

    if "motion" in stages:
        idx += 1
        mname = motion_path.stem
        label = f"모션 적용 ({mname})"
        out = OUTPUT_BASE / "animated" / f"{stem}_{mname}.fbx"
        if "motion" in completed and out.is_file():
            skip_stage(idx, total, label, logger); anim = str(out)
        else:
            anim = run_stage(idx, total, label,
                lambda: apply_motion(rigged, str(motion_path),
                                     OUTPUT_BASE / "animated", motion_name=mname),
                logger)
            completed.add("motion"); save_cp(image_path, completed, backend=args.backend)
        final = anim

        if not args.no_preview:
            # Preview generation is not yet implemented; flag is reserved for future use.
            logger.info("Preview generation skipped (not implemented in this version).")

    if final:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(final, output_path)
        print(f"\n✓ 최종 출력: {output_path}")
        logger.info(f"Final output: {output_path}")

    if not args.keep_intermediate:
        for sub in ("nobg", "mesh", "rigged", "animated"):
            for f in (OUTPUT_BASE / sub).glob(f"{stem}*"):
                try: f.unlink()
                except Exception: pass

    clear_cp(image_path)


# ============================================================
# Batch
# ============================================================
def batch_process(folder, args, motion_path, logger):
    images = sorted(p for p in Path(folder).iterdir()
                    if p.suffix.lower() in SUPPORTED_IMAGE_EXTS)
    if not images:
        print(f"[!] 폴더에서 이미지를 찾지 못함: {folder}")
        return
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ext = args.format if args.mesh_only else "fbx"
    successes, failures = [], []
    for i, img in enumerate(images, 1):
        out_file = output_dir / f"{img.stem}.{ext}"
        print(f"\n=== [{i}/{len(images)}] {img.name} ===")
        try:
            run_pipeline(args, str(img), out_file, motion_path, logger)
            successes.append(img.name)
        except Exception as e:
            logger.exception(f"Batch item failed: {img.name}")
            short = str(e).split("\n")[0][:100]
            print(f"[!] 실패: {short}")
            failures.append((img.name, short))

    print("\n=== 배치 결과 ===")
    print(f"성공: {len(successes)}, 실패: {len(failures)}")
    if failures:
        print("실패 목록:")
        for name, err in failures:
            print(f"  - {name}: {err}")


# ============================================================
# Error handling
# ============================================================
def handle_error(e, logger):
    msg = str(e)
    low = msg.lower()
    if isinstance(e, FileNotFoundError):
        if "Blender" in msg:
            print("\n[!] Blender가 설치되어 있지 않거나 PATH에 없습니다.")
            print("    https://www.blender.org/download/ 에서 설치 후 재시도해주세요.")
            return 1
        if ".bvh" in msg:
            print(f"\n[!] {msg}")
            return 1
    if "out of memory" in low or "outofmemory" in low or "cuda oom" in low:
        try:
            import torch
            if torch.cuda.is_available():
                free, total_mem = torch.cuda.mem_get_info()
                used_gb = (total_mem - free) / (1024 ** 3)
                total_gb = total_mem / (1024 ** 3)
                print(f"\n[!] VRAM 부족: 현재 {used_gb:.1f}GB / {total_gb:.1f}GB.")
                print("    --format obj로 재시도하거나 입력 이미지 해상도를 줄여보세요.")
                return 1
        except Exception:
            pass
    print(f"\n[!] 파이프라인 실패: {msg}")
    logger.exception("Pipeline error")
    return 1


# ============================================================
# CLI
# ============================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="img2mesh 통합 파이프라인: 이미지 → 배경 제거 → 3D 메시 → 리깅 → 모션",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", help="입력 이미지 경로 (단일 파일 또는 폴더)")
    p.add_argument("--output", help="최종 출력 경로 (.fbx/.glb 파일 또는 폴더)")
    p.add_argument("--motion", help="모션 이름(walk/run/idle/wave/jump) 또는 .bvh 경로")
    p.add_argument("--list-motions", action="store_true", help="사용 가능한 모션 목록 출력 후 종료")
    p.add_argument("--skip-bg", action="store_true", help="배경 제거 단계 건너뜀")
    p.add_argument("--mesh-only", action="store_true", help="메시 생성까지만 수행")
    p.add_argument("--rig-only", action="store_true", help="리깅까지만 수행 (모션 단계 건너뜀)")
    p.add_argument("--format", default="glb", choices=["glb", "obj"], help="메시 출력 포맷")
    p.add_argument("--backend", default="triposr", choices=list(VALID_BACKENDS),
                   help="3D 메시 생성 백엔드 (triposr=빠름·컬러, triposg=고품질·지오메트리)")
    p.add_argument("--texturer", default="mvadapter", choices=list(VALID_TEXTURERS),
                   help="텍스처 파이프라인 (triposg 백엔드 전용): "
                        "none=텍스처 없음, ortho=직교 투영(빠름), hunyuan=Hunyuan3D-2 Paint(고품질)")
    p.add_argument("--keep-intermediate", action=argparse.BooleanOptionalAction,
                   default=True, help="중간 결과 파일 보존 (기본 True)")
    p.add_argument("--no-preview", action="store_true",
                   help="모션 미리보기 썸네일 생성 안 함 (현 버전은 미구현)")
    p.add_argument("--resume", action="store_true",
                   help="이전 실행이 중단된 지점부터 재시작 (.checkpoint 사용)")
    return p


def main():
    args = build_parser().parse_args()

    if args.list_motions:
        m = list_motions()
        if m:
            print("사용 가능한 모션:")
            for x in m:
                print(f"  - {x}")
        else:
            print("(motions/ 폴더에 BVH 파일 없음. 'bash download_motions.sh' 실행)")
        return 0

    if not args.input or not args.output:
        build_parser().error("--input과 --output은 필수입니다 (또는 --list-motions).")

    logger, log_path = setup_logger()
    print(f"[log] {log_path}")
    logger.info(f"Pipeline start | input={args.input} output={args.output} "
                f"motion={args.motion} backend={args.backend}")

    motion_path = None
    if args.motion and not args.mesh_only and not args.rig_only:
        try:
            motion_path = resolve_motion(args.motion)
        except FileNotFoundError as e:
            print(f"\n[!] {e}")
            return 1

    input_path = Path(args.input)
    try:
        if input_path.is_dir():
            batch_process(input_path, args, motion_path, logger)
        elif input_path.is_file():
            run_pipeline(args, str(input_path), args.output, motion_path, logger)
        else:
            print(f"[!] 입력 경로를 찾을 수 없음: {input_path}")
            return 1
    except KeyboardInterrupt:
        print("\n[!] 사용자 중단 — --resume으로 재시작 가능")
        return 130
    except Exception as e:
        return handle_error(e, logger)

    return 0


if __name__ == "__main__":
    sys.exit(main())
