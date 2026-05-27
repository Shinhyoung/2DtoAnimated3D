"""Gradio web UI for the img2mesh pipeline.

Run: python app.py  →  http://localhost:7860
"""
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import gradio as gr

from remove_bg import remove_background
from generate_mesh import generate_mesh, VALID_BACKENDS, VALID_TEXTURERS
from auto_rig import auto_rig
from apply_motion import apply_motion
from auto_rig import _find_blender  # reuse blender discovery

PROJ_DIR = Path(__file__).resolve().parent
MOTIONS_DIR = PROJ_DIR / "motions"
OUTPUT_BASE = PROJ_DIR / "output"
UI_OUTPUT = OUTPUT_BASE / "ui"
UI_OUTPUT.mkdir(parents=True, exist_ok=True)
_FBX_TO_GLB_SCRIPT = PROJ_DIR / "fbx_to_glb.py"


def _convert_fbx_to_glb(fbx_path: Path, floor_grid: bool = False) -> Path:
    """Convert FBX to GLB via Blender so gr.Model3D can render it.

    The grid/no-grid GLBs live under different file names so toggling the
    checkbox doesn't invalidate the other variant's cache, and the
    appropriate one is reused across pipeline runs when its FBX hasn't
    changed (cheap reload on a re-toggle without rerunning the pipeline)."""
    fbx_path = Path(fbx_path).resolve()
    suffix = "_grid.glb" if floor_grid else ".glb"
    glb_path = fbx_path.with_name(fbx_path.stem + suffix)
    if glb_path.is_file():
        try:
            fbx_mt = fbx_path.stat().st_mtime
            glb_mt = glb_path.stat().st_mtime
            if glb_mt >= fbx_mt - 5:  # 5 s slack for sequential exports
                print(f"[app] reusing cached GLB: {glb_path}")
                return glb_path
        except Exception:
            pass
    cmd = [
        _find_blender(),
        "--background",
        "--python", str(_FBX_TO_GLB_SCRIPT),
        "--",
        "--fbx", str(fbx_path),
        "--glb", str(glb_path),
    ]
    if floor_grid:
        cmd.append("--floor-grid")
    # Match how rig_in_blender.py names the baked texture: "<rigged stem>_color.png".
    # The motion-applied FBX's stem is "<rigged stem>_<motion>"; the motion
    # name has no underscores in our pipeline so peel off only the last
    # segment to recover the rigged stem.
    stem = fbx_path.stem
    if "_" in stem:
        rigged_stem = stem.rsplit("_", 1)[0]
    else:
        rigged_stem = stem
    candidates = [
        (OUTPUT_BASE / "rigged" / f"{rigged_stem}_color.png").resolve(),
        (fbx_path.parent / f"{rigged_stem}_color.png").resolve(),
        fbx_path.with_name(f"{fbx_path.stem}_color.png").resolve(),
    ]
    for tex in candidates:
        if tex.is_file():
            cmd += ["--texture", str(tex)]
            break
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"FBX→GLB conversion failed (exit {res.returncode})")
    return glb_path


def _list_motions():
    if not MOTIONS_DIR.is_dir():
        return []
    return sorted(p.stem for p in MOTIONS_DIR.glob("*.bvh"))


def run_pipeline_from_mesh(mesh_path, stem, motion, log, add, progress,
                           stage_offset=0.6, floor_grid=False):
    """Shared rig + (optional) motion + GLB-preview tail.
    `stage_offset` controls the progress bar starting point so this can be
    chained either after stages 1-2 (image flow) or as the first real stage
    (GLB-only flow)."""
    progress(stage_offset, desc="자동 리깅 중 (Blender)...")
    t0 = time.time()
    rigged = auto_rig(mesh_path, OUTPUT_BASE / "rigged")
    add(f"✅ 자동 리깅: {time.time()-t0:.1f}s")

    if motion and motion != "(none)":
        progress(stage_offset + 0.2, desc=f"모션 적용 중 ({motion})...")
        t0 = time.time()
        anim = apply_motion(rigged, motion, OUTPUT_BASE / "animated", motion_name=motion)
        add(f"✅ 모션 적용 ({motion}): {time.time()-t0:.1f}s")
        final = anim
        out_name = f"{stem}_{motion}.fbx"
    else:
        final = rigged
        out_name = f"{stem}.fbx"
        add("⏭ 모션 적용: 건너뜀 (모션 미선택)")

    download_path = UI_OUTPUT / out_name
    shutil.copy(final, download_path)

    progress(0.95, desc="3D 미리보기 변환 중...")
    try:
        glb_preview = _convert_fbx_to_glb(download_path, floor_grid=floor_grid)
        suffix = " (+grid)" if floor_grid else ""
        add(f"✅ 미리보기 변환{suffix}: {glb_preview.name}")
    except Exception as ce:
        glb_preview = None
        add(f"⚠ 미리보기 변환 실패: {ce}")

    progress(1.0, desc="완료")
    add(f"\n✓ 결과: {out_name}")
    return str(download_path), str(glb_preview) if glb_preview else None


def run_pipeline_ui(image_path, motion, skip_bg, backend, texturer, floor_grid,
                    progress=gr.Progress()):
    if not image_path:
        return None, None, "❌ 이미지를 업로드해주세요."

    backend = (backend or "triposr").lower()
    if backend not in VALID_BACKENDS:
        return None, None, f"❌ 알 수 없는 백엔드: {backend}"

    texturer = (texturer or "mvadapter").lower()
    if texturer not in VALID_TEXTURERS:
        return None, None, f"❌ 알 수 없는 텍스처러: {texturer}"

    src = Path(image_path)
    stem = src.stem
    log = []

    def add(msg):
        log.append(msg)
        return "\n".join(log)

    try:
        # Stage 1: BG removal
        progress(0.05, desc="배경 제거 중...")
        if skip_bg:
            nobg = str(src)
            add("⏭ [1/4] 배경 제거: 건너뜀 (입력이 이미 RGBA)")
        else:
            t0 = time.time()
            nobg = remove_background(str(src), OUTPUT_BASE / "nobg")
            add(f"✅ [1/4] 배경 제거: {time.time()-t0:.1f}s")

        # Stage 2: Mesh (+ optional texture)
        stage2_label = backend if backend == "triposr" else f"{backend} + {texturer}"
        progress(0.30, desc=f"3D 메시 생성 중 ({stage2_label})...")
        t0 = time.time()
        mesh = generate_mesh(
            nobg, OUTPUT_BASE / "mesh",
            format="glb", backend=backend, texturer=texturer,
        )
        add(f"✅ [2/4] 메시 생성 ({stage2_label}): {time.time()-t0:.1f}s")

        # Stages 3-4 + preview (shared)
        download_path, glb_preview = run_pipeline_from_mesh(
            mesh, stem, motion, log, add, progress, stage_offset=0.6,
            floor_grid=floor_grid,
        )
        return download_path, glb_preview, "\n".join(log)

    except FileNotFoundError as e:
        msg = str(e)
        if "Blender" in msg:
            add("❌ Blender 미설치 또는 PATH 미등록. https://www.blender.org/download/")
        else:
            add(f"❌ 파일 없음: {msg}")
        return None, None, "\n".join(log)
    except Exception as e:
        add(f"❌ 실패: {type(e).__name__}: {e}")
        log.append("\n[traceback]")
        log.append(traceback.format_exc())
        return None, None, "\n".join(log)


def run_pipeline_from_glb_ui(mesh_file, motion, floor_grid, progress=gr.Progress()):
    if not mesh_file:
        return None, None, "❌ 3D 메시 파일(.glb/.gltf/.obj)을 업로드해주세요."

    src = Path(mesh_file)
    suffix = src.suffix.lower()
    if suffix not in (".glb", ".gltf", ".obj"):
        return None, None, f"❌ 지원하지 않는 형식: {suffix} (.glb/.gltf/.obj만 가능)"

    stem = src.stem
    log = []

    def add(msg):
        log.append(msg)
        return "\n".join(log)

    add(f"✅ 입력 메시: {src.name} ({suffix[1:].upper()})")
    try:
        download_path, glb_preview = run_pipeline_from_mesh(
            str(src), stem, motion, log, add, progress, stage_offset=0.1,
            floor_grid=floor_grid,
        )
        return download_path, glb_preview, "\n".join(log)
    except FileNotFoundError as e:
        msg = str(e)
        if "Blender" in msg:
            add("❌ Blender 미설치 또는 PATH 미등록. https://www.blender.org/download/")
        else:
            add(f"❌ 파일 없음: {msg}")
        return None, None, "\n".join(log)
    except Exception as e:
        add(f"❌ 실패: {type(e).__name__}: {e}")
        log.append("\n[traceback]")
        log.append(traceback.format_exc())
        return None, None, "\n".join(log)


_DESCRIPTION = """
# K3I — 사진/3D 모델로 애니메이션 FBX 생성

두 가지 입력 모드:
1. **이미지에서 시작** — 사진 → 배경 제거 → 3D 메시(TripoSR / TripoSG) → 자동 리깅 → 모션 적용
2. **3D 모델에서 시작** — 이미 가진 .glb/.gltf/.obj → 자동 리깅 → 모션 적용

3D 메시 백엔드:
- **TripoSR** (기본) — 빠름(~10s), 정점 컬러 보존, ~6GB VRAM
- **TripoSG** — 느림(~30–60s), 지오메트리만(컬러 없음), ~8GB VRAM, 더 깔끔한 토폴로지

권장 입력: 전신 정면 T-pose, Mixamo 호환 휴머노이드 비율.
"""


def _output_panel(label_prefix=""):
    """Common right-side output panel: 3D preview + FBX download + log."""
    preview = gr.Model3D(
        label=f"{label_prefix}3D 미리보기 (회전·줌·애니메이션 재생)",
        height=360, interactive=False, clear_color=[0.15, 0.15, 0.18, 1.0],
    )
    file_out = gr.File(label="결과 FBX 다운로드", interactive=False)
    log_out = gr.Textbox(label="진행 로그", lines=8, interactive=False)
    return preview, file_out, log_out


def build_ui():
    motion_choices = ["(none)"] + _list_motions()
    default_motion = "walk" if "walk" in motion_choices else motion_choices[0]

    with gr.Blocks(title="K3I") as demo:
        gr.Markdown(_DESCRIPTION)
        with gr.Tabs():
            # === Tab 1: image -> full pipeline ===
            with gr.Tab("이미지에서 시작"):
                with gr.Row():
                    with gr.Column(scale=1):
                        inp_img = gr.Image(type="filepath", label="입력 이미지", height=320)
                        inp_backend_a = gr.Dropdown(
                            choices=list(VALID_BACKENDS), value="triposr",
                            label="3D 메시 백엔드 (triposr=빠름·컬러 / triposg=고품질·지오메트리)",
                        )
                        inp_texturer_a = gr.Dropdown(
                            choices=list(VALID_TEXTURERS), value="mvadapter",
                            label="텍스처 (triposg 전용): none=없음 / ortho=직교 투영(빠름) / "
                                  "hunyuan=Hunyuan3D-2 Paint(비상업) / "
                                  "mvadapter=MV-Adapter SDXL(기본, 상업 OK)",
                        )
                        inp_motion_a = gr.Dropdown(choices=motion_choices, value=default_motion,
                                                   label="모션 선택")
                        inp_skip = gr.Checkbox(label="배경 제거 건너뛰기 (RGBA 입력 시)", value=False)
                        inp_grid_a = gr.Checkbox(label="바닥 그리드 표시", value=False)
                        btn_a = gr.Button("실행", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        prev_a, file_a, log_a = _output_panel()
                btn_a.click(
                    fn=run_pipeline_ui,
                    inputs=[inp_img, inp_motion_a, inp_skip, inp_backend_a,
                            inp_texturer_a, inp_grid_a],
                    outputs=[file_a, prev_a, log_a],
                )

            # === Tab 2: GLB/OBJ -> rig + motion only ===
            with gr.Tab("3D 모델에서 시작"):
                with gr.Row():
                    with gr.Column(scale=1):
                        inp_mesh = gr.File(
                            label="3D 메시 업로드 (.glb / .gltf / .obj)",
                            file_types=[".glb", ".gltf", ".obj"],
                            type="filepath",
                        )
                        inp_motion_b = gr.Dropdown(choices=motion_choices, value=default_motion,
                                                   label="모션 선택")
                        inp_grid_b = gr.Checkbox(label="바닥 그리드 표시", value=False)
                        gr.Markdown(
                            "*Mixamo 호환 휴머노이드 비율(T-pose, Z-up)에서 가장 좋은 결과. "
                            "다른 방향이면 Blender에서 미리 회전 후 export 권장.*"
                        )
                        btn_b = gr.Button("실행", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        prev_b, file_b, log_b = _output_panel()
                btn_b.click(
                    fn=run_pipeline_from_glb_ui,
                    inputs=[inp_mesh, inp_motion_b, inp_grid_b],
                    outputs=[file_b, prev_b, log_b],
                )
    return demo


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    build_ui().launch(server_name="127.0.0.1", server_port=port, inbrowser=True)
