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
from generate_mesh import generate_mesh
from auto_rig import auto_rig
from apply_motion import apply_motion
from auto_rig import _find_blender  # reuse blender discovery

PROJ_DIR = Path(__file__).resolve().parent
MOTIONS_DIR = PROJ_DIR / "motions"
OUTPUT_BASE = PROJ_DIR / "output"
UI_OUTPUT = OUTPUT_BASE / "ui"
UI_OUTPUT.mkdir(parents=True, exist_ok=True)
_FBX_TO_GLB_SCRIPT = PROJ_DIR / "fbx_to_glb.py"


def _convert_fbx_to_glb(fbx_path: Path) -> Path:
    """Convert FBX to GLB via Blender so gr.Model3D can render it."""
    glb_path = fbx_path.with_suffix(".glb")
    cmd = [
        _find_blender(),
        "--background",
        "--python", str(_FBX_TO_GLB_SCRIPT),
        "--",
        "--fbx", str(fbx_path),
        "--glb", str(glb_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"FBX→GLB conversion failed (exit {res.returncode})")
    return glb_path


def _list_motions():
    if not MOTIONS_DIR.is_dir():
        return []
    return sorted(p.stem for p in MOTIONS_DIR.glob("*.bvh"))


def run_pipeline_ui(image_path, motion, skip_bg, progress=gr.Progress()):
    if not image_path:
        return None, None, "❌ 이미지를 업로드해주세요."

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

        # Stage 2: Mesh
        progress(0.30, desc="3D 메시 생성 중 (TripoSR)...")
        t0 = time.time()
        mesh = generate_mesh(nobg, OUTPUT_BASE / "mesh", format="glb")
        add(f"✅ [2/4] 메시 생성: {time.time()-t0:.1f}s")

        # Stage 3: Rig
        progress(0.60, desc="자동 리깅 중 (Blender)...")
        t0 = time.time()
        rigged = auto_rig(mesh, OUTPUT_BASE / "rigged")
        add(f"✅ [3/4] 자동 리깅: {time.time()-t0:.1f}s")

        # Stage 4: Motion (optional)
        if motion and motion != "(none)":
            progress(0.85, desc=f"모션 적용 중 ({motion})...")
            t0 = time.time()
            anim = apply_motion(rigged, motion, OUTPUT_BASE / "animated", motion_name=motion)
            add(f"✅ [4/4] 모션 적용 ({motion}): {time.time()-t0:.1f}s")
            final = anim
            out_name = f"{stem}_{motion}.fbx"
        else:
            final = rigged
            out_name = f"{stem}.fbx"
            add("⏭ [4/4] 모션 적용: 건너뜀 (모션 미선택)")

        # Copy to UI output dir for download
        download_path = UI_OUTPUT / out_name
        shutil.copy(final, download_path)

        # Convert to GLB for in-browser preview (Model3D needs glTF/GLB).
        progress(0.95, desc="3D 미리보기 변환 중...")
        try:
            glb_preview = _convert_fbx_to_glb(download_path)
            add(f"✅ 미리보기 변환: {glb_preview.name}")
        except Exception as ce:
            glb_preview = None
            add(f"⚠ 미리보기 변환 실패: {ce}")

        progress(1.0, desc="완료")
        add(f"\n✓ 결과: {out_name}")
        return str(download_path), str(glb_preview) if glb_preview else None, "\n".join(log)

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
# img2mesh — 사진 한 장으로 애니메이션 FBX 생성

이미지를 업로드하고 모션을 선택하면 다음 4단계 파이프라인이 자동 실행됩니다:

**배경 제거 → 3D 메시(TripoSR) → 자동 리깅(Mixamo 22-bone) → 모션 적용**

- 권장 입력: 전신이 보이는 정면 T-pose 사진 (1024×1024 이하)
- 모션을 `(none)`으로 두면 리깅까지만 수행해 정적 FBX 출력
- "배경 제거 건너뛰기"는 이미 알파 채널이 있는 PNG에서 사용
"""


def build_ui():
    motion_choices = ["(none)"] + _list_motions()
    default_motion = "walk" if "walk" in motion_choices else motion_choices[0]

    with gr.Blocks(title="img2mesh") as demo:
        gr.Markdown(_DESCRIPTION)
        with gr.Row():
            with gr.Column(scale=1):
                inp_img = gr.Image(type="filepath", label="입력 이미지", height=320)
                inp_motion = gr.Dropdown(choices=motion_choices, value=default_motion,
                                         label="모션 선택")
                inp_skip = gr.Checkbox(label="배경 제거 건너뛰기 (RGBA 입력 시)", value=False)
                btn = gr.Button("실행", variant="primary", size="lg")
            with gr.Column(scale=1):
                out_preview = gr.Model3D(
                    label="3D 미리보기 (회전·줌·애니메이션 재생)",
                    height=360,
                    interactive=False,
                    clear_color=[0.15, 0.15, 0.18, 1.0],
                )
                out_file = gr.File(label="결과 FBX 다운로드", interactive=False)
                out_log = gr.Textbox(label="진행 로그", lines=8, interactive=False)

        btn.click(
            fn=run_pipeline_ui,
            inputs=[inp_img, inp_motion, inp_skip],
            outputs=[out_file, out_preview, out_log],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
