"""3D mesh generation from a single image.

Two backends are supported:

* ``triposr`` (default) — Stability AI TripoSR. Regression-style triplane NeRF
  + marching cubes. ~6 GB VRAM, produces a colored mesh, fast (~5–15 s on a
  modern GPU).
* ``triposg`` — VAST-AI TripoSG. Rectified-flow diffusion model. ~8 GB VRAM,
  geometry-only (no vertex colors), slower (~30–60 s with 50 inference steps)
  but cleaner topology.

Both backends load lazily; switching backend within a process re-loads the
new model into VRAM and frees the other (if requested).

Public API (unchanged for callers; new ``backend`` kwarg defaults to TripoSR):
    generate_mesh(image_path, output_dir, format="glb", backend="triposr", ...)

CLI:
    python generate_mesh.py --backend triposr --input nobg.png
    python generate_mesh.py --backend triposg --input nobg.png --num-inference-steps 30
"""
import argparse
import sys
import types
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

_PROJ_DIR = Path(__file__).resolve().parent
_TRIPOSR_DIR = _PROJ_DIR / "TripoSR"
_TRIPOSG_DIR = _PROJ_DIR / "TripoSG"
_TRIPOSG_SCRIPTS_DIR = _TRIPOSG_DIR / "scripts"
_HUNYUAN_DIR = _PROJ_DIR / "Hunyuan3D-2"
_MVADAPTER_DIR = _PROJ_DIR / "MV-Adapter"
_MVADAPTER_SCRIPTS_DIR = _MVADAPTER_DIR / "scripts"
_CHECKPOINTS_DIR = _PROJ_DIR / "checkpoints"

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VALID_BACKENDS = ("triposr", "triposg")
# Texture pipelines applied on top of geometry-only backends (triposg).
# - none:      no texture (raw geometry)
# - ortho:     orthographic projection of input image (front + mirrored back)
# - hunyuan:   Hunyuan3D-2 Paint multi-view diffusion (non-commercial license)
# - mvadapter: MV-Adapter SDXL multi-view + nvdiffrast back-projection
#              (Apache-2.0, official TripoSG demo pipeline, best quality)
VALID_TEXTURERS = ("none", "ortho", "hunyuan", "mvadapter")


# ============================================================
# TripoSR backend
# ============================================================
_TRIPOSR_MODEL = None

# 12GB VRAM safe defaults.
_TSR_DEFAULT_CHUNK_SIZE = 8192
_TSR_DEFAULT_MC_RESOLUTION = 256
_TSR_DEFAULT_FOREGROUND_RATIO = 0.85


def _ensure_triposr_on_path():
    if not _TRIPOSR_DIR.is_dir():
        raise FileNotFoundError(
            f"TripoSR not found at {_TRIPOSR_DIR}. Run `bash install_3d_model.sh` first."
        )
    if str(_TRIPOSR_DIR) not in sys.path:
        sys.path.insert(0, str(_TRIPOSR_DIR))


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


def _load_triposr_model():
    global _TRIPOSR_MODEL
    if _TRIPOSR_MODEL is not None:
        return _TRIPOSR_MODEL

    _ensure_triposr_on_path()
    _install_torchmcubes_shim()
    from tsr.system import TSR  # noqa: E402

    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(_TSR_DEFAULT_CHUNK_SIZE)
    model.to(_DEVICE)
    _TRIPOSR_MODEL = model
    return _TRIPOSR_MODEL


def _prepare_image_triposr(image_path: Path, foreground_ratio: float) -> Image.Image:
    _ensure_triposr_on_path()
    from tsr.utils import resize_foreground

    img = Image.open(image_path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    img = resize_foreground(img, foreground_ratio)
    arr = np.array(img).astype(np.float32) / 255.0
    rgb = arr[..., :3] * arr[..., 3:4] + (1 - arr[..., 3:4]) * 0.5
    return Image.fromarray((rgb * 255.0).astype(np.uint8))


def _generate_mesh_triposr(
    image_path: Path,
    output_dir: Path,
    fmt: str,
    mc_resolution: int,
    foreground_ratio: float,
    use_fp16: bool,
) -> str:
    model = _load_triposr_model()
    image = _prepare_image_triposr(image_path, foreground_ratio)

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
    print(f"[generate_mesh] [triposr] {image_path.name} -> {out_path}")

    if _DEVICE == "cuda":
        torch.cuda.empty_cache()

    return str(out_path)


# ============================================================
# TripoSG backend
# ============================================================
_TRIPOSG_PIPE = None
_TRIPOSG_RMBG = None

# Upstream defaults from VAST-AI/TripoSG/scripts/inference_triposg.py.
_TSG_DEFAULT_SEED = 42
_TSG_DEFAULT_NUM_STEPS = 50
_TSG_DEFAULT_GUIDANCE = 7.0
_TSG_DEFAULT_FACES = -1  # -1 means no simplification
# Iso-surface octree depth: upstream default is 9 (=512^3 grid) which pushes
# 12GB cards past their limit and falls into system-RAM paging during the
# flash decoder pass — iso-surface stalls for many minutes. 8 (=256^3 grid)
# keeps the geometry sharp enough for downstream rigging while completing in
# under a minute on 12GB. Bump back to 9 on >=24GB cards if needed.
_TSG_DEFAULT_OCTREE_DEPTH = 8
_TSG_TEXTURE_FALLBACK = np.array([180, 180, 180, 255], dtype=np.uint8)


def _build_texture_source(image_path: Path) -> np.ndarray:
    """Replicate TripoSG's prepare_image crop+pad on RGBA so the mesh's XY frame
    aligns with image pixels. Returns a square (S, S, 4) RGBA canvas."""
    img = Image.open(str(image_path)).convert("RGBA")
    arr = np.array(img)
    H, W = arr.shape[:2]
    alpha = arr[..., 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        x0, y0, x1, y1 = 0, 0, W, H
    else:
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    # TripoSG's load_image pads by ~10% of the longer side, then squares the bbox.
    pad = int(max(bw, bh) * 0.1)
    if bw > bh:
        extra = (bw - bh) // 2
        x0p, x1p = x0 - pad, x1 + pad
        y0p, y1p = y0 - pad - extra, y1 + pad + extra
    else:
        extra = (bh - bw) // 2
        x0p, x1p = x0 - pad - extra, x1 + pad + extra
        y0p, y1p = y0 - pad, y1 + pad
    side = max(x1p - x0p, y1p - y0p)
    canvas = np.zeros((side, side, 4), dtype=np.uint8)
    sx0, sy0 = max(0, x0p), max(0, y0p)
    sx1, sy1 = min(W, x1p), min(H, y1p)
    dx0, dy0 = sx0 - x0p, sy0 - y0p
    canvas[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = arr[sy0:sy1, sx0:sx1]
    return canvas


# ------------------------------------------------------------
# Hunyuan3D-2 Paint texture pipeline (high quality, multi-view diffusion)
# ------------------------------------------------------------
_HUNYUAN_PAINT = None
_HUNYUAN_REPO = "tencent/Hunyuan3D-2"
# Distilled variant: smaller VRAM (~8GB) and ~3x faster than the full v2-0.
_HUNYUAN_DEFAULT_SUBFOLDER = "hunyuan3d-paint-v2-0-turbo"


def _ensure_hunyuan_on_path():
    if not _HUNYUAN_DIR.is_dir():
        raise FileNotFoundError(
            f"Hunyuan3D-2 not found at {_HUNYUAN_DIR}. Run `bash install_3d_model.sh` first."
        )
    sp = str(_HUNYUAN_DIR)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def _log_vram(tag: str):
    """Print current/peak VRAM usage with explicit stdout flush so messages
    survive subprocess buffering."""
    if _DEVICE != "cuda":
        return
    cur = torch.cuda.memory_allocated() / 1024 ** 3
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    free, total = torch.cuda.mem_get_info()
    free_gb, total_gb = free / 1024 ** 3, total / 1024 ** 3
    print(f"[generate_mesh] VRAM {tag}: alloc={cur:.2f}GB peak={peak:.2f}GB "
          f"free={free_gb:.2f}/{total_gb:.2f}GB", flush=True)


def _free_triposg_from_gpu():
    """Move the cached TripoSG pipeline + RMBG net off the GPU and reclaim
    VRAM before loading Hunyuan. Without this, the delight model (SD2-depth)
    plus multiview diffusion silently OOM on a 12GB card."""
    global _TRIPOSG_PIPE, _TRIPOSG_RMBG
    if _DEVICE != "cuda":
        return
    if _TRIPOSG_PIPE is not None:
        try:
            _TRIPOSG_PIPE.to("cpu")
        except Exception as e:
            print(f"[generate_mesh] could not move TripoSG to CPU: {e}", flush=True)
    if _TRIPOSG_RMBG is not None:
        try:
            _TRIPOSG_RMBG.to("cpu")
        except Exception as e:
            print(f"[generate_mesh] could not move RMBG to CPU: {e}", flush=True)
    torch.cuda.empty_cache()


def _load_hunyuan_pipeline(subfolder: str = _HUNYUAN_DEFAULT_SUBFOLDER,
                           cpu_offload: bool = True):
    """Lazy-load Hunyuan3DPaintPipeline. CPU offload trades a small speed hit
    for staying under 12GB VRAM (typical for the turbo variant)."""
    global _HUNYUAN_PAINT
    if _HUNYUAN_PAINT is not None:
        return _HUNYUAN_PAINT

    _ensure_hunyuan_on_path()
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    print(f"[generate_mesh] Loading Hunyuan3D Paint ({subfolder}) ...", flush=True)
    _log_vram("before Hunyuan load")
    pipe = Hunyuan3DPaintPipeline.from_pretrained(_HUNYUAN_REPO, subfolder=subfolder)
    _log_vram("after Hunyuan load")
    if cpu_offload:
        try:
            pipe.enable_model_cpu_offload()
            print("[generate_mesh] Hunyuan CPU offload enabled.", flush=True)
        except Exception as e:
            print(f"[generate_mesh] CPU offload not available: {e}", flush=True)
    _HUNYUAN_PAINT = pipe
    return pipe


def _apply_texture_hunyuan(mesh, image_path: Path,
                           subfolder: str = _HUNYUAN_DEFAULT_SUBFOLDER):
    """Apply Hunyuan3D-2 Paint multi-view texture synthesis to the mesh.

    Pipeline (internal to Hunyuan3DPaintPipeline):
      1. Delight model removes shadows/lighting from the input photo.
      2. xatlas UV-unwraps the mesh.
      3. Multi-view diffusion synthesizes 6 view images (front/back/sides/up/down)
         conditioned on the de-lit input + per-view normal/position maps.
      4. Differentiable renderer back-projects views onto the UV atlas with
         per-view weight maps; an inpainter fills UV regions seen by no view.
      5. Returns a trimesh with UVs + PIL texture set on `mesh.visual.material`.

    Returns the textured trimesh (or the original mesh on failure)."""
    # Free ~10GB of VRAM held by the TripoSG pipeline before pulling in
    # delight (SD2-depth) + multiview diffusion, otherwise we silently
    # OOM-thrash on 12GB cards.
    _free_triposg_from_gpu()
    _log_vram("after freeing TripoSG")
    pipe = _load_hunyuan_pipeline(subfolder=subfolder)
    img = Image.open(str(image_path)).convert("RGBA")
    print(f"[generate_mesh] Running Hunyuan paint on mesh "
          f"(verts={len(mesh.vertices)}, faces={len(mesh.faces)}) ...", flush=True)
    try:
        result = pipe(mesh, image=img)
        _log_vram("after Hunyuan paint")
        print("[generate_mesh] Hunyuan paint done.", flush=True)
        return result
    except Exception as e:
        print(f"[generate_mesh] Hunyuan texture failed: {type(e).__name__}: {e}. "
              "Falling back to ortho projection.", flush=True)
        _apply_texture_from_image(mesh, image_path)
        return mesh


# ------------------------------------------------------------
# MV-Adapter texture pipeline (TripoSG demo pipeline, Apache-2.0)
# ------------------------------------------------------------
_MVADAPTER_PIPE = None
_MVADAPTER_BIREFNET = None
_MVADAPTER_TEX_PIPE = None
_MVADAPTER_TRANSFORM = None
_MVADAPTER_NUM_VIEWS = 6
# MV-Adapter SDXL is trained at 768² — running it at 512² takes it out of
# distribution and the 6 views come back as rainbow-saturated noise (std
# rises from ~30 to ~75, per-view means desync). Only the sd21 variant is
# trained at 512². Keep 768² for sdxl and pair it with attention_slicing
# to stay under 12 GB VRAM without paging — that alone is what made the
# original 122 s/step run slow, not the resolution.
_MVADAPTER_HEIGHT = 768
_MVADAPTER_UV_SIZE = 4096
# Six orthogonal cameras: 0°/90°/180°/270° around Y, plus top and bottom.
_MVADAPTER_AZIMUTHS = [x - 90 for x in [0, 90, 180, 270, 180, 180]]
_MVADAPTER_ELEVATIONS = [0, 0, 0, 0, 89.99, -89.99]


def _ensure_mvadapter_on_path():
    if not _MVADAPTER_DIR.is_dir():
        raise FileNotFoundError(
            f"MV-Adapter not found at {_MVADAPTER_DIR}. Run `bash install_3d_model.sh` first."
        )
    for p in (_MVADAPTER_DIR, _MVADAPTER_SCRIPTS_DIR):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def _download_mvadapter_aux_checkpoints():
    """RealESRGAN upscaler + LaMa inpainter are required by TexturePipeline.
    They are not on the HF model hub directly — pulled from their own
    release URLs and cached under ./checkpoints/."""
    _CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    upscaler = _CHECKPOINTS_DIR / "RealESRGAN_x2plus.pth"
    inpainter = _CHECKPOINTS_DIR / "big-lama.pt"
    if not upscaler.is_file():
        print("[generate_mesh] Downloading RealESRGAN x2plus checkpoint ...", flush=True)
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("dtarnow/UPscaler", filename="RealESRGAN_x2plus.pth",
                               local_dir=str(_CHECKPOINTS_DIR))
        if not upscaler.is_file():
            # hf may save with different name; symlink if needed
            try:
                upscaler.symlink_to(path)
            except Exception:
                import shutil
                shutil.copy(path, upscaler)
    if not inpainter.is_file():
        print("[generate_mesh] Downloading LaMa big-lama checkpoint ...", flush=True)
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
            str(inpainter),
        )
    return str(upscaler), str(inpainter)


def _load_mvadapter_pipeline(cpu_offload: bool = False):
    """Lazy-load MV-Adapter SDXL multi-view pipeline + BiRefNet bg remover +
    TexturePipeline.

    cpu_offload=True saves VRAM but breaks MV-Adapter's reference-image
    attention cache (`cross_attention_kwargs["cache_hidden_states"]` is not
    reliably forwarded through accelerate's pre/post-forward hooks, leaving
    `ref_hidden_states` empty and triggering KeyError during generation).
    Default to keeping the SDXL stack on GPU."""
    global _MVADAPTER_PIPE, _MVADAPTER_BIREFNET, _MVADAPTER_TEX_PIPE, _MVADAPTER_TRANSFORM
    if _MVADAPTER_PIPE is not None:
        return _MVADAPTER_PIPE, _MVADAPTER_BIREFNET, _MVADAPTER_TEX_PIPE, _MVADAPTER_TRANSFORM

    _ensure_mvadapter_on_path()
    from inference_ig2mv_sdxl import prepare_pipeline
    from mvadapter.pipelines.pipeline_texture import TexturePipeline
    from transformers import AutoModelForImageSegmentation
    from torchvision import transforms as _tvt

    print("[generate_mesh] Loading MV-Adapter SDXL pipeline (LCM-fast mode) ...",
          flush=True)
    _log_vram("before MV-Adapter load")
    # LCM-LoRA distills SDXL down to ~4-step inference. MV-Adapter's
    # `prepare_pipeline` natively supports `scheduler="lcm"` (wraps
    # LCMScheduler with ShiftSNR), and accepts `lora_model="<repo>/<file>"`
    # to fuse a LoRA on top of the custom MV attention processors. At 768²
    # the 6-view diffusion drops from ~30 min @ 15 steps to ~5 min @ 4 steps.
    pipe = prepare_pipeline(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        vae_model="madebyollin/sdxl-vae-fp16-fix",
        unet_model=None,
        lora_model="latent-consistency/lcm-lora-sdxl/pytorch_lora_weights.safetensors",
        adapter_path="huanngzh/mv-adapter",
        scheduler="lcm",
        num_views=_MVADAPTER_NUM_VIEWS,
        device=_DEVICE,
        dtype=torch.float16,
    )
    if cpu_offload:
        try:
            pipe.enable_model_cpu_offload()
            print("[generate_mesh] MV-Adapter CPU offload enabled.", flush=True)
        except Exception as e:
            print(f"[generate_mesh] MV-Adapter CPU offload unavailable: {e}", flush=True)
    else:
        # accelerate hooks drop cross_attention_kwargs["cache_hidden_states"]
        # before reaching MV custom processors; keep the whole pipeline on
        # GPU so the reference-image attention cache stays populated.
        pipe.to(_DEVICE)
        print("[generate_mesh] MV-Adapter kept on GPU (no CPU offload).", flush=True)

    # Earlier we enabled attention_slicing("auto") to trim VRAM, but it bypasses
    # MV-Adapter's custom DecoupledMVRowColSelfAttnProcessor2_0 cross-attention
    # to the reference image — the slicer rewrites self-attention's internal
    # math and breaks the chunked reference-cache lookup, so SDXL has nothing
    # to anchor colors to and falls back to gray "high-quality" outputs.
    # PyTorch 2.x SDPA + 768² already fits in 12 GB without slicing, so leave
    # it off to keep reference conditioning intact.
    print("[generate_mesh] MV-Adapter attention slicing disabled to preserve "
          "reference cross-attention.", flush=True)

    print("[generate_mesh] Loading BiRefNet bg-removal ...", flush=True)
    birefnet = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    ).to(_DEVICE)
    birefnet.eval()
    transform_image = _tvt.Compose([
        _tvt.Resize((1024, 1024)),
        _tvt.ToTensor(),
        _tvt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    upscaler_ckpt, inpaint_ckpt = _download_mvadapter_aux_checkpoints()
    tex_pipe = TexturePipeline(
        upscaler_ckpt_path=upscaler_ckpt,
        inpaint_ckpt_path=inpaint_ckpt,
        device=_DEVICE,
    )

    _log_vram("after MV-Adapter load")
    _MVADAPTER_PIPE = pipe
    _MVADAPTER_BIREFNET = birefnet
    _MVADAPTER_TEX_PIPE = tex_pipe
    _MVADAPTER_TRANSFORM = transform_image
    return pipe, birefnet, tex_pipe, transform_image


def _apply_texture_mvadapter(mesh, image_path: Path):
    """Apply MV-Adapter multi-view texture synthesis to the mesh.

    Pipeline (same as the official TripoSG HuggingFace Space):
      1. Save mesh to a temp GLB and load it through nvdiffrast.
      2. Render six orthogonal views as position + normal control maps.
      3. MV-Adapter SDXL generates six RGB views conditioned on the reference
         photo and the position/normal maps.
      4. TexturePipeline back-projects views onto the UV atlas (xatlas via
         open3d), upscales with RealESRGAN, and inpaints with LaMa.
      5. Loads the resulting GLB back through trimesh and returns it.
    """
    if _DEVICE != "cuda":
        raise RuntimeError("MV-Adapter requires CUDA. Use --texturer ortho on CPU-only systems.")

    _free_triposg_from_gpu()
    _log_vram("after freeing TripoSG")
    pipe, birefnet, tex_pipe, transform_image = _load_mvadapter_pipeline()

    _ensure_mvadapter_on_path()
    from inference_ig2mv_sdxl import preprocess_image, remove_bg
    from mvadapter.utils.mesh_utils import (
        NVDiffRastContextWrapper, get_orthogonal_camera, load_mesh, render,
    )
    from mvadapter.utils import make_image_grid

    import tempfile
    work = Path(tempfile.mkdtemp(prefix="mvadapter_"))
    src_glb = work / "input.glb"
    # Persist the trimesh to disk so MV-Adapter's nvdiffrast loader can read
    # it (it expects a file path, not an in-memory Trimesh).
    mesh.export(str(src_glb))

    print("[generate_mesh] MV-Adapter: rendering 6-view control maps ...", flush=True)
    cameras = get_orthogonal_camera(
        elevation_deg=_MVADAPTER_ELEVATIONS,
        distance=[1.8] * _MVADAPTER_NUM_VIEWS,
        left=-0.55, right=0.55, bottom=-0.55, top=0.55,
        azimuth_deg=_MVADAPTER_AZIMUTHS,
        device=_DEVICE,
    )
    ctx = NVDiffRastContextWrapper(device=_DEVICE, context_type="cuda")
    nv_mesh = load_mesh(str(src_glb), rescale=True, device=_DEVICE)
    render_out = render(
        ctx, nv_mesh, cameras,
        height=_MVADAPTER_HEIGHT, width=_MVADAPTER_HEIGHT,
        render_attr=False, normal_background=0.0,
    )
    control_images = (
        torch.cat([
            (render_out.pos + 0.5).clamp(0, 1),
            (render_out.normal / 2 + 0.5).clamp(0, 1),
        ], dim=-1)
        .permute(0, 3, 1, 2)
        .to(_DEVICE)
    )

    print("[generate_mesh] MV-Adapter: removing bg + 6-view diffusion ...", flush=True)
    ref_img = Image.open(str(image_path))
    # MV-Adapter's preprocess_image alpha-blends with a 0.5 gray background:
    #   image = image[:, :, :3] * alpha + (1 - alpha) * 0.5
    # If we let BiRefNet recompute the mask from a flattened RGB image, the
    # soft edges of the new mask plus the black-flattened RGB premultiply
    # produces a globally desaturated reference (colorful subjects come back
    # as washed-out gray). When the input is already bg-removed RGBA, keep
    # its alpha verbatim and skip BiRefNet entirely — colors stay saturated.
    if ref_img.mode == "RGBA":
        print("[generate_mesh] MV-Adapter: reusing input RGBA alpha "
              "(skipping BiRefNet re-segmentation).", flush=True)
    else:
        # Raw RGB photo without alpha: BiRefNet runs to produce a mask.
        ref_img = ref_img.convert("RGB")
        ref_img = remove_bg(ref_img, birefnet, transform_image, _DEVICE)
    ref_img = preprocess_image(ref_img, _MVADAPTER_HEIGHT, _MVADAPTER_HEIGHT)

    # LCM mode: 4 steps with low CFG. LCM-distilled SDXL no longer needs
    # classifier-free guidance (guidance_scale=1.0 disables it); higher CFG
    # over-amplifies the distilled output and brings back noise. Step count
    # matches the LCM-LoRA's distillation target (best quality at 4-8 steps).
    images = pipe(
        "high quality",
        height=_MVADAPTER_HEIGHT, width=_MVADAPTER_HEIGHT,
        num_inference_steps=4,
        guidance_scale=1.0,
        num_images_per_prompt=_MVADAPTER_NUM_VIEWS,
        control_image=control_images,
        control_conditioning_scale=1.0,
        reference_image=ref_img,
        reference_conditioning_scale=1.0,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        cross_attention_kwargs={"scale": 1.0},
    ).images

    if _DEVICE == "cuda":
        torch.cuda.empty_cache()

    mv_path = work / "mv.png"
    make_image_grid(images, rows=1).save(mv_path)

    from mvadapter.pipelines.pipeline_texture import ModProcessConfig
    print("[generate_mesh] MV-Adapter: back-projecting views to UV ...", flush=True)
    out = tex_pipe(
        mesh_path=str(src_glb),
        save_dir=str(work),
        save_name="textured",
        uv_unwarp=True,
        uv_size=_MVADAPTER_UV_SIZE,
        rgb_path=str(mv_path),
        rgb_process_config=ModProcessConfig(view_upscale=True, inpaint_mode="view"),
        camera_azimuth_deg=_MVADAPTER_AZIMUTHS,
    )

    result_path = getattr(out, "shaded_model_save_path", None) or \
                  getattr(out, "pbr_model_save_path", None)
    if not result_path or not Path(result_path).is_file():
        raise RuntimeError(f"MV-Adapter TexturePipeline produced no output (got {out!r})")

    print(f"[generate_mesh] MV-Adapter: loading textured GLB {result_path}", flush=True)
    import trimesh
    result = trimesh.load(result_path, force="mesh")
    _log_vram("after MV-Adapter paint")
    return result


def _apply_texture_from_image(mesh, image_path: Path):
    """Project the bg-removed input image onto the mesh as vertex colors.

    Front-facing vertices (Z >= 0 in TripoSG frame, which is +Z toward camera)
    get a direct orthographic projection. Back-facing vertices get a mirrored-X
    projection (humanoid bilateral-symmetry assumption). Transparent samples
    fall back to a neutral gray. Call BEFORE the orientation rotation so XY
    alignment with image pixels holds.
    """
    canvas = _build_texture_source(image_path)
    S = canvas.shape[0]
    if S == 0:
        return mesh

    verts = mesh.vertices.astype(np.float32)
    xy = verts[:, :2]
    xy_min, xy_max = xy.min(axis=0), xy.max(axis=0)
    extent = float((xy_max - xy_min).max())
    if extent < 1e-6:
        return mesh
    center = (xy_min + xy_max) / 2.0

    def sample(vx, vy):
        u = (vx - center[0]) / extent + 0.5
        v = (vy - center[1]) / extent + 0.5
        ix = np.clip((u * (S - 1)).astype(np.int32), 0, S - 1)
        iy = np.clip(((1.0 - v) * (S - 1)).astype(np.int32), 0, S - 1)
        return canvas[iy, ix]

    front = sample(verts[:, 0], verts[:, 1])
    back = sample(-verts[:, 0], verts[:, 1])  # mirror X for occluded side
    front_mask = (verts[:, 2] >= 0)[:, None]
    colors = np.where(front_mask, front, back).astype(np.uint8)

    # Fallback for transparent / out-of-foreground samples on both sides.
    transparent = colors[:, 3] < 128
    colors[transparent] = _TSG_TEXTURE_FALLBACK
    colors[:, 3] = 255

    mesh.visual.vertex_colors = colors
    return mesh


def _ensure_triposg_on_path():
    if not _TRIPOSG_DIR.is_dir():
        raise FileNotFoundError(
            f"TripoSG not found at {_TRIPOSG_DIR}. Run `bash install_3d_model.sh` first."
        )
    # The `triposg` package lives at the repo root; helper modules
    # (image_process.py, briarmbg.py) live in `scripts/`.
    for p in (_TRIPOSG_DIR, _TRIPOSG_SCRIPTS_DIR):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def _load_triposg_models(dtype=torch.float16):
    global _TRIPOSG_PIPE, _TRIPOSG_RMBG
    if _TRIPOSG_PIPE is not None and _TRIPOSG_RMBG is not None:
        return _TRIPOSG_PIPE, _TRIPOSG_RMBG

    _ensure_triposg_on_path()
    from huggingface_hub import snapshot_download
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline
    from briarmbg import BriaRMBG

    weights_root = _TRIPOSG_DIR / "pretrained_weights"
    triposg_weights = weights_root / "TripoSG"
    rmbg_weights = weights_root / "RMBG-1.4"

    if not triposg_weights.is_dir() or not any(triposg_weights.iterdir()):
        print("[generate_mesh] Downloading TripoSG weights (VAST-AI/TripoSG) ...")
        snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=str(triposg_weights))
    if not rmbg_weights.is_dir() or not any(rmbg_weights.iterdir()):
        print("[generate_mesh] Downloading RMBG-1.4 weights (briaai/RMBG-1.4) ...")
        snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=str(rmbg_weights))

    rmbg_net = BriaRMBG.from_pretrained(str(rmbg_weights)).to(_DEVICE)
    rmbg_net.eval()

    pipe = TripoSGPipeline.from_pretrained(str(triposg_weights)).to(_DEVICE, dtype)

    _TRIPOSG_PIPE = pipe
    _TRIPOSG_RMBG = rmbg_net
    return _TRIPOSG_PIPE, _TRIPOSG_RMBG


def _generate_mesh_triposg(
    image_path: Path,
    output_dir: Path,
    fmt: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    faces: int,
    use_fp16: bool,
    texturer: str = "mvadapter",
    hunyuan_subfolder: str = _HUNYUAN_DEFAULT_SUBFOLDER,
    octree_depth: int = _TSG_DEFAULT_OCTREE_DEPTH,
) -> str:
    if _DEVICE != "cuda":
        # TripoSG's image_process.py hard-codes `.cuda()`. CPU is not supported.
        raise RuntimeError(
            "TripoSG requires a CUDA-enabled GPU. Use --backend triposr on CPU-only systems."
        )

    dtype = torch.float16 if use_fp16 else torch.float32
    pipe, rmbg_net = _load_triposg_models(dtype=dtype)

    _ensure_triposg_on_path()
    from image_process import prepare_image as _tsg_prepare_image

    # TripoSG expects a file path; we pass the already-bg-removed PNG directly.
    # `prepare_image` re-runs RMBG if the input lacks valid alpha.
    img_pil = _tsg_prepare_image(
        str(image_path),
        bg_color=np.array([1.0, 1.0, 1.0]),
        rmbg_net=rmbg_net,
    )

    with torch.no_grad():
        outputs = pipe(
            image=img_pil,
            generator=torch.Generator(device=pipe.device).manual_seed(seed),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            flash_octree_depth=octree_depth,
            dense_octree_depth=min(octree_depth, 8),
            hierarchical_octree_depth=octree_depth,
        ).samples[0]

    import trimesh
    mesh = trimesh.Trimesh(
        outputs[0].astype(np.float32),
        np.ascontiguousarray(outputs[1]),
    )
    print(f"[generate_mesh] [triposg] iso-surface done: "
          f"verts={len(mesh.vertices)} faces={len(mesh.faces)}", flush=True)
    _log_vram("after TripoSG iso-surface")

    if faces and faces > 0:
        mesh = _simplify_mesh(mesh, faces)

    # TripoSG is geometry-only — no vertex colors or UVs come from the model.
    # Texture is applied in TripoSG's frame (camera looking down -Z, character
    # facing +Z toward viewer) so image-space sampling and the multi-view
    # rendering used by Hunyuan both line up with the input photo.
    texturer = (texturer or "none").lower()
    if texturer == "mvadapter":
        mesh = _apply_texture_mvadapter(mesh, image_path)
    elif texturer == "hunyuan":
        mesh = _apply_texture_hunyuan(mesh, image_path, subfolder=hunyuan_subfolder)
    elif texturer == "ortho":
        _apply_texture_from_image(mesh, image_path)
    elif texturer == "none":
        pass
    else:
        raise ValueError(f"Unknown texturer '{texturer}'. Choose from {VALID_TEXTURERS}.")

    # TripoSG outputs Y-up with the character facing +Z. Blender's glTF
    # importer maps glTF +Z → Blender -Y, which is exactly the direction the
    # default Blender camera looks from, so the character already faces the
    # camera after import. An earlier 180° rotation flipped face to -Z (i.e.
    # Blender +Y, away from camera) — that's why animated FBX showed the
    # character walking away from the camera. The skeleton template's foot
    # toes at Z=+0.15 also point toward Blender +Y, so leaving face=+Z makes
    # face direction and walking direction agree.

    out_path = output_dir / f"{image_path.stem}.{fmt}"
    mesh.export(str(out_path))
    print(f"[generate_mesh] [triposg] {image_path.name} -> {out_path}")

    if _DEVICE == "cuda":
        torch.cuda.empty_cache()

    return str(out_path)


def _simplify_mesh(mesh, n_faces):
    """Quadric edge-collapse decimation via pymeshlab — mirrors TripoSG's helper."""
    if mesh.faces.shape[0] <= n_faces:
        return mesh
    import pymeshlab
    import trimesh
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
    ms.meshing_merge_close_vertices()
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=n_faces)
    current = ms.current_mesh()
    return trimesh.Trimesh(vertices=current.vertex_matrix(), faces=current.face_matrix())


# ============================================================
# Public dispatcher
# ============================================================
def generate_mesh(
    image_path: Union[str, Path],
    output_dir: Union[str, Path] = "./output/mesh",
    format: str = "glb",
    backend: str = "triposr",
    # TripoSR knobs (ignored by TripoSG)
    mc_resolution: int = _TSR_DEFAULT_MC_RESOLUTION,
    foreground_ratio: float = _TSR_DEFAULT_FOREGROUND_RATIO,
    # TripoSG knobs (ignored by TripoSR)
    seed: int = _TSG_DEFAULT_SEED,
    num_inference_steps: int = _TSG_DEFAULT_NUM_STEPS,
    guidance_scale: float = _TSG_DEFAULT_GUIDANCE,
    faces: int = _TSG_DEFAULT_FACES,
    octree_depth: int = _TSG_DEFAULT_OCTREE_DEPTH,
    texturer: str = "mvadapter",
    hunyuan_subfolder: str = _HUNYUAN_DEFAULT_SUBFOLDER,
    # Shared
    use_fp16: bool = False,
) -> str:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = format.lower().lstrip(".")
    if fmt not in {"glb", "obj"}:
        raise ValueError(f"Unsupported format: {format}")

    backend = (backend or "triposr").lower()
    if backend not in VALID_BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose from {VALID_BACKENDS}.")

    if backend == "triposr":
        # TripoSR already produces vertex colors; texturer is not applied.
        return _generate_mesh_triposr(
            image_path, output_dir, fmt,
            mc_resolution=mc_resolution,
            foreground_ratio=foreground_ratio,
            use_fp16=use_fp16,
        )
    return _generate_mesh_triposg(
        image_path, output_dir, fmt,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        faces=faces,
        use_fp16=use_fp16,
        texturer=texturer,
        hunyuan_subfolder=hunyuan_subfolder,
        octree_depth=octree_depth,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate 3D mesh from a single image (TripoSR or TripoSG).")
    parser.add_argument("--input", required=True, help="Input RGBA image (background-removed).")
    parser.add_argument("--output", default="./output/mesh", help="Output directory.")
    parser.add_argument("--format", default="glb", choices=["glb", "obj"], help="Mesh format.")
    parser.add_argument("--backend", default="triposr", choices=list(VALID_BACKENDS),
                        help="Mesh generation backend (default: triposr).")
    # TripoSR
    parser.add_argument("--mc-resolution", type=int, default=_TSR_DEFAULT_MC_RESOLUTION,
                        help="[triposr] Marching cubes resolution (lower=less VRAM).")
    parser.add_argument("--foreground-ratio", type=float, default=_TSR_DEFAULT_FOREGROUND_RATIO,
                        help="[triposr] Foreground crop ratio.")
    # TripoSG
    parser.add_argument("--seed", type=int, default=_TSG_DEFAULT_SEED,
                        help="[triposg] Random seed for the diffusion sampler.")
    parser.add_argument("--num-inference-steps", type=int, default=_TSG_DEFAULT_NUM_STEPS,
                        help="[triposg] Number of denoising steps (default 50).")
    parser.add_argument("--guidance-scale", type=float, default=_TSG_DEFAULT_GUIDANCE,
                        help="[triposg] Classifier-free guidance scale (default 7.0).")
    parser.add_argument("--faces", type=int, default=_TSG_DEFAULT_FACES,
                        help="[triposg] Target face count for decimation; -1 = no simplification.")
    parser.add_argument("--octree-depth", type=int, default=_TSG_DEFAULT_OCTREE_DEPTH,
                        help="[triposg] Iso-surface octree depth (8=256³ grid≈safe on 12GB, 9=512³ needs 24GB).")
    parser.add_argument("--texturer", default="mvadapter", choices=list(VALID_TEXTURERS),
                        help="[triposg] Texture pipeline: none | ortho | hunyuan | mvadapter (default).")
    parser.add_argument("--hunyuan-subfolder", default=_HUNYUAN_DEFAULT_SUBFOLDER,
                        help="[triposg+hunyuan] HF subfolder for Hunyuan paint weights.")
    # Shared
    parser.add_argument("--fp16", action="store_true",
                        help="Enable fp16 (TripoSG: model dtype; TripoSR: autocast).")
    args = parser.parse_args()

    generate_mesh(
        args.input,
        args.output,
        args.format,
        backend=args.backend,
        mc_resolution=args.mc_resolution,
        foreground_ratio=args.foreground_ratio,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        faces=args.faces,
        octree_depth=args.octree_depth,
        texturer=args.texturer,
        hunyuan_subfolder=args.hunyuan_subfolder,
        use_fp16=args.fp16,
    )


if __name__ == "__main__":
    main()
