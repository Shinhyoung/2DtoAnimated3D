# img2mesh_pipeline

2D 이미지 한 장에서 시작해 **배경 제거 → 3D 메시 → 자동 리깅 → 모션 적용**까지 한 번에 수행하는 파이프라인.

```
input.png → output/result_walk.fbx (Mixamo-rigged + 걷기 애니메이션)
```

## 설치

### 1. Conda 환경
```bash
bash setup_env.sh          # img2mesh 환경 생성
conda activate img2mesh
```

### 2. 3D 모델 (TripoSR + TripoSG + MV-Adapter)
```bash
bash install_3d_model.sh   # TripoSR + TripoSG + MV-Adapter clone + 의존성 설치
```
스크립트는 세 저장소를 모두 클론하고 필요한 패키지를 설치합니다.
실제 가중치는 첫 실행 시 huggingface_hub로 자동 다운로드됩니다.

### 3. Blender (자동 리깅/모션 적용용)
- https://www.blender.org/download/ 에서 4.5 LTS 설치
- `BLENDER_PATH` 환경변수 등록 또는 PATH에 등록

### 4. 모션 데이터
`motions/` 폴더에 BVH 파일을 둡니다. 현재 동봉된 모션:
- `walk.bvh`, `breakdance.bvh`, `sillydancing.bvh`, `flair.bvh` — 정규화된 BVH (이전 프로젝트 호환 규약)
- `idle.bvh`, `Thriller.bvh`, `zombie_walk.bvh`, `test.bvh` — Mixamo 표준 BVH (정규화 권장 — 아래 "Mixamo BVH 정규화" 섹션 참조)

## 사용

### 웹 UI (가장 쉬움)
```bash
python app.py
```
브라우저가 자동으로 http://127.0.0.1:7860 에 열립니다.

두 가지 탭:
- **이미지에서 시작** — 사진 → 배경 제거 → 3D 메시(TripoSR/TripoSG) → 자동 리깅 → 모션 적용
- **3D 모델에서 시작** — 이미 가진 `.glb`/`.gltf`/`.obj` → 자동 리깅 → 모션 적용

### 3D 메시 백엔드 + 텍스처러 선택

| 백엔드 | 속도 | VRAM | 컬러 | 토폴로지 |
|---|---|---|---|---|
| `triposr` (기본) | ~10s | ~6GB | 정점 컬러 | 거침 (NeRF + marching cubes) |
| `triposg` | ~30–60s | ~8GB | 없음 (지오메트리만) | 깔끔 (rectified-flow diffusion) |

TripoSG 선택 시 텍스처러를 함께 선택:

| 텍스처러 | 속도 | 라이선스 | 비고 |
|---|---|---|---|
| `none` | 0s | — | 텍스처 없음 |
| `ortho` | 빠름 | — | 직교 투영(간단) |
| `hunyuan` | 보통 | 비상업 | Hunyuan3D-2 Paint |
| `mvadapter` (기본) | ~3분 (LCM-LoRA) | Apache-2.0 | MV-Adapter SDXL (TripoSG 공식 권장) |

### CLI: 전체 파이프라인 (모션 포함)
```bash
python main.py --input photo.jpg --output result.fbx --motion walk
```

### CLI: 백엔드/텍스처러 명시
```bash
python main.py --input photo.jpg --output result.fbx --motion walk \
    --backend triposg --texturer mvadapter
```

### 기타 CLI 옵션
```bash
python main.py --input photo.jpg --output result.fbx              # 리깅까지 (모션 생략)
python main.py --input photo.jpg --output result.glb --mesh-only  # 메시까지만
python main.py --input nobg.png --output result.fbx --skip-bg --motion run
python main.py --input ./photos/ --output ./output/ --motion idle # 폴더 일괄
python main.py --list-motions                                     # 모션 목록
python main.py --input photo.jpg --output result.fbx --motion walk --resume
```

## Mixamo BVH 정규화 (`normalize_bvh.py`)

### 왜 필요한가
Mixamo에서 받은 FBX를 Blender로 BVH 변환하면 본 이름에 `mixamorig:` prefix가 붙고, 캐릭터 키가 cm 단위(~160)로 들어있습니다. 우리 파이프라인의 `retarget_config.json`은 prefix 없는 본 이름과 meter-ish 스케일(`walk.bvh` 등은 ~0.5 unit 키)을 기준으로 동작합니다. prefix는 retarget 코드에서 자동 fallback 처리되지만, 스케일/본 roll 차이로 모션이 부자연스럽게 보일 수 있습니다.

`normalize_bvh.py`는 다음을 자동 처리:
1. `mixamorig:` prefix 제거
2. 지정한 target 높이로 armature 스케일 + Apply Scale (본 roll/오프셋 재계산)
3. 동일 FPS·프레임 범위로 BVH 재출력

### 사용 (PowerShell)

먼저 변수 설정 (한 번):
```powershell
$BLENDER = "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
$PIPE = "c:\0.shinhyoung\Project\autoRigging(TripoSG)\img2mesh_pipeline"
```

**단일 파일:**
```powershell
& $BLENDER --background --python "$PIPE\normalize_bvh.py" -- `
    --input  "$PIPE\motions\test.bvh" `
    --output "$PIPE\motions\test_normalized.bvh" `
    --height 0.5
```

**일괄 변환 (별도 폴더):**
```powershell
& $BLENDER --background --python "$PIPE\normalize_bvh.py" -- `
    --input-dir  "$PIPE\motions" `
    --output-dir "$PIPE\motions_normalized" `
    --height 0.5
```

**옵션:**
- `--height FLOAT` — target rest 높이 (기본 0.5, 기존 정규화 BVH 스케일과 일치)
- `--overwrite` — 일괄 모드에서 기존 파일 덮어쓰기 (기본은 skip)

### 워크플로우
1. Mixamo에서 모션을 FBX로 다운로드 (Settings: `Without Skin`, FPS 30)
2. Blender에서 FBX → BVH 일반 export (`motions_raw/`에 저장)
3. `normalize_bvh.py` 실행 → `motions/`에 정규화된 결과
4. Gradio 재시작 → dropdown에 새 모션 표시, 정상 모션 적용

## 출력 콘솔 예시
```
[log] output/logs/pipeline_20260527_182301.log
[1/4] 배경 제거 ...............  완료 (3.2s, VRAM: 1.1GB)
[2/4] 메시 생성 (triposg + mvadapter) ... 완료 (180.5s, VRAM: 11.2GB)
[3/4] 자동 리깅 ...............  완료 (12.1s, VRAM: 0.0GB)
[4/4] 모션 적용 (walk) ........  완료 (8.3s, VRAM: 0.0GB)

✓ 최종 출력: result.fbx
```

## 디렉토리 구조

```
img2mesh_pipeline/
├── main.py                    ← 통합 CLI 진입점
├── app.py                     ← Gradio 웹 UI
├── remove_bg.py               ← 배경 제거 (rembg/u2net)
├── generate_mesh.py           ← 3D 메시 (TripoSR / TripoSG + 텍스처)
├── auto_rig.py                ← 자동 리깅 orchestrator
├── apply_motion.py            ← 모션 적용 orchestrator
├── rig_in_blender.py          ← Blender headless 리깅
├── retarget_in_blender.py     ← Blender headless 리타게팅 (mixamorig: prefix 자동 fallback)
├── fbx_to_glb.py              ← FBX → GLB 변환 (Gradio 미리보기용)
├── normalize_bvh.py           ← Mixamo BVH 정규화 유틸리티
├── skeleton_config.json       ← Mixamo 22-bone 템플릿
├── retarget_config.json       ← BVH → FBX 본 매핑
├── motions/                   ← BVH 파일
└── output/
    ├── nobg/        ← 배경 제거 결과
    ├── mesh/        ← 3D 메시
    ├── rigged/      ← 리깅된 FBX
    ├── animated/    ← 애니메이션 FBX
    ├── ui/          ← Gradio 다운로드용 사본
    └── logs/        ← 실행 로그
```

## CLI 인자

| 인자 | 설명 |
|---|---|
| `--input PATH` | 입력 이미지(파일 또는 폴더) |
| `--output PATH` | 최종 출력(파일 또는 폴더) |
| `--motion NAME\|PATH` | 모션 이름 또는 `.bvh` 파일 경로 |
| `--list-motions` | 사용 가능한 모션 목록 출력 후 종료 |
| `--skip-bg` | 배경 제거 단계 건너뜀 (이미 RGBA 입력) |
| `--mesh-only` | 메시 생성까지만 |
| `--rig-only` | 리깅까지만 (모션 단계 건너뜀) |
| `--format glb\|obj` | 메시 포맷 (기본 `glb`) |
| `--backend triposr\|triposg` | 3D 메시 백엔드 (기본 `triposr`) |
| `--texturer none\|ortho\|hunyuan\|mvadapter` | 텍스처러 (`triposg` 전용, 기본 `mvadapter`) |
| `--keep-intermediate` / `--no-keep-intermediate` | 중간 결과 보존 (기본 보존) |
| `--resume` | 이전 실행 중단 지점부터 재시작 |

## 알려진 제한사항

- **VRAM**: RTX 5070(12GB) 기준 입력 이미지 1024×1024 권장. MV-Adapter는 ~11GB 사용
- **입력 사진**: 전신 정면, T-pose에 가까운 자세에서 최상의 결과. 어깨 패드/근육 굴곡이 강한 의상(슈퍼히어로 슈트 등)은 자동 리깅 한계로 어깨 변형이 어색할 수 있음
- **비휴머노이드**: 동물·기물 등은 Mixamo 22-bone 템플릿이 강제 fitting되므로 weight painting이 어색할 수 있음
- **BVH 호환성**: 우리 retarget 파이프라인은 본 이름 매칭 + COPY_ROTATION LOCAL→LOCAL 방식이라 source/target의 본 roll·rest pose가 일치해야 시각적으로 자연스럽습니다. Mixamo 표준 BVH를 사용할 때는 `normalize_bvh.py`로 전처리 권장
- **루트 이동**: retargeting은 회전만 복사 (root translation 미적용) — 걷기/달리기는 "제자리" 동작으로 표시. jump의 vertical lift도 미반영
- **모션 dropdown**: Gradio 시작 시점에 dropdown이 빌드됩니다. 새 BVH를 `motions/`에 추가한 후 적용하려면 Gradio 재시작 필요

## 기술 스택

- **Background removal**: rembg (u2net) + ONNX Runtime GPU
- **3D mesh**:
  - TripoSR (Stability AI) + PyMCubes (CPU marching cubes) — 기본
  - TripoSG (VAST-AI) — rectified-flow diffusion + DISO iso-surface
- **Texture**:
  - MV-Adapter SDXL (huanngzh, Apache-2.0) + LCM-LoRA 4-step (~3분, 기본)
  - Hunyuan3D-2 Paint (Tencent, 비상업)
  - 직교 투영(ortho) 폴백
- **Auto-rig**: Blender 4.5 headless + Mixamo 22-bone template fitting + bone-heat 자동 가중치
- **Motion**: Blender headless + Copy Rotation LOCAL constraints + bake + `mixamorig:` prefix 자동 fallback
- **GPU**: PyTorch 2.11+cu128 (RTX 5070 sm_120 호환)

## 변경 이력 (최근)

- **2026-05-27**
  - Rigging/motion 파이프라인을 이전 프로젝트(autoRigging) 단순 버전으로 롤백 — 어깨 weight bias, 다리 격리, 메시 separation 등 customization 모두 제거. 기본 bone-heat 자동 가중치 + LOCAL→LOCAL copy_rotation
  - `retarget_in_blender.py`에 `mixamorig:` prefix 자동 fallback 추가 — Mixamo 표준 BVH도 본 매칭 통과
  - `normalize_bvh.py` 추가 — Mixamo BVH 정규화 유틸리티
  - 직전 시도들의 백업 파일이 `*.bak.20260527`로 보존됨
- **2026-05-25 이전**
  - TripoSG 백엔드 통합
  - MV-Adapter 텍스처 파이프라인 통합 (LCM-LoRA 가속)
  - Hunyuan3D-2 Paint 텍스처 옵션 추가
  - 어깨/팔 anatomical detection, UV-seam weight welding 등 다수 customization (이후 롤백)
