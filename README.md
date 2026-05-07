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

### 2. 3D 모델 (TripoSR)
```bash
bash install_3d_model.sh   # TripoSR clone + PyMCubes 등 의존성 설치
```

### 3. Blender (자동 리깅/모션 적용용)
- https://www.blender.org/download/ 에서 4.x LTS 설치
- `BLENDER_PATH` 환경변수 등록 또는 PATH에 등록

### 4. 모션 데이터
```bash
bash download_motions.sh   # walk/run/idle/wave/jump.bvh 생성
```

## 사용

### 웹 UI (가장 쉬움)
```bash
python app.py
```
브라우저가 자동으로 http://127.0.0.1:7860 에 열립니다. 이미지 업로드 → 모션 선택 → "실행" → FBX 다운로드.

### 전체 파이프라인 (모션 포함)
```bash
python main.py --input photo.jpg --output result.fbx --motion walk
```

### 리깅까지만 (모션 없음)
```bash
python main.py --input photo.jpg --output result.fbx
```

### 메시까지만
```bash
python main.py --input photo.jpg --output result.glb --mesh-only
```

### 배경이 이미 제거된 이미지로 시작
```bash
python main.py --input nobg.png --output result.fbx --skip-bg --motion run
```

### 폴더 일괄 처리
```bash
python main.py --input ./photos/ --output ./output/ --motion idle
```

### 사용 가능한 모션 확인
```bash
python main.py --list-motions
```

### 중단 후 재시작
```bash
python main.py --input photo.jpg --output result.fbx --motion walk --resume
```

## 출력 콘솔 예시
```
[log] output/logs/pipeline_20260506_182301.log
[1/4] 배경 제거 ...............  완료 (3.2s, VRAM: 1.1GB)
[2/4] 메시 생성 ...............  완료 (45.7s, VRAM: 9.8GB)
[3/4] 자동 리깅 ...............  완료 (12.1s, VRAM: 0.0GB)
[4/4] 모션 적용 (walk) ........  완료 (8.3s, VRAM: 0.0GB)

✓ 최종 출력: result.fbx
```

## 디렉토리 구조

```
img2mesh_pipeline/
├── main.py                    ← 통합 CLI 진입점
├── remove_bg.py               ← 배경 제거 (rembg/u2net)
├── generate_mesh.py           ← 3D 메시 생성 (TripoSR)
├── auto_rig.py                ← 자동 리깅 orchestrator
├── apply_motion.py            ← 모션 적용 orchestrator
├── rig_in_blender.py          ← Blender headless 리깅 스크립트
├── retarget_in_blender.py     ← Blender headless 리타게팅 스크립트
├── skeleton_config.json       ← Mixamo 22-bone 템플릿
├── retarget_config.json       ← BVH→FBX 본 매핑
├── setup_env.sh
├── environment.yml
├── install_3d_model.sh
├── download_motions.sh
├── motions/
│   ├── generate_bvh.py
│   ├── walk.bvh / run.bvh / idle.bvh / wave.bvh / jump.bvh
│   └── README.md
├── output/
│   ├── nobg/        ← 배경 제거 결과
│   ├── mesh/        ← 3D 메시
│   ├── rigged/      ← 리깅된 FBX
│   ├── animated/    ← 애니메이션 FBX
│   ├── preview/     ← (예약)
│   └── logs/        ← 실행 로그
└── README.md
```

## CLI 인자

| 인자 | 설명 |
|---|---|
| `--input PATH` | 입력 이미지(파일 또는 폴더) |
| `--output PATH` | 최종 출력(파일 또는 폴더) |
| `--motion NAME\|PATH` | `walk`/`run`/`idle`/`wave`/`jump` 또는 `.bvh` 파일 경로 |
| `--list-motions` | 사용 가능한 모션 목록 출력 후 종료 |
| `--skip-bg` | 배경 제거 단계 건너뜀 (이미 RGBA 입력) |
| `--mesh-only` | 메시 생성까지만 |
| `--rig-only` | 리깅까지만 (모션 단계 건너뜀) |
| `--format glb\|obj` | 메시 포맷 (기본 `glb`) |
| `--keep-intermediate` / `--no-keep-intermediate` | 중간 결과 보존 (기본 보존) |
| `--no-preview` | 모션 미리보기 생략 (현 버전 미구현, 플래그 예약) |
| `--resume` | 이전 실행 중단 지점부터 재시작 |

## 알려진 제한사항

- **VRAM**: 12GB(RTX 5070) 기준 입력 이미지 1024×1024 권장. 더 큰 입력은 OOM 위험
- **입력 사진**: 전신이 보이는 정면, T-pose에 가까운 자세에서 최상의 결과
- **비휴머노이드**: 동물·기물 등은 Mixamo 22-bone 템플릿이 강제 fitting되므로 weight painting이 어색할 수 있음
- **BVH 모션**: 현재 동봉된 5개는 절차적 생성으로 단순한 사인파 패턴. 실제 mocap 품질이 필요하면 CMU/Mixamo BVH로 교체하고 `retarget_config.json` 매핑 수정
- **루트 이동**: 현재 retargeting은 회전만 복사 (root translation 미적용) — 걷기/달리기는 "제자리" 동작으로 표시됨. jump의 vertical lift도 미반영
- **미리보기**: `output/preview/` 폴더는 예약돼 있지만 현 버전에서는 자동 생성하지 않음

## 기술 스택

- **Background removal**: rembg (u2net) + ONNX Runtime GPU
- **3D mesh**: TripoSR (Stability AI) + PyMCubes (CPU marching cubes)
- **Auto-rig**: Blender 4.5 headless + Mixamo 22-bone template fitting
- **Motion**: Blender headless + Copy Rotation constraints + bake
- **GPU**: PyTorch 2.11+cu128 (RTX 5070 sm_120 호환)
