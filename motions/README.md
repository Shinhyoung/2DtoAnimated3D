# Motions

Procedurally generated BVH files with Mixamo-conventional bone names.

| Motion | Description |
|---|---|
| walk.bvh | 2-cycle walking gait |
| run.bvh | 4-cycle running with vertical bob |
| idle.bvh | Subtle hip sway + breathing |
| wave.bvh | Right-arm raised wave |
| jump.bvh | Crouch → extend → land |
| dance.bvh | Hip sway + arms-up cycling |
| clap.bvh | Both arms forward, 3 claps |
| bow.bvh | Forward spine bend + return |
| spin.bvh | Full 360° turn around vertical axis |
| stretch.bvh | Both arms reach overhead with back arch |
| kick.bvh | Alternating front kicks (2 reps) |
| breakdance.bvh | External Mixamo-named BVH (~8s, 250 frames) |

All BVH files use the same skeleton hierarchy (matches `../skeleton_config.json`), so retargeting is a 1:1 bone-name mapping defined in `../retarget_config.json`.

To regenerate (or after editing `generate_bvh.py`):
```bash
bash ../download_motions.sh
# or directly:
python generate_bvh.py
```

Custom BVH files dropped into this folder are usable via `--motion <stem>` automatically.
