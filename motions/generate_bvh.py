"""Procedurally generate Mixamo-named BVH motion files.

Public motion datasets (CMU, Mixamo) require account login or have
non-Mixamo bone names; generating our own keeps the pipeline self-contained
and makes retargeting a 1:1 name match.
"""
import math
from pathlib import Path

# Bone hierarchy: (name, parent, offset_vec, end_offset_or_None)
# Offsets are relative to parent in cm. Y is up, X is character's left.
BONES = [
    ("Hips",          None,            (0.0,  0.0, 0.0),  None),
    ("Spine",         "Hips",          (0.0, 10.0, 0.0),  None),
    ("Spine1",        "Spine",         (0.0, 10.0, 0.0),  None),
    ("Spine2",        "Spine1",        (0.0, 10.0, 0.0),  None),
    ("Neck",          "Spine2",        (0.0, 10.0, 0.0),  None),
    ("Head",          "Neck",          (0.0, 10.0, 0.0),  (0.0, 10.0, 0.0)),
    ("LeftShoulder",  "Spine2",        (5.0,  5.0, 0.0),  None),
    ("LeftArm",       "LeftShoulder",  (10.0, 0.0, 0.0),  None),
    ("LeftForeArm",   "LeftArm",       (20.0, 0.0, 0.0),  None),
    ("LeftHand",      "LeftForeArm",   (20.0, 0.0, 0.0),  (10.0, 0.0, 0.0)),
    ("RightShoulder", "Spine2",        (-5.0, 5.0, 0.0),  None),
    ("RightArm",      "RightShoulder", (-10.0, 0.0, 0.0), None),
    ("RightForeArm",  "RightArm",      (-20.0, 0.0, 0.0), None),
    ("RightHand",     "RightForeArm",  (-20.0, 0.0, 0.0), (-10.0, 0.0, 0.0)),
    ("LeftUpLeg",     "Hips",          (10.0, 0.0, 0.0),  None),
    ("LeftLeg",       "LeftUpLeg",     (0.0, -40.0, 0.0), None),
    ("LeftFoot",      "LeftLeg",       (0.0, -40.0, 0.0), None),
    ("LeftToeBase",   "LeftFoot",      (0.0, -5.0, 10.0), (0.0, 0.0, 5.0)),
    ("RightUpLeg",    "Hips",          (-10.0, 0.0, 0.0), None),
    ("RightLeg",      "RightUpLeg",    (0.0, -40.0, 0.0), None),
    ("RightFoot",     "RightLeg",      (0.0, -40.0, 0.0), None),
    ("RightToeBase",  "RightFoot",     (0.0, -5.0, 10.0), (0.0, 0.0, 5.0)),
]

ROOT_HEIGHT = 100.0  # initial Hips Y
FPS = 30
DURATION = 2.0
N_FRAMES = int(FPS * DURATION)


def _children_of(parent):
    return [b for b in BONES if b[1] == parent]


def _write_joint(lines, bone, depth):
    name, parent, offset, end_off = bone
    indent = "  " * depth
    kw = "ROOT" if parent is None else "JOINT"
    lines.append(f"{indent}{kw} {name}")
    lines.append(f"{indent}{{")
    lines.append(f"{indent}  OFFSET {offset[0]:.4f} {offset[1]:.4f} {offset[2]:.4f}")
    if parent is None:
        lines.append(f"{indent}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
    else:
        lines.append(f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation")
    children = _children_of(name)
    for c in children:
        _write_joint(lines, c, depth + 1)
    if end_off is not None and not children:
        lines.append(f"{indent}  End Site")
        lines.append(f"{indent}  {{")
        lines.append(f"{indent}    OFFSET {end_off[0]:.4f} {end_off[1]:.4f} {end_off[2]:.4f}")
        lines.append(f"{indent}  }}")
    lines.append(f"{indent}}}")


def _hierarchy_lines():
    lines = ["HIERARCHY"]
    root = next(b for b in BONES if b[1] is None)
    _write_joint(lines, root, 0)
    return lines


def _bone_order():
    """Return bones in same order as written in HIERARCHY (DFS)."""
    order = []
    def walk(name):
        order.append(name)
        for c in _children_of(name):
            walk(c[0])
    walk("Hips")
    return order


def _frame_to_channels(rotations, root_pos):
    """rotations: dict {bone_name: (rz, rx, ry)} in degrees. Missing bones default to 0."""
    parts = []
    for name in _bone_order():
        if name == "Hips":
            parts.extend([f"{root_pos[0]:.4f}", f"{root_pos[1]:.4f}", f"{root_pos[2]:.4f}"])
        rz, rx, ry = rotations.get(name, (0.0, 0.0, 0.0))
        parts.extend([f"{rz:.4f}", f"{rx:.4f}", f"{ry:.4f}"])
    return " ".join(parts)


def _write_bvh(path, motion_func):
    lines = _hierarchy_lines()
    lines.append("MOTION")
    lines.append(f"Frames: {N_FRAMES}")
    lines.append(f"Frame Time: {1.0 / FPS:.6f}")
    for f in range(N_FRAMES):
        rots, root_pos = motion_func(f)
        lines.append(_frame_to_channels(rots, root_pos))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[motion] wrote {path}")


# ---------- Motion patterns ----------

def _walk(f):
    phase = (f / N_FRAMES) * 2 * math.pi * 2  # 2 cycles
    leg = math.sin(phase) * 25
    arm = math.sin(phase + math.pi) * 25
    rots = {
        "Hips":         (0, 0, math.sin(phase) * 3),
        "LeftUpLeg":    (0, leg, 0),
        "RightUpLeg":   (0, -leg, 0),
        "LeftLeg":      (0, max(0, -leg) * 1.2, 0),
        "RightLeg":     (0, max(0, leg) * 1.2, 0),
        "LeftArm":      (0, 0, arm),
        "RightArm":     (0, 0, -arm),
        "LeftForeArm":  (0, max(0, arm) * 0.5, 0),
        "RightForeArm": (0, max(0, -arm) * 0.5, 0),
    }
    return rots, (0, ROOT_HEIGHT, 0)


def _run(f):
    phase = (f / N_FRAMES) * 2 * math.pi * 4  # 4 cycles
    leg = math.sin(phase) * 45
    arm = math.sin(phase + math.pi) * 45
    bob = abs(math.sin(phase * 2)) * 4
    rots = {
        "Hips":         (0, 0, math.sin(phase) * 5),
        "Spine1":       (math.sin(phase) * 5, 0, 0),
        "LeftUpLeg":    (0, leg, 0),
        "RightUpLeg":   (0, -leg, 0),
        "LeftLeg":      (0, max(0, -leg) * 1.5, 0),
        "RightLeg":     (0, max(0, leg) * 1.5, 0),
        "LeftArm":      (0, 0, arm),
        "RightArm":     (0, 0, -arm),
        "LeftForeArm":  (0, 60, 0),
        "RightForeArm": (0, 60, 0),
    }
    return rots, (0, ROOT_HEIGHT + bob, 0)


def _idle(f):
    phase = (f / N_FRAMES) * 2 * math.pi
    sway = math.sin(phase) * 2
    breath = math.sin(phase * 2) * 1.5
    rots = {
        "Hips":   (0, 0, sway),
        "Spine":  (breath, 0, 0),
        "Spine2": (-breath * 0.5, 0, 0),
        "Head":   (0, math.sin(phase) * 3, 0),
    }
    return rots, (0, ROOT_HEIGHT, 0)


def _wave(f):
    phase = (f / N_FRAMES) * 2 * math.pi * 3  # 3 wave cycles
    wave = math.sin(phase) * 30
    rots = {
        "RightShoulder": (0, 0, -30),
        "RightArm":      (0, 0, -130),  # raise arm
        "RightForeArm":  (0, wave, 0),
        "RightHand":     (0, wave * 0.5, 0),
    }
    return rots, (0, ROOT_HEIGHT, 0)


def _jump(f):
    t = f / (N_FRAMES - 1)
    if t < 0.3:
        # Crouch
        crouch = (t / 0.3)
        leg_bend = crouch * 60
        hip_y = ROOT_HEIGHT - crouch * 20
        height_off = 0
    elif t < 0.55:
        # Extend + lift off
        ext = (t - 0.3) / 0.25
        leg_bend = (1 - ext) * 60
        hip_y = ROOT_HEIGHT - (1 - ext) * 20 + ext * 30
        height_off = 0
    elif t < 0.75:
        # Apex / float
        leg_bend = 30 - (t - 0.55) / 0.2 * 30
        hip_y = ROOT_HEIGHT + 30
        height_off = 0
    else:
        # Land + recover
        rec = (t - 0.75) / 0.25
        leg_bend = rec * 40
        hip_y = ROOT_HEIGHT + (1 - rec) * 30 - rec * 10
        height_off = 0
    rots = {
        "LeftUpLeg":  (0, leg_bend, 0),
        "RightUpLeg": (0, -leg_bend, 0),
        "LeftLeg":    (0, leg_bend * 1.5, 0),
        "RightLeg":   (0, leg_bend * 1.5, 0),
        "LeftArm":    (0, 0, -30 if t < 0.55 else 60),
        "RightArm":   (0, 0,  30 if t < 0.55 else -60),
    }
    return rots, (0, hip_y, height_off)


def _dance(f):
    phase = (f / N_FRAMES) * 2 * math.pi * 2
    sway = math.sin(phase) * 12
    arm_lift = (math.sin(phase + math.pi / 2) + 1) * 30  # 0..60
    rots = {
        "Hips":         (sway, 0, 0),
        "Spine":        (sway * 0.5, 0, 0),
        "Spine2":       (-sway * 0.3, 0, 0),
        "Head":         (0, math.sin(phase * 2) * 8, 0),
        "LeftArm":      (0, 0, 50 + arm_lift),
        "RightArm":     (0, 0, -(50 + arm_lift)),
        "LeftForeArm":  (0, 30, 0),
        "RightForeArm": (0, 30, 0),
        "LeftUpLeg":    (0, math.sin(phase) * 10, 0),
        "RightUpLeg":   (0, -math.sin(phase) * 10, 0),
    }
    return rots, (0, ROOT_HEIGHT, 0)


def _clap(f):
    phase = (f / N_FRAMES) * 2 * math.pi * 3  # 3 claps
    close = (math.sin(phase) + 1) * 0.5  # 0..1
    swing = close * 25
    rots = {
        "LeftShoulder":  (0, 25, 0),
        "LeftArm":       (0, 70, -swing),
        "LeftForeArm":   (0, 70, 0),
        "RightShoulder": (0, 25, 0),
        "RightArm":      (0, 70, swing),
        "RightForeArm":  (0, 70, 0),
        "Spine":         (math.sin(phase) * 3, 0, 0),
    }
    return rots, (0, ROOT_HEIGHT, 0)


def _bow(f):
    t = f / (N_FRAMES - 1)
    bend = (math.sin(t * math.pi)) * 55  # ease in/out, peak at middle
    rots = {
        "Spine":  (bend * 0.35, 0, 0),
        "Spine1": (bend * 0.30, 0, 0),
        "Spine2": (bend * 0.25, 0, 0),
        "Neck":   (bend * 0.20, 0, 0),
        "Head":   (-bend * 0.15, 0, 0),
        "LeftArm":  (0, 0,  bend * 0.3),
        "RightArm": (0, 0, -bend * 0.3),
    }
    return rots, (0, ROOT_HEIGHT, 0)


def _spin(f):
    t = f / (N_FRAMES - 1)
    angle = t * 360.0  # one full revolution
    rots = {
        "Hips":     (0, 0, angle),
        "LeftArm":  (0, 0,  60),
        "RightArm": (0, 0, -60),
        "Head":     (0, 0, math.sin(t * math.pi * 2) * 10),
    }
    return rots, (0, ROOT_HEIGHT, 0)


def _stretch(f):
    t = f / (N_FRAMES - 1)
    reach = math.sin(t * math.pi)  # ease in/out
    arm_up = reach * 165
    arch = reach * -12
    rots = {
        "Spine":         (arch * 0.5, 0, 0),
        "Spine1":        (arch * 0.3, 0, 0),
        "Spine2":        (arch * 0.2, 0, 0),
        "Head":          (arch * 0.5, 0, 0),
        "LeftShoulder":  (0, 0, arm_up * 0.15),
        "LeftArm":       (0, 0, arm_up),
        "LeftForeArm":   (0, reach * 20, 0),
        "RightShoulder": (0, 0, -arm_up * 0.15),
        "RightArm":      (0, 0, -arm_up),
        "RightForeArm":  (0, reach * 20, 0),
    }
    return rots, (0, ROOT_HEIGHT, 0)


def _kick(f):
    phase = (f / N_FRAMES) * 2 * math.pi * 2  # 2 kicks
    left_kick = max(0, math.sin(phase)) * 70
    right_kick = max(0, math.sin(phase + math.pi)) * 70
    arm_swing = math.sin(phase) * 30
    rots = {
        "LeftUpLeg":    (0, left_kick, 0),
        "LeftLeg":      (0, -left_kick * 0.4, 0),
        "RightUpLeg":   (0, right_kick, 0),
        "RightLeg":     (0, -right_kick * 0.4, 0),
        "LeftArm":      (0, 0, 30 + arm_swing),
        "RightArm":     (0, 0, -30 - arm_swing),
        "Hips":         (0, 0, math.sin(phase) * 5),
    }
    return rots, (0, ROOT_HEIGHT, 0)


MOTIONS = {
    "walk":    _walk,
    "run":     _run,
    "idle":    _idle,
    "wave":    _wave,
    "jump":    _jump,
    "dance":   _dance,
    "clap":    _clap,
    "bow":     _bow,
    "spin":    _spin,
    "stretch": _stretch,
    "kick":    _kick,
}


def main():
    out_dir = Path(__file__).resolve().parent
    for name, fn in MOTIONS.items():
        _write_bvh(out_dir / f"{name}.bvh", fn)
    print(f"[motion] generated {len(MOTIONS)} BVH files in {out_dir}")


if __name__ == "__main__":
    main()
