"""Procedural walk rig for a pictogram pedestrian.

World units = output pixels in the figure plane. y is UP from the ground (0).
Feet are planted in world space (no sliding); legs are solved with 2-bone IK.
"""
import math
import numpy as np

# ---------------- timing ----------------
CYCLE_S = 1.04          # seconds per full gait cycle (2 steps) -> ~115 steps/min
STRIDE = 157.0          # px per cycle (2 steps of ~78 px ~ 0.69 m)

# ---------------- proportions (px) ----------------
THIGH = 48.0
SHIN = 48.0
LEG = THIGH + SHIN
R_HIP = 10.0            # leg stroke radius at hip
R_KNEE = 8.6
R_ANKLE = 8.0
FOOT = 17.0             # ankle-center -> toe-center
R_TOE = 5.6
TORSO = 51.0            # pelvis -> shoulder line
R_PELVIS = 11.5
R_CHEST = 15.5
NECK_GAP = 5.0
R_HEAD = 17.0
UPPER_ARM = 35.0
FOREARM = 32.0
R_SHOULDER = 8.2
R_ELBOW = 7.4
R_HAND = 7.0

# ---------------- gait parameters ----------------
P_FF = 0.10             # heel strike -> foot flat
P_HO = 0.32             # heel off
P_TO = 0.62             # toe off (stance = 62 %)
A_STRIKE = math.radians(20)   # toes-up at heel strike
A_TOEOFF = math.radians(62)   # heel-up at toe off
TR_EXP = 2.5
OFS = 0.23 * STRIDE     # ankle ahead of pelvis at heel strike
BUMP_EXP = 0.8
W0K = 0.8
LIFT = 4.0              # extra swing lift (px)

H0 = 100.4              # mean pelvis height
BOB = 2.7               # pelvis bob amplitude
LOW_AT = -0.02          # lowest pelvis point, cycles after heel strike
PELVIS_ROT = 5.0        # near/far hip x offset (pelvis rotation seen from side)
SHOULDER_ROT = 2.4      # counter-rotation of shoulders

LEAN = math.radians(4.0)
LEAN_OSC = math.radians(0.9)

ARM_MEAN = math.radians(0.0)
ARM_AMP = math.radians(30.0)
ARM_LAG = 0.055         # arms trail legs (drag / overlap)
ELBOW_MIN = math.radians(10)
ELBOW_AMP = math.radians(30)
ELBOW_LAG = 0.045       # forearm trails upper arm

HEAD_LAG = 0.035        # head follows torso bob a touch late

# flat-foot angle (both caps tangent to the ground)
A_FLAT = math.atan2(R_TOE - R_ANKLE, math.sqrt(FOOT**2 - (R_ANKLE - R_TOE)**2))


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def hermite(p0, m0, p1, m1, s):
    s2, s3 = s * s, s * s * s
    return ((2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * m0
            + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * m1)


def pelvis(c):
    """Pelvis center at global cycle time c."""
    x = c * STRIDE
    # lowest just after each heel strike (two strikes per cycle)
    y = H0 - BOB * math.cos(4 * math.pi * (c - LOW_AT))
    return np.array([x, y])


def foot_state(c, o):
    """Ankle position + foot angle (relative to flat) for a leg with phase offset o."""
    u = c + o
    k = math.floor(u)
    p = u - k
    xs = pelvis(k - o)[0] + OFS            # ankle x at this cycle's heel strike

    def toe_rocker(q):
        a = -A_TOEOFF * q ** TR_EXP        # accelerate into toe-off
        toe = np.array([xs + FOOT * math.cos(A_FLAT), R_TOE])
        ang = A_FLAT + a
        ank = toe - FOOT * np.array([math.cos(ang), math.sin(ang)])
        return ank, a

    if p < P_FF:
        a = A_STRIKE * (1 - smoothstep(p / P_FF))
        return np.array([xs, R_ANKLE]), a, True
    if p < P_HO:
        return np.array([xs, R_ANKLE]), 0.0, True
    if p < P_TO:
        q = (p - P_HO) / (P_TO - P_HO)
        ank, a = toe_rocker(q)
        return ank, a, True

    # ---- swing ----
    s = (p - P_TO) / (1 - P_TO)
    dur = (1 - P_TO) / (P_TO - P_HO)       # rescale derivatives from q to s
    a0_ank, a0 = toe_rocker(1.0)
    e = 1e-4
    a0b, ab = toe_rocker(1.0 - e)
    v_ank = (a0_ank - a0b) / e * dur
    w0 = (a0 - ab) / e * dur
    end = np.array([xs + STRIDE, R_ANKLE])
    x = hermite(a0_ank[0], v_ank[0], end[0], 0.0, s)
    sb = s ** BUMP_EXP
    bump = 16 * (sb * (1 - sb)) ** 2
    y = hermite(a0_ank[1], v_ank[1], end[1], -6.0, s) + LIFT * bump
    a = hermite(-A_TOEOFF, w0 * W0K, A_STRIKE, 0.0, s)
    return np.array([x, y]), a, False


def ik(hip, target, l1, l2):
    d_vec = target - hip
    d = float(np.hypot(*d_vec))
    ext = d / (l1 + l2)
    d_c = min(d, (l1 + l2) * 0.9995)
    cos_b = (l1 * l1 + d_c * d_c - l2 * l2) / (2 * l1 * d_c)
    b = math.acos(max(-1.0, min(1.0, cos_b)))
    ang = math.atan2(d_vec[1], d_vec[0]) + b   # +b -> knee bends forward (+x)
    knee = hip + l1 * np.array([math.cos(ang), math.sin(ang)])
    return knee, ext


def leg(c, o, hip):
    ank, a, planted = foot_state(c, o)
    knee, ext = ik(hip, ank, THIGH, SHIN)
    ang = A_FLAT + a
    toe = ank + FOOT * np.array([math.cos(ang), math.sin(ang)])
    return dict(hip=hip, knee=knee, ankle=ank, toe=toe, ext=ext, planted=planted)


def arm(c, o, shoulder):
    """o = phase of the same-side leg. Arm swings opposite to that leg."""
    p = c + o
    th = ARM_MEAN - ARM_AMP * math.cos(2 * math.pi * (p - ARM_LAG))
    # elbow flexes most when the arm is forward (p ~ 0.5)
    fl = ELBOW_MIN + ELBOW_AMP * 0.5 * (1 - math.cos(2 * math.pi * (p - ARM_LAG - ELBOW_LAG)))
    elbow = shoulder + UPPER_ARM * np.array([math.sin(th), -math.cos(th)])
    hand = elbow + FOREARM * np.array([math.sin(th + fl), -math.cos(th + fl)])
    return dict(shoulder=shoulder, elbow=elbow, hand=hand)


def pose(c):
    """Full pose at global cycle time c. Near (camera-side) leg has phase 0."""
    pv = pelvis(c)
    rot = math.cos(2 * math.pi * c)               # near leg forward at c=0
    lean = LEAN + LEAN_OSC * math.cos(4 * math.pi * (c - 0.45))
    tdir = np.array([math.sin(lean), math.cos(lean)])
    shoulder = pv + TORSO * tdir
    near_hip = pv + np.array([PELVIS_ROT * rot, 0.0])
    far_hip = pv - np.array([PELVIS_ROT * rot, 0.0])
    near_leg = leg(c, 0.0, near_hip)
    far_leg = leg(c, 0.5, far_hip)
    sh = shoulder - tdir * 3.0
    near_arm = arm(c, 0.0, sh - np.array([SHOULDER_ROT * rot, 0.0]))
    far_arm = arm(c, 0.5, sh + np.array([SHOULDER_ROT * rot, 0.0]))
    # head: rides on the torso, vertical bob arrives slightly late
    lag_y = pelvis(c - HEAD_LAG)[1] - pv[1]
    head = shoulder + tdir * (R_CHEST + NECK_GAP + R_HEAD) + np.array([0.0, lag_y * 0.8])
    return dict(pelvis=pv, shoulder=shoulder, head=head, near_leg=near_leg, far_leg=far_leg,
                near_arm=near_arm, far_arm=far_arm)


if __name__ == "__main__":
    n = 400
    exts, toe_min, slide = [], 1e9, 0.0
    prev = {}
    for i in range(n):
        c = i / n
        P = pose(c)
        for name in ("near_leg", "far_leg"):
            L = P[name]
            exts.append(L["ext"])
            toe_min = min(toe_min, L["toe"][1] - R_TOE, L["ankle"][1] - R_ANKLE)
    print("leg extension min/max: %.3f / %.3f" % (min(exts), max(exts)))
    print("min clearance (toe/ankle cap vs ground): %.2f px" % toe_min)
    print("A_FLAT deg", math.degrees(A_FLAT))
    hs = [pose(i / n)["head"][1] + R_HEAD for i in range(n)]
    print("figure height min/max %.1f %.1f" % (min(hs), max(hs)))
