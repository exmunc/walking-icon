"""Moonwalk icon: pictogram moonwalks backward over a zebra crossing (in place, stripes scroll right).

Per step: one foot is up on its ball (heel raised, knee popped forward) and the ball is locked to the
ground; the other leg is straight and its flat shoe slides back. Then a quick snap swaps the roles.
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

from render import capsule, _morph, downsample

SIZE, SS, FPS = 600, 4, 50
STEP_FR = 30                       # 0.6 s per slide
N = 2 * STEP_FR                    # loop = 2 steps = 1.2 s
K = 1.9                            # canvas px per rig unit
GROUND, CENTER_X = 488.0, 288.0
BG, INK = (255, 255, 255), (26, 26, 28)
INK_FAR = INK                      # one-colour icon: limbs may merge

# ---- rig (units; y up) ----
THIGH = SHIN = 48.0
L = THIGH + SHIN
R_HIP, R_KNEE, R_CUFF = 10.0, 8.6, 8.6
R_HEEL, R_TOE, SHOE = 7.6, 6.2, 25.0        # shoe: heel circle (= ankle) -> toe circle
SOCK = 3.6                                   # visible white sock between trouser cuff and shoe
TORSO, R_PELVIS, R_CHEST = 51.0, 11.5, 15.5
NECK_GAP, R_HEAD = 5.0, 17.0
UPPER_ARM, FOREARM = 35.0, 32.0
R_SHOULDER, R_ELBOW, R_HAND = 8.2, 7.4, 7.0
A_FLAT = math.atan2(R_TOE - R_HEEL, math.sqrt(SHOE ** 2 - (R_HEEL - R_TOE) ** 2))

D = 40.0                     # ground travel per step (rig units)
C0 = SHOE * math.cos(A_FLAT) # flat ankle slides symmetrically around the hip
BALL_UP = math.radians(46)   # heel raised on the "toe" foot
SWITCH = 0.27                # fraction of a step used by the snap
EXT = 0.985                  # sliding leg stays this straight
LEAN = math.radians(5)

# stripes (canvas px): ground travel per loop is a whole number of periods -> seamless
PERIOD = D * K                     # two stripes per loop
STRIPE_W, STRIPE_TOP, STRIPE_BOT, SKEW, STRIPE_R = 44.0, GROUND + 12, GROUND + 62, 18.0, 5.0
FADE_IN, FADE_W = 40.0, 150.0


def ease_out(u):
    """Snappy ease-out (Hermite, start slope 1): a pop that still spreads over ~8 frames."""
    u = min(max(u, 0.0), 1.0)
    return (u ** 3 - 2 * u * u + u) + (-2 * u ** 3 + 3 * u * u)


def slide(t):
    """Body-relative slide progress 0..1; never stalls, a bit quicker mid-step."""
    return t - 0.3 * math.sin(2 * math.pi * t) / (2 * math.pi)


def foot(ball_x, lift):
    """Ball (toe circle) on the ground at ball_x; lift = heel raise angle (rotates about the ball)."""
    toe = np.array([ball_x, R_TOE])
    a = A_FLAT - lift
    heel = toe - SHOE * np.array([math.cos(a), math.sin(a)])
    return heel, toe


def ik(hip, target):
    d = target - hip
    dist = float(np.hypot(*d))
    dc = min(dist, L * 0.9995)
    b = math.acos(max(-1.0, min(1.0, (THIGH ** 2 + dc ** 2 - SHIN ** 2) / (2 * THIGH * dc))))
    ang = math.atan2(d[1], d[0]) + b                # knee forward (+x)
    return hip + THIGH * np.array([math.cos(ang), math.sin(ang)]), dist / L


def pose(i):
    step, f = divmod(i % N, STEP_FR)
    t = f / STEP_FR
    u = ease_out(t / SWITCH)
    # ball positions relative to the pelvis (pelvis x = 0)
    toe_ball = C0 - D / 2 + D * t                 # locked to the ground (ground moves +x)
    flat_ball = C0 + D / 2 - D * slide(t)         # slides back
    toe_lift = BALL_UP * u                        # heel pops up
    flat_lift = BALL_UP * (1 - u)                 # heel drops
    # pelvis height keeps the sliding leg straight; small dip on the snap = weight
    flat_ank_x = flat_ball - C0
    h = R_HEEL + math.sqrt((EXT * L) ** 2 - flat_ank_x ** 2)
    h -= 1.6 * math.sin(math.pi * min(1.0, t / 0.35))
    pv = np.array([0.0, h])

    legs = {}
    near_is_toe = step == 0
    for name, is_toe in (("near", near_is_toe), ("far", not near_is_toe)):
        heel, toe = foot(toe_ball, toe_lift) if is_toe else foot(flat_ball, flat_lift)
        hip = pv + np.array([1.5 if name == "near" else -1.5, 0.0])
        knee, ext = ik(hip, heel)
        legs[name] = dict(hip=hip, knee=knee, ankle=heel, toe=toe, ext=ext)

    # upper body: steady, tiny counter-move after the snap (overlap)
    tl = t - 0.06
    nod = 1.0 * math.sin(math.pi * min(1.0, max(0.0, tl) / 0.4))
    lean = LEAN + math.radians(1.2) * nod
    tdir = np.array([math.sin(lean), math.cos(lean)])
    shoulder = pv + TORSO * tdir
    head = shoulder + tdir * (R_CHEST + NECK_GAP + R_HEAD) + np.array([0.0, -0.6 * nod])
    arms = {}
    loop_t = (i % N) / N
    for name, sgn in (("near", 1), ("far", -1)):
        # gentle opposite sway, trailing the feet a little
        sway = math.radians(7) * sgn * math.cos(2 * math.pi * (loop_t - 0.08))
        th = math.radians(6) + sway
        fl = math.radians(30) + 0.5 * sway
        sh = shoulder - tdir * 3.0
        el = sh + UPPER_ARM * np.array([math.sin(th), -math.cos(th)])
        ha = el + FOREARM * np.array([math.sin(th + fl), -math.cos(th + fl)])
        arms[name] = dict(shoulder=sh, elbow=el, hand=ha)
    return dict(pelvis=pv, shoulder=shoulder, head=head, tdir=tdir, legs=legs, arms=arms, t=t, step=step)


def masks(P):
    size = (SIZE * SS, SIZE * SS)
    k = K * SS

    def px(w):
        return ((CENTER_X + w[0] * K) * SS, (GROUND - w[1] * K) * SS)

    def m(fn):
        img = Image.new("L", size, 0)
        fn(ImageDraw.Draw(img))
        return np.asarray(img) > 127

    def trouser(d, Lg):
        knee, ank = Lg["knee"], Lg["ankle"]
        up = (knee - ank) / np.hypot(*(knee - ank))
        cuff = ank + up * (R_HEEL + SOCK + R_CUFF)
        capsule(d, px(Lg["hip"]), R_HIP * k, px(knee), R_KNEE * k)
        capsule(d, px(knee), R_KNEE * k, px(cuff), R_CUFF * k)

    def shoe(d, Lg):
        hl, tp = Lg["ankle"], Lg["toe"]
        capsule(d, px(hl), R_HEEL * k, px(tp), R_TOE * k)
        # instep: loafer silhouette instead of a plain stick
        u = (tp - hl) / np.hypot(*(tp - hl))
        n = np.array([-u[1], u[0]])
        capsule(d, px(hl + n * 1.8), R_HEEL * k, px(hl + u * 9.0 + n * 2.6), 6.4 * k)

    def arm(d, A, g=0.0, partial=False):
        capsule(d, px(A["shoulder"]), (R_SHOULDER - (1.5 if partial else -g)) * k, px(A["elbow"]), (R_ELBOW + g) * k)
        capsule(d, px(A["elbow"]), (R_ELBOW + g) * k, px(A["hand"]), (R_HAND + g) * k)

    def body(d):
        capsule(d, px(P["pelvis"]), R_PELVIS * k, px(P["shoulder"]), R_CHEST * k)
        h = px(P["head"])
        r = R_HEAD * k
        d.ellipse([h[0] - r, h[1] - r, h[0] + r, h[1] + r], fill=255)
        # fedora, tipped down at the front
        tilt = -math.radians(9)
        c, s = math.cos(tilt), math.sin(tilt)

        def hp(x, y):
            return px(P["head"] + np.array([x * c - y * s, x * s + y * c]))

        capsule(d, hp(-23, 9), 3.2 * k, hp(26, 9), 3.2 * k)                   # brim
        crown = [hp(-15.5, 9), hp(15.5, 9), hp(13, 27), hp(4, 25.5), hp(-12.5, 27.5)]
        d.polygon(crown, fill=255)
        for a, b in zip(crown, crown[1:] + crown[:1]):
            capsule(d, a, 2.2 * k, b, 2.2 * k)

    NL, FL = P["legs"]["near"], P["legs"]["far"]
    near = m(body) | m(lambda d: trouser(d, NL)) | m(lambda d: shoe(d, NL)) | m(lambda d: arm(d, P["arms"]["near"]))
    far = m(lambda d: trouser(d, FL)) | m(lambda d: shoe(d, FL)) | m(lambda d: arm(d, P["arms"]["far"]))
    far &= ~near
    # contour line of the near arm where it crosses the torso / near trouser leg (kept inside the edge)
    NA = m(lambda d: arm(d, P["arms"]["near"]))
    NAd = m(lambda d: arm(d, P["arms"]["near"], 3.0, partial=True))
    inner = _morph(near & ~NA, int(round(3.0 * k)), np.minimum)
    line = NAd & ~NA & inner
    return near & ~line, far


def stripes(i):
    offset = (i / N) * PERIOD                     # scroll right: he travels backward
    total = np.zeros((SIZE, SIZE), np.float32)
    r = STRIPE_R
    x = offset % PERIOD - PERIOD * 2
    while x < SIZE + PERIOD:
        cx = x + (SKEW + STRIPE_W) / 2
        tt = min(cx - FADE_IN, SIZE - FADE_IN - cx) / FADE_W
        alpha = 0.0 if tt <= 0 else (1.0 if tt >= 1 else tt * tt * (3 - 2 * tt))
        if alpha > 0:
            img = Image.new("L", (SIZE * SS, SIZE * SS), 0)
            d = ImageDraw.Draw(img)
            pts = [(x + SKEW + r * 1.2, STRIPE_TOP + r), (x + SKEW + STRIPE_W - r * 1.2, STRIPE_TOP + r),
                   (x + STRIPE_W - r * 1.2, STRIPE_BOT - r), (x + r * 1.2, STRIPE_BOT - r)]
            sp = [(a * SS, b * SS) for a, b in pts]
            d.polygon(sp, fill=255)
            for p0, p1 in zip(sp, sp[1:] + sp[:1]):
                capsule(d, p0, r * SS, p1, r * SS)
            total = np.maximum(total, downsample(np.asarray(img) > 127) * alpha)
        x += PERIOD
    return total


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "moon_frames"
    os.makedirs(out, exist_ok=True)
    for i in range(N):
        P = pose(i)
        near, far = masks(P)
        cn = np.maximum(downsample(near), stripes(i))[..., None]
        cf = np.minimum(downsample(far)[..., None], 1 - cn)
        img = np.array(BG, np.float32) * (1 - cn - cf) + np.array(INK, np.float32) * cn \
            + np.array(INK_FAR, np.float32) * cf
        Image.fromarray(np.clip(img + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(out, "f%04d.png" % i))
    print("frames", N)


if __name__ == "__main__":
    main()
