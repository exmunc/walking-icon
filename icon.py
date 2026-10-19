"""Animated filled icon: pictogram walking in place over a scrolling zebra crossing.

The zebra scrolls at exactly the walking speed, so planted feet stay locked to the stripes.
One gait cycle = one seamless loop.
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

import rig
from render import capsule, _morph, downsample

SIZE = 600
SS = 4
FPS = 50
N = round(rig.CYCLE_S * FPS)          # 52 frames = 1 cycle
K = 2.0                               # canvas px per rig unit
GROUND = 470.0                        # canvas y of the ground line
CENTER_X = 300.0
BG = (255, 255, 255)
INK = (26, 26, 28)
INK_FAR = (160, 160, 166)
GAP = 3.0                             # rig units: separation between overlapping parts
INNER_MARGIN = 3.0

# zebra: 3 stripes per gait cycle so the loop is seamless
PERIOD = rig.STRIDE * K / 3           # 104.7 px
STRIPE_W = 60.0
STRIPE_TOP = GROUND + 12
STRIPE_BOT = GROUND + 66
SKEW = 18.0                           # top edge leans forward
STRIPE_R = 5.0
FADE_IN, FADE_W = 40.0, 150.0         # stripes fade out toward the icon edges


def figure_masks(P, shift_x):
    size = (SIZE * SS, SIZE * SS)

    def to_px(w):
        return ((CENTER_X + (w[0] - shift_x) * K) * SS, (GROUND - w[1] * K) * SS)

    k = K * SS

    def new():
        img = Image.new("L", size, 0)
        return img, ImageDraw.Draw(img)

    def leg(d, L, g=0.0, partial=False):
        hip, knee, ank, toe = (to_px(L[n]) for n in ("hip", "knee", "ankle", "toe"))
        capsule(d, hip, (rig.R_HIP + (0 if partial else g)) * k, knee, (rig.R_KNEE + g) * k)
        capsule(d, knee, (rig.R_KNEE + g) * k, ank, (rig.R_ANKLE + g) * k)
        capsule(d, ank, (rig.R_ANKLE + g) * k, toe, (rig.R_TOE + g) * k)

    def arm(d, A, g=0.0, partial=False):
        sh, el, ha = (to_px(A[n]) for n in ("shoulder", "elbow", "hand"))
        capsule(d, sh, (rig.R_SHOULDER - (1.5 if partial else -g)) * k, el, (rig.R_ELBOW + g) * k)
        capsule(d, el, (rig.R_ELBOW + g) * k, ha, (rig.R_HAND + g) * k)

    def body(d, g=0.0):
        capsule(d, to_px(P["pelvis"]), (rig.R_PELVIS + g) * k, to_px(P["shoulder"]), (rig.R_CHEST + g) * k)
        h = to_px(P["head"])
        r = (rig.R_HEAD + g) * k
        d.ellipse([h[0] - r, h[1] - r, h[0] + r, h[1] + r], fill=255)

    def m(fn, *a, **kw):
        img, d = new()
        fn(d, *a, **kw)
        return np.asarray(img) > 127

    FA, FL = m(arm, P["far_arm"]), m(leg, P["far_leg"])
    B, NL, NA = m(body), m(leg, P["near_leg"]), m(arm, P["near_arm"])
    NAd = m(arm, P["near_arm"], GAP, partial=True)

    near = B | NL | NA
    far = (FA | FL) & ~near                 # far limbs: lighter tone, simply occluded
    # near arm crossing the body: a gap line kept inside the silhouette (edge never notched)
    inner = _morph(B | NL, int(round(INNER_MARGIN * k)), np.minimum)
    line = NAd & ~NA & inner
    return near & ~line, far


def stripes_cov(frame):
    """Each stripe is a flat shape whose opacity fades in/out near the icon edges."""
    offset = -(frame / N) * rig.STRIDE * K          # scroll left at walking speed
    r = STRIPE_R
    total = np.zeros((SIZE, SIZE), np.float32)
    x = offset % PERIOD - PERIOD * 2
    while x < SIZE + PERIOD:
        cx = x + (SKEW + STRIPE_W) / 2
        t = min(cx - FADE_IN, SIZE - FADE_IN - cx) / FADE_W
        alpha = 0.0 if t <= 0 else (1.0 if t >= 1 else t * t * (3 - 2 * t))
        if alpha > 0:
            img = Image.new("L", (SIZE * SS, SIZE * SS), 0)
            d = ImageDraw.Draw(img)
            # parallelogram shrunk by r, then rounded by capsules along its edges
            pts = [(x + SKEW + r * 1.2, STRIPE_TOP + r), (x + SKEW + STRIPE_W - r * 1.2, STRIPE_TOP + r),
                   (x + STRIPE_W - r * 1.2, STRIPE_BOT - r), (x + r * 1.2, STRIPE_BOT - r)]
            sp = [(px * SS, py * SS) for px, py in pts]
            d.polygon(sp, fill=255)
            for p0, p1 in zip(sp, sp[1:] + sp[:1]):
                capsule(d, p0, r * SS, p1, r * SS)
            total = np.maximum(total, downsample(np.asarray(img) > 127) * alpha)
        x += PERIOD
    return total


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "icon_frames"
    os.makedirs(out, exist_ok=True)
    for i in range(N):
        c = i / N
        P = rig.pose(c)
        near, far = figure_masks(P, P["pelvis"][0])          # walk in place
        cn = np.maximum(downsample(near), stripes_cov(i))[..., None]
        cf = np.minimum(downsample(far)[..., None], 1 - cn)
        img = (np.array(BG, np.float32) * (1 - cn - cf) + np.array(INK, np.float32) * cn
               + np.array(INK_FAR, np.float32) * cf)
        Image.fromarray(np.clip(img + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(out, "f%04d.png" % i))
    print("frames", N)


if __name__ == "__main__":
    main()
