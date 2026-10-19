"""Render the crosswalk walk-cycle animation to PNG frames."""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import rig

W, H = 960, 540
SS = 4
FPS = int(os.environ.get("FPS", 50))
N_CYCLES = 7
FRAMES_PER_CYCLE = round(rig.CYCLE_S * FPS)
N_FRAMES = N_CYCLES * FRAMES_PER_CYCLE
X_START = -72.0                 # pelvis x at frame 0 (fully off-screen)

# ---------------- camera / ground projection ----------------
CX, Y_H = 480.0, 257.0          # vanishing point / horizon
F = 1200.0
CAM_H = 1.6                     # m
Y_FEET = 440.0
D = F * CAM_H / (Y_FEET - Y_H)  # depth of the walking line (m)

ROAD_HALF = 2.6
CURB = 0.16
CROSS_HALF = 1.6
STRIPE_W, STRIPE_GAP, N_STRIPES = 0.5, 0.5, 5

# ---------------- palette ----------------
SKY = (226, 235, 240)
CITY_FAR = (208, 219, 226)
CITY_NEAR = (193, 207, 216)
SIDEWALK = (216, 210, 198)
CURB_C = (238, 234, 226)
ROAD = (122, 129, 140)
STRIPE = (245, 243, 238)
DASH = (226, 222, 212)
POLE = (52, 57, 66)
NEAR = (29, 38, 53)
FAR = (64, 77, 97)
LINE = (92, 104, 124)
SHADOW = (38, 44, 56)


def proj(X, Z):
    return CX + F * X / Z, Y_H + F * CAM_H / Z


def S(pt):
    return (pt[0] * SS, pt[1] * SS)


# ---------------- primitive: tapered capsule ----------------
def capsule(draw, p0, r0, p1, r1, fill=255):
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    L = float(np.hypot(*d))
    for p, r in ((p0, r0), (p1, r1)):
        draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=fill)
    if L <= abs(r0 - r1) + 1e-6:
        return
    u = d / L
    n = np.array([-u[1], u[0]])
    sg = (r0 - r1) / L
    cg = math.sqrt(max(0.0, 1 - sg * sg))
    pts = [p0 + r0 * (sg * u + cg * n), p1 + r1 * (sg * u + cg * n),
           p1 + r1 * (sg * u - cg * n), p0 + r0 * (sg * u - cg * n)]
    draw.polygon([tuple(p) for p in pts], fill=fill)


# ---------------- figure ----------------
INNER_MARGIN = 3.0   # world px: contour line stops this far from the silhouette edge


def figure_layers(P, to_px, k, size, gap):
    """Return disjoint coverage masks (near, far) at supersampled res.

    to_px: world point -> canvas pixel (supersampled). k: px per world unit.
    gap: knockout width in world units (pictogram separation lines).
    """
    def new():
        img = Image.new("L", size, 0)
        return img, ImageDraw.Draw(img)

    def leg_shapes(d, L, grow=0.0, partial=False):
        g = grow
        hip, knee, ank, toe = (to_px(L[n]) for n in ("hip", "knee", "ankle", "toe"))
        # partial: the outline tapers to nothing at the hip so the thigh stays attached
        capsule(d, hip, (rig.R_HIP + (0 if partial else g)) * k, knee, (rig.R_KNEE + g) * k)
        capsule(d, knee, (rig.R_KNEE + g) * k, ank, (rig.R_ANKLE + g) * k)
        capsule(d, ank, (rig.R_ANKLE + g) * k, toe, (rig.R_TOE + g) * k)

    def arm_shapes(d, A, grow=0.0, partial=False):
        g = grow
        sh, el, ha = (to_px(A[n]) for n in ("shoulder", "elbow", "hand"))
        capsule(d, sh, (rig.R_SHOULDER - (1.5 if partial else -g)) * k, el, (rig.R_ELBOW + g) * k)
        capsule(d, el, (rig.R_ELBOW + g) * k, ha, (rig.R_HAND + g) * k)

    def body_shapes(d, grow=0.0):
        g = grow
        capsule(d, to_px(P["pelvis"]), (rig.R_PELVIS + g) * k,
                to_px(P["shoulder"]), (rig.R_CHEST + g) * k)
        h = to_px(P["head"])
        r = (rig.R_HEAD + g) * k
        d.ellipse([h[0] - r, h[1] - r, h[0] + r, h[1] + r], fill=255)

    def mask(fn, *a, **kw):
        img, d = new()
        fn(d, *a, **kw)
        return np.asarray(img) > 127

    FA = mask(arm_shapes, P["far_arm"])
    FL = mask(leg_shapes, P["far_leg"])
    B = mask(body_shapes)
    NL = mask(leg_shapes, P["near_leg"])
    NA = mask(arm_shapes, P["near_arm"])
    NAd = mask(arm_shapes, P["near_arm"], gap, partial=True)

    near = B | NL | NA
    far = (FA | FL) & ~near
    # pictogram contour of the near arm where it crosses the body: drawn only
    # inside the (eroded) silhouette, so the outer edge never gets notched
    inner = _morph(B | NL, max(1, int(round(INNER_MARGIN * k))), np.minimum)
    line = NAd & ~NA & inner
    return near & ~line, far, line


def _morph(m, r, op):
    """Binary erosion/dilation by an (approximate) disk of radius r px."""
    m = m.copy()
    for k in range(r):
        p = np.pad(m, 1, constant_values=False if op is np.maximum else True)
        c = p[1:-1, 1:-1]
        out = op(op(c, p[:-2, 1:-1]), p[2:, 1:-1])
        out = op(op(out, p[1:-1, :-2]), p[1:-1, 2:])
        if k % 2:   # alternate cross / square -> octagon
            out = op(op(op(out, p[:-2, :-2]), p[:-2, 2:]), op(p[2:, :-2], p[2:, 2:]))
        m = out
    return m


def downsample(m):
    h, w = m.shape
    return m.reshape(h // SS, SS, w // SS, SS).mean(axis=(1, 3))


# ---------------- background ----------------
def city(draw, rng, color, base_y, hmin, hmax, wmin, wmax):
    x = -20.0
    while x < W + 20:
        w = rng.uniform(wmin, wmax)
        h = rng.uniform(hmin, hmax)
        draw.rectangle(S((x, base_y - h)) + S((x + w, base_y + 2)), fill=color)
        x += w + rng.uniform(-4, 6)


def ground_quad(draw, X0, X1, Z0, Z1, fill):
    pts = [proj(X0, Z0), proj(X1, Z0), proj(X1, Z1), proj(X0, Z1)]
    draw.polygon([S(p) for p in pts], fill=fill)


def paste_icon(bg, cx, cy, scale, P, color, gap_world=1.6):
    """Draw a small pictogram pose centred at (cx, cy) (output px), height ~ 200*scale."""
    size = (int(220 * scale * SS) // SS * SS + SS * 4,) * 2
    ox, oy = math.floor(cx - size[0] / SS / 2), math.floor(cy - size[1] / SS / 2)
    pv = P["pelvis"]

    def to_px(w):
        sx = cx + (w[0] - pv[0]) * scale
        sy = cy - (w[1] - 100.0) * scale
        return ((sx - ox) * SS, (sy - oy) * SS)

    near, far, line = figure_layers(P, to_px, scale * SS, size, gap_world)
    cov = downsample(near | far | line)
    return cov, ox, oy


def build_background():
    big = Image.new("RGB", (W * SS, H * SS), SKY)
    d = ImageDraw.Draw(big)
    rng = np.random.default_rng(7)
    city(d, rng, CITY_FAR, Y_H, 26, 70, 26, 60)
    city(d, rng, CITY_NEAR, Y_H, 10, 38, 34, 80)
    # ground
    d.rectangle(S((0, Y_H)) + S((W, H)), fill=SIDEWALK)
    zmin, zmax = F * CAM_H / (H + 2 - Y_H), 4000.0
    # road + curbs
    ground_quad(d, -ROAD_HALF, ROAD_HALF, zmin, zmax, ROAD)
    for sgn in (-1, 1):
        ground_quad(d, sgn * ROAD_HALF, sgn * (ROAD_HALF + CURB), zmin, zmax, CURB_C)
    # zebra
    total = N_STRIPES * STRIPE_W + (N_STRIPES - 1) * STRIPE_GAP
    for i in range(N_STRIPES):
        x0 = -total / 2 + i * (STRIPE_W + STRIPE_GAP)
        ground_quad(d, x0, x0 + STRIPE_W, D - CROSS_HALF, D + CROSS_HALF, STRIPE)
    # centre line dashes
    z = D + CROSS_HALF + 1.2
    while z < 400:
        ground_quad(d, -0.07, 0.07, z, z + 2.4, DASH)
        z += 2.4 + 3.6
    ground_quad(d, -0.07, 0.07, zmin, D - CROSS_HALF - 1.0, DASH)
    big = big.resize((W, H), Image.LANCZOS)

    # ---- poles (drawn at SS on a separate layer) ----
    Zp = D + CROSS_HALF + 0.9
    k = F / Zp                                  # px per metre at the poles
    layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    icons = []
    # right: pedestrian traffic light
    bx, by = proj(ROAD_HALF + 0.45, Zp)
    pole_w = 0.085 * k
    top = by - 2.55 * k
    ld.rectangle(S((bx - pole_w / 2, top)) + S((bx + pole_w / 2, by)), fill=POLE + (255,))
    bw, bh = 0.34 * k, 0.70 * k
    box = (bx - bw / 2, top - 0.05 * k, bx + bw / 2, top - 0.05 * k + bh)
    ld.rounded_rectangle(S(box[:2]) + S(box[2:]), radius=0.05 * k * SS, fill=(40, 44, 52, 255))
    lr = 0.125 * k
    for j, col in enumerate([(70, 46, 50), (26, 44, 36)]):
        cy = box[1] + bh * (0.27 + 0.48 * j)
        ld.ellipse(S((bx - lr, cy - lr)) + S((bx + lr, cy + lr)), fill=col + (255,))
    # lit walking man in the lower lens + soft glow
    g_cy = box[1] + bh * 0.75
    glow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(glow).ellipse([bx - lr * 1.9, g_cy - lr * 1.9, bx + lr * 1.9, g_cy + lr * 1.9], fill=70)
    glow = glow.filter(ImageFilter.GaussianBlur(lr * 0.9))
    icons.append(("glow", glow))
    icons.append(("icon", paste_icon(None, bx, g_cy, lr * 1.55 / 200, rig.pose(0.0), None), (88, 232, 140)))
    sb_cy = box[1] + bh * 0.27
    icons.append(("icon", paste_icon(None, bx, sb_cy, lr * 1.6 / 200, standing_pose(), None), (120, 66, 68)))

    # left: pedestrian crossing sign 5.19
    sx, sy = proj(-(ROAD_HALF + 0.45), Zp)
    stop = sy - 2.45 * k
    ld.rectangle(S((sx - pole_w / 2, stop)) + S((sx + pole_w / 2, sy)), fill=POLE + (255,))
    sz = 0.62 * k
    sq = (sx - sz / 2, stop - sz * 0.2, sx + sz / 2, stop + sz * 0.8)
    ld.rounded_rectangle(S(sq[:2]) + S(sq[2:]), radius=0.06 * k * SS, fill=(250, 250, 250, 255))
    inset = sz * 0.05
    sq2 = (sq[0] + inset, sq[1] + inset, sq[2] - inset, sq[3] - inset)
    ld.rounded_rectangle(S(sq2[:2]) + S(sq2[2:]), radius=0.04 * k * SS, fill=(30, 96, 190, 255))
    tri_m = sz * 0.12
    tri = [((sq2[0] + sq2[2]) / 2, sq2[1] + tri_m), (sq2[2] - tri_m, sq2[3] - tri_m * 0.9),
           (sq2[0] + tri_m, sq2[3] - tri_m * 0.9)]
    ld.polygon([S(p) for p in tri], fill=(250, 250, 250, 255))
    tri_cx = (sq2[0] + sq2[2]) / 2
    base_y = sq2[3] - tri_m * 0.9
    # little zebra on the sign
    for j in range(4):
        x0 = tri_cx - sz * 0.22 + j * sz * 0.12
        ld.rectangle(S((x0, base_y - sz * 0.075)) + S((x0 + sz * 0.065, base_y - sz * 0.045)),
                     fill=(20, 20, 24, 255))
    icons.append(("icon", paste_icon(None, tri_cx, base_y - sz * 0.27, sz * 0.36 / 200,
                                     rig.pose(0.0), None), (20, 20, 24)))

    layer = layer.resize((W, H), Image.LANCZOS)
    arr = np.asarray(big).astype(np.float32)
    for item in icons:           # glow sits behind the light housing
        if item[0] == "glow":
            g = np.asarray(item[1]).astype(np.float32)[..., None] / 255.0
            arr = arr * (1 - g) + np.array([88, 232, 140], np.float32) * g
    big = Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8)).convert("RGBA")
    big.alpha_composite(layer)
    arr = np.asarray(big.convert("RGB")).astype(np.float32)
    for item in icons:
        if item[0] == "glow":
            continue
        else:
            (cov, ox, oy), col = item[1], item[2]
            h, w = cov.shape
            sub = arr[oy:oy + h, ox:ox + w]
            c = cov[..., None]
            arr[oy:oy + h, ox:ox + w] = sub * (1 - c) + np.array(col, np.float32) * c
    return arr


def standing_pose():
    P = rig.pose(0.0)
    pv = P["pelvis"].copy()
    pv[1] = 100.0
    sh = pv + np.array([0, rig.TORSO])
    head = sh + np.array([0, rig.R_CHEST + rig.NECK_GAP + rig.R_HEAD])

    def leg(dx):
        hip = pv + np.array([dx, 0])
        ank = np.array([hip[0], rig.R_ANKLE + 1.5])
        knee = (hip + ank) / 2
        toe = ank + np.array([rig.FOOT * math.cos(rig.A_FLAT), rig.FOOT * math.sin(rig.A_FLAT)])
        return dict(hip=hip, knee=knee, ankle=ank, toe=toe)

    def arm(dx):
        s0 = sh - np.array([dx, 3])
        el = s0 + np.array([0, -rig.UPPER_ARM])
        return dict(shoulder=s0, elbow=el, hand=el + np.array([0, -rig.FOREARM]))

    return dict(pelvis=pv, shoulder=sh, head=head, near_leg=leg(-8.5), far_leg=leg(8.5),
                near_arm=arm(-21), far_arm=arm(21))


# ---------------- per-frame ----------------
LOCAL_W, LOCAL_H = 300, 272
LOCAL_TOP = 196


def frame_pose(i):
    c = i / FRAMES_PER_CYCLE
    P = rig.pose(c)
    shift = X_START  # pose() puts pelvis x at c*STRIDE
    return c, P, shift


def shadow_layer(P, shift, ox, oy):
    m = Image.new("L", (LOCAL_W * SS, LOCAL_H * SS), 0)
    d = ImageDraw.Draw(m)

    def to(x, y):
        return ((x - ox) * SS, (y - oy) * SS)

    # soft body shadow
    ax = [P[n]["ankle"][0] + shift for n in ("near_leg", "far_leg")]
    cx = 0.5 * (P["pelvis"][0] + shift) + 0.25 * sum(ax)
    rx = 26 + abs(ax[0] - ax[1]) * 0.45
    x0, y0 = to(cx - rx, Y_FEET - 3.5)
    x1, y1 = to(cx + rx, Y_FEET + 4.5)
    d.ellipse([x0, y0, x1, y1], fill=70)
    m = m.reduce(SS).filter(ImageFilter.GaussianBlur(3.2))
    a = np.asarray(m).astype(np.float32) / 255.0
    # contact shadows under each foot, fade as the foot lifts
    for n in ("near_leg", "far_leg"):
        L = P[n]
        lift = min(L["ankle"][1], L["toe"][1] + 2.4) - rig.R_TOE
        k = max(0.0, 1.0 - lift / 14.0)
        if k <= 0:
            continue
        fx = 0.5 * (L["ankle"][0] + L["toe"][0]) + shift
        f = Image.new("L", (LOCAL_W * SS, LOCAL_H * SS), 0)
        fd = ImageDraw.Draw(f)
        rx = 15 + 4 * (1 - k)
        x0, y0 = to(fx - rx, Y_FEET - 2.2)
        x1, y1 = to(fx + rx, Y_FEET + 2.6)
        fd.ellipse([x0, y0, x1, y1], fill=int(95 * k))
        f = f.reduce(SS).filter(ImageFilter.GaussianBlur(1.6))
        a = 1 - (1 - a) * (1 - np.asarray(f).astype(np.float32) / 255.0)
    return a


def render_frame(bg, i):
    c, P, shift = frame_pose(i)
    px = P["pelvis"][0] + shift
    ox = math.floor(px) - LOCAL_W // 2
    oy = LOCAL_TOP
    out = bg.copy()
    # visible window of the local canvas
    x0, x1 = max(ox, 0), min(ox + LOCAL_W, W)
    if x1 <= x0:
        return out, None

    def to_px(w):
        return ((w[0] + shift - ox) * SS, (Y_FEET - w[1] - oy) * SS)

    near, far, line = figure_layers(P, to_px, SS, (LOCAL_W * SS, LOCAL_H * SS), 2.4)
    cn = downsample(near)[..., None]
    cf = downsample(far)[..., None]
    cl = downsample(line)[..., None]
    sh = shadow_layer(P, shift, ox, oy)[..., None] * 0.55
    region = out[oy:oy + LOCAL_H, x0:x1]
    sl = slice(x0 - ox, x1 - ox)
    region = region * (1 - sh[:, sl]) + np.array(SHADOW, np.float32) * sh[:, sl]
    region = region * (1 - cn[:, sl] - cf[:, sl] - cl[:, sl]) + np.array(NEAR, np.float32) * cn[:, sl] \
        + np.array(FAR, np.float32) * cf[:, sl] + np.array(LINE, np.float32) * cl[:, sl]
    out[oy:oy + LOCAL_H, x0:x1] = region
    bbox = None
    cov = (cn[..., 0] + cf[..., 0] + cl[..., 0]) > 0.01
    ys, xs = np.nonzero(cov)
    if len(xs):
        bbox = (xs.min() + ox, ys.min() + oy, xs.max() + ox, ys.max() + oy)
    return out, bbox


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "frames"
    only = [int(a) for a in sys.argv[2:]]
    os.makedirs(outdir, exist_ok=True)
    bg = build_background()
    Image.fromarray(np.clip(bg + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(outdir, "bg.png"))
    frames = only or range(N_FRAMES)
    with open(os.path.join(outdir, "bbox.txt"), "w") as fh:
        for i in frames:
            img, bbox = render_frame(bg, i)
            Image.fromarray(np.clip(img + 0.5, 0, 255).astype(np.uint8)).save(
                os.path.join(outdir, "f%04d.png" % i))
            fh.write("%d %s\n" % (i, bbox))
    print("frames:", len(list(frames)), "per cycle:", FRAMES_PER_CYCLE, "fps:", FPS)


if __name__ == "__main__":
    main()
