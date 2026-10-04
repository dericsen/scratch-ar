"""Synthetic scene renderer -- the source of ground truth.

Why bother rendering our own input? Because "the cube looks like it's stuck to
the marker" is not a measurement. By generating frames from a known camera and
known marker placements we get the exact pose the detector *should* recover, so
every number in the report (rotation error in degrees, translation error in
millimetres, corner error in pixels) is a real comparison rather than a vibe.

The renderer is a ray caster over axis-free planes: for each pixel we build the
view ray from the pinhole model, intersect it with the floor and with every
marker sheet, and shade the nearest hit. Planes only, so there is no need for a
z-buffer and perspective-correct texturing is automatic.

Then the frame is deliberately degraded -- lens blur, sensor noise, vignetting,
a brightness ramp -- because a detector that only works on perfect synthetic
pixels is not a detector.
"""

import math
import random

from . import linalg as la
from . import marker as mk
from .filters import blur_rgb
from .image import Image


class MarkerPlacement:
    """One printed marker sitting somewhere in the world.

    ``r_wm``/``t_wm`` is the marker-to-world transform. The marker's own frame
    has +X right across its face, +Y up across its face and +Z along the
    outward normal, matching the convention used by the pose solver.
    """

    __slots__ = ("marker_id", "grid", "r_wm", "t_wm", "side", "paper_half", "full")

    def __init__(self, marker_id, grid, r_wm, t_wm, side):
        self.marker_id = marker_id
        self.grid = grid
        self.r_wm = r_wm
        self.t_wm = t_wm
        self.side = side  # physical side length of the black 7x7 square
        # The white quiet zone adds one cell on every side.
        self.paper_half = 0.5 * side * (mk.GRID + 2) / mk.GRID
        self.full = mk.full_grid(grid)

    def axes(self):
        """World-space (x_axis, y_axis, normal) of the marker face."""
        r = self.r_wm
        return (
            [r[0][0], r[1][0], r[2][0]],
            [r[0][1], r[1][1], r[2][1]],
            [r[0][2], r[1][2], r[2][2]],
        )

    def object_corners(self):
        """Marker corners in its own frame, in dictionary order:
        top-left, top-right, bottom-right, bottom-left."""
        s = self.side / 2.0
        return [(-s, s, 0.0), (s, s, 0.0), (s, -s, 0.0), (-s, -s, 0.0)]

    def pose_in_camera(self, r_cw, t_cw):
        """Ground-truth marker-to-camera pose for a given camera extrinsic."""
        r_cm = la.matmul(r_cw, self.r_wm)
        t_cm = la.vadd(la.matvec(r_cw, self.t_wm), t_cw)
        return r_cm, t_cm

    def texel(self, a, b):
        """Shade lookup in marker-plane coordinates (metres from the centre).

        Returns a reflectance in [0, 1]: the printed cell value inside the black
        square, white paper in the quiet zone.
        """
        s = self.side / 2.0
        if -s <= a <= s and -s <= b <= s:
            col = int((a + s) / self.side * mk.GRID)
            row = int((s - b) / self.side * mk.GRID)  # row 0 is the top (+Y)
            if col >= mk.GRID:
                col = mk.GRID - 1
            if row >= mk.GRID:
                row = mk.GRID - 1
            return 0.93 if self.full[row][col] else 0.045
        return 0.93  # paper


class GroundTruth:
    __slots__ = ("marker_id", "r_cm", "t_cm", "corners", "distance")

    def __init__(self, marker_id, r_cm, t_cm, corners, distance):
        self.marker_id = marker_id
        self.r_cm = r_cm
        self.t_cm = t_cm
        self.corners = corners  # true projected image corners, dictionary order
        self.distance = distance


PAPER_LIFT = 0.0008  # 0.8 mm: real paper has thickness, and it also keeps the
# sheet from fighting the floor plane for the same ray at grazing angles


def marker_on_floor(marker_id, grid, x, z, yaw_deg, side, y=PAPER_LIFT):
    """Place a marker lying flat on the floor, rotated by ``yaw_deg``."""
    y_rot = math.radians(yaw_deg)
    cx, sx = math.cos(y_rot), math.sin(y_rot)
    x_axis = [cx, 0.0, sx]
    normal = [0.0, 1.0, 0.0]
    y_axis = la.cross(normal, x_axis)
    r_wm = [
        [x_axis[0], y_axis[0], normal[0]],
        [x_axis[1], y_axis[1], normal[1]],
        [x_axis[2], y_axis[2], normal[2]],
    ]
    return MarkerPlacement(marker_id, grid, r_wm, [x, y, z], side)


def marker_tilted(marker_id, grid, x, z, yaw_deg, tilt_deg, side, y=PAPER_LIFT):
    """Place a marker leaning back like a photo on an easel.

    Included on purpose: a strongly tilted marker is the case where the naive
    homography decomposition is weakest and the LM refinement earns its keep.
    """
    base = marker_on_floor(marker_id, grid, x, z, yaw_deg, side, y)
    # Rotate about the marker's own X axis so it tips up out of the floor.
    x_axis, y_axis, normal = base.axes()
    tilt = la.rodrigues(la.vscale(la.normalize(x_axis), math.radians(tilt_deg)))
    y_axis = la.matvec(tilt, y_axis)
    normal = la.matvec(tilt, normal)
    r_wm = [
        [x_axis[0], y_axis[0], normal[0]],
        [x_axis[1], y_axis[1], normal[1]],
        [x_axis[2], y_axis[2], normal[2]],
    ]
    # Lift the centre so the lower edge still rests on the floor. Note the
    # abs(): the sheet rises by the same amount whichever way it leans, and
    # without it a negative tilt sinks the marker under the floor, where the
    # floor plane occludes all but a thin sliver.
    lift = base.paper_half * abs(math.sin(math.radians(tilt_deg)))
    return MarkerPlacement(marker_id, grid, r_wm, [x, y + lift, z], base.side)


def default_scene(dictionary, side=0.10):
    """Three markers: two flat with different yaw, one tilted up on an easel.

    Spread out enough that they are never all at the same scale or tilt, so the
    accuracy numbers average over genuinely different viewing conditions.
    """
    return [
        marker_on_floor(0, dictionary.grids[0], -0.105, 0.045, 18.0, side),
        marker_on_floor(1, dictionary.grids[1], 0.115, -0.075, -27.0, side),
        # Negative tilt leans this one back towards the orbit's starting
        # viewpoint. It inevitably turns edge-on partway round the orbit, and
        # losing it there is correct behaviour, not a detector failure.
        marker_tilted(2, dictionary.grids[2], 0.015, 0.155, 8.0, -30.0, side),
    ]


# ------------------------------------------------------------------ shading

_LIGHT = la.normalize([-0.45, 1.0, -0.35])
_SKY_TOP = (26, 30, 48)
_SKY_BOT = (78, 86, 112)
_TILE_A = (196, 198, 206)
_TILE_B = (128, 132, 146)


def _background(dy):
    """Vertical gradient for rays that hit nothing."""
    t = max(0.0, min(1.0, dy * 1.6 + 0.5))
    return (
        int(_SKY_BOT[0] + (_SKY_TOP[0] - _SKY_BOT[0]) * t),
        int(_SKY_BOT[1] + (_SKY_TOP[1] - _SKY_BOT[1]) * t),
        int(_SKY_BOT[2] + (_SKY_TOP[2] - _SKY_BOT[2]) * t),
    )


def render_frame(
    cam,
    r_cw,
    t_cw,
    placements,
    supersample=2,
    blur=1,
    noise=3.0,
    vignette=0.35,
    light_ramp=0.18,
    seed=0,
    floor_tile=0.06,
):
    """Ray-cast one frame.

    ``r_cw``/``t_cw`` is the world-to-camera transform (``x_cam = r_cw x + t_cw``).
    Returns ``(Image, [GroundTruth, ...])``.
    """
    w, h = cam.width, cam.height
    ss = max(1, supersample)
    rw, rh = w * ss, h * ss

    # Camera centre in world coordinates, and the rotation taking camera-frame
    # directions back to world.
    r_wc = la.transpose(r_cw)
    cam_pos = la.vscale(la.matvec(r_wc, t_cw), -1.0)

    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    # Precompute the world-space ray basis so the inner loop is pure arithmetic:
    # d_world = r_wc @ ((x-cx)/fx, (y-cy)/fy, 1)
    ax, ay, az = r_wc[0][0], r_wc[1][0], r_wc[2][0]  # world dir per camera +x
    bx, by, bz = r_wc[0][1], r_wc[1][1], r_wc[2][1]  # per camera +y
    cx3, cy3, cz3 = r_wc[0][2], r_wc[1][2], r_wc[2][2]  # per camera +z

    # Flatten marker geometry for the hot loop.
    surfaces = []
    for p in placements:
        xa, ya, na = p.axes()
        surfaces.append((p, xa, ya, na, la.dot(na, p.t_wm), p.paper_half))

    ox, oy, oz = cam_pos
    px_buf = bytearray(rw * rh * 3)
    light = _LIGHT
    inv_tile = 1.0 / floor_tile

    for yy in range(rh):
        v = ((yy + 0.5) / ss - cy) / fy
        rowbase = yy * rw * 3
        for xx in range(rw):
            u = ((xx + 0.5) / ss - cx) / fx
            dx = ax * u + bx * v + cx3
            dy = ay * u + by * v + cy3
            dz = az * u + bz * v + cz3

            best_t = None
            best_rgb = None

            # --- marker sheets (tested before the floor: a sheet resting on the
            # floor is nearly coplanar with it, and the sheet must win that tie)
            for p, xa, ya, na, nd, half in surfaces:
                denom = na[0] * dx + na[1] * dy + na[2] * dz
                if abs(denom) < 1e-9:
                    continue
                t = (nd - (na[0] * ox + na[1] * oy + na[2] * oz)) / denom
                if t <= 1e-6 or (best_t is not None and t >= best_t):
                    continue
                hx = ox + dx * t - p.t_wm[0]
                hy = oy + dy * t - p.t_wm[1]
                hz = oz + dz * t - p.t_wm[2]
                a = xa[0] * hx + xa[1] * hy + xa[2] * hz
                if a < -half or a > half:
                    continue
                b = ya[0] * hx + ya[1] * hy + ya[2] * hz
                if b < -half or b > half:
                    continue
                refl = p.texel(a, b)
                ndl = na[0] * light[0] + na[1] * light[1] + na[2] * light[2]
                if ndl < 0.0:
                    ndl = -ndl  # the sheet is printed on both sides, so light it either way
                shade = 0.55 + 0.45 * ndl
                lum = 255.0 * refl * shade
                fog = min(1.0, t / 3.2) * 0.35
                bg = _background(0.0)
                best_t = t
                best_rgb = (
                    lum * (1 - fog) + bg[0] * fog,
                    lum * (1 - fog) + bg[1] * fog,
                    lum * (1 - fog) + bg[2] * fog,
                )

            # --- floor plane y = 0
            if dy < -1e-9:  # only rays pointing downwards can hit it
                t = -oy / dy
                if t > 1e-6 and (best_t is None or t < best_t):
                    hx = ox + dx * t
                    hz = oz + dz * t
                    if abs(hx) < 3.0 and abs(hz) < 3.0:
                        tile = (int(math.floor(hx * inv_tile)) + int(math.floor(hz * inv_tile))) & 1
                        base = _TILE_A if tile else _TILE_B
                        shade = 0.58 + 0.42 * light[1]  # floor normal is +Y
                        fog = min(1.0, t / 2.6)
                        bg = _background(0.0)
                        best_t = t
                        best_rgb = (
                            base[0] * shade * (1 - fog) + bg[0] * fog,
                            base[1] * shade * (1 - fog) + bg[1] * fog,
                            base[2] * shade * (1 - fog) + bg[2] * fog,
                        )

            if best_rgb is None:
                n = math.sqrt(dx * dx + dy * dy + dz * dz)
                best_rgb = _background(dy / n if n else 0.0)

            i = rowbase + xx * 3
            r, g, b = best_rgb
            px_buf[i] = 255 if r > 255 else (0 if r < 0 else int(r))
            px_buf[i + 1] = 255 if g > 255 else (0 if g < 0 else int(g))
            px_buf[i + 2] = 255 if b > 255 else (0 if b < 0 else int(b))

    img = Image(rw, rh, px=px_buf)
    if ss > 1:
        img = _downsample(img, ss)

    # --- sensor imperfections (order matters: optics, then shading, then noise)
    if blur:
        img = blur_rgb(img, blur)
    if vignette > 0.0 or light_ramp > 0.0:
        _apply_shading_artifacts(img, vignette, light_ramp)
    if noise > 0.0:
        _apply_noise(img, noise, seed)

    # --- ground truth
    gts = []
    for p in placements:
        r_cm, t_cm = p.pose_in_camera(r_cw, t_cw)
        corners = []
        ok = True
        for q in p.object_corners():
            pc = la.vadd(la.matvec(r_cm, list(q)), t_cm)
            pr = cam.project_point(pc)
            if pr is None:
                ok = False
                break
            corners.append(pr)
        if ok:
            gts.append(GroundTruth(p.marker_id, r_cm, t_cm, corners, la.norm(t_cm)))
    return img, gts


def _downsample(img, ss):
    """Box-average an ``ss``x supersampled buffer down to final resolution.

    This is the anti-aliasing: without it marker edges land on hard pixel
    boundaries and corner refinement has nothing subpixel to work with.
    """
    w, h = img.w // ss, img.h // ss
    out = bytearray(w * h * 3)
    src = img.px
    sw = img.w
    n = ss * ss
    for y in range(h):
        for x in range(w):
            r = g = b = 0
            for oy in range(ss):
                base = ((y * ss + oy) * sw + x * ss) * 3
                for ox in range(ss):
                    i = base + ox * 3
                    r += src[i]
                    g += src[i + 1]
                    b += src[i + 2]
            o = (y * w + x) * 3
            out[o] = r // n
            out[o + 1] = g // n
            out[o + 2] = b // n
    return Image(w, h, px=out)


def _apply_shading_artifacts(img, vignette, light_ramp):
    """Radial falloff plus a horizontal brightness ramp.

    This is what breaks a global threshold and forces the detector to be
    locally adaptive.
    """
    w, h = img.w, img.h
    px = img.px
    cx, cy = w / 2.0, h / 2.0
    max_r2 = cx * cx + cy * cy
    for y in range(h):
        dy = y - cy
        for x in range(w):
            dx = x - cx
            r2 = (dx * dx + dy * dy) / max_r2
            gain = (1.0 - vignette * r2) * (1.0 - light_ramp * (x / w))
            i = (y * w + x) * 3
            v = int(px[i] * gain)
            px[i] = 255 if v > 255 else v
            v = int(px[i + 1] * gain)
            px[i + 1] = 255 if v > 255 else v
            v = int(px[i + 2] * gain)
            px[i + 2] = 255 if v > 255 else v


def _apply_noise(img, sigma, seed):
    rng = random.Random(seed)
    px = img.px
    gauss = rng.gauss
    for i in range(len(px)):
        v = px[i] + int(gauss(0.0, sigma))
        px[i] = 255 if v > 255 else (0 if v < 0 else v)


# ------------------------------------------------------------- trajectories


def orbit_trajectory(n_frames, radius=0.40, height=0.42, target=(0.0, 0.0, 0.0),
                     start_deg=-120.0, sweep_deg=360.0, bob=0.07):
    """Camera poses orbiting the scene, with a little vertical bob.

    Returns a list of ``(r_cw, t_cw)``. The bob keeps the markers from being
    viewed at a constant tilt, which would make the demo easier than it should be.
    """
    out = []
    for i in range(n_frames):
        frac = i / n_frames
        ang = math.radians(start_deg + sweep_deg * frac)
        h = height + bob * math.sin(2.0 * math.pi * frac * 2.0)
        eye = [radius * math.cos(ang), h, radius * math.sin(ang)]
        out.append(la.look_at(eye, list(target)))
    return out
