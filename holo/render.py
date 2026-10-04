"""3-D overlay rendering -- the "augmented" half of augmented reality.

Everything here is drawn in the *marker's* coordinate frame and pushed through
the estimated pose, which is the honest test of the whole pipeline: if the pose
is wrong by a degree the cube visibly slides off the paper.

Hidden surfaces are handled with the painter's algorithm (sort faces by camera
depth, draw far to near). For convex meshes that is exactly correct, and a
z-buffer would be wasted effort.
"""

import math

from . import linalg as la
from . import pose as pose_mod

# Palette
CYAN = (0, 229, 255)
MAGENTA = (255, 64, 180)
LIME = (168, 255, 70)
AMBER = (255, 196, 40)
WHITE = (245, 248, 255)
RED = (255, 72, 66)
GREEN = (80, 230, 110)
BLUE = (90, 150, 255)
PANEL = (12, 14, 22)


# ------------------------------------------------------------------- meshes


def cube_mesh(size, lift=0.0):
    """Unit-ish cube resting on the marker plane, extending along +Z (outward)."""
    s = size / 2.0
    z0, z1 = lift, lift + size
    verts = [
        (-s, -s, z0), (s, -s, z0), (s, s, z0), (-s, s, z0),
        (-s, -s, z1), (s, -s, z1), (s, s, z1), (-s, s, z1),
    ]
    faces = [
        (0, 1, 2, 3),  # bottom (on the marker)
        (4, 5, 6, 7),  # top
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    colors = [
        (40, 70, 110),
        (0, 190, 220),
        (230, 70, 150),
        (255, 170, 50),
        (120, 220, 110),
        (150, 110, 230),
    ]
    return verts, faces, colors


def octahedron_mesh(radius, center_z):
    """Diamond floating above the marker; spun per frame to show that the
    anchoring is genuinely 3-D and not a 2-D sticker."""
    r = radius
    verts = [
        (0.0, 0.0, center_z + r),
        (r, 0.0, center_z),
        (0.0, r, center_z),
        (-r, 0.0, center_z),
        (0.0, -r, center_z),
        (0.0, 0.0, center_z - r),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1),
        (5, 2, 1), (5, 3, 2), (5, 4, 3), (5, 1, 4),
    ]
    base = [(0, 220, 255), (255, 80, 170), (170, 255, 90), (255, 200, 60)]
    colors = [base[i % 4] for i in range(8)]
    return verts, faces, colors


def rotate_mesh(verts, r):
    return [tuple(la.matvec(r, list(v))) for v in verts]


# ------------------------------------------------------------------ drawing


def draw_mesh(img, cam, r, t, verts, faces, colors, alpha=0.55, wire=True,
              wire_color=None, wire_thickness=1):
    """Project and paint a mesh under the pose ``(r, t)``."""
    cam_pts = [pose_mod.transform(r, t, list(v)) for v in verts]
    proj = [cam.project_point(p) for p in cam_pts]
    if any(p is None for p in proj):
        return False  # some vertex is behind the camera; skip rather than smear
    order = sorted(
        range(len(faces)),
        key=lambda fi: -sum(cam_pts[i][2] for i in faces[fi]) / len(faces[fi]),
    )
    for fi in order:
        face = faces[fi]
        pts = [proj[i] for i in face]
        img.fill_poly(pts, colors[fi], alpha)
        if wire:
            img.draw_poly(pts, wire_color or colors[fi], wire_thickness, min(1.0, alpha + 0.35))
    return True


def draw_axes(img, cam, r, t, length, thickness=2, labels=True):
    """Draw the marker's coordinate frame: X red, Y green, Z blue."""
    origin = cam.project_point(pose_mod.transform(r, t, [0.0, 0.0, 0.0]))
    if origin is None:
        return
    ends = {
        "X": ((length, 0.0, 0.0), RED),
        "Y": ((0.0, length, 0.0), GREEN),
        "Z": ((0.0, 0.0, length), BLUE),
    }
    for name, (vec, col) in ends.items():
        p = cam.project_point(pose_mod.transform(r, t, list(vec)))
        if p is None:
            continue
        img.draw_line(origin[0], origin[1], p[0], p[1], col, thickness)
        if labels:
            img.draw_text(p[0] + 2, p[1] - 3, name, col, scale=1)


def draw_quad_outline(img, corners, color=CYAN, thickness=1, dot=2, label_corners=False):
    """Outline the detected marker and mark its corners.

    Corner 0 is drawn filled so the recovered *orientation* is visible, not just
    the location -- that is what the rotation part of the decode buys us.
    """
    img.draw_poly([(c[0], c[1]) for c in corners], color, thickness)
    for i, c in enumerate(corners):
        if i == 0:
            img.fill_circle(c[0], c[1], dot + 1, AMBER)
        else:
            img.fill_circle(c[0], c[1], dot, color)
        if label_corners:
            img.draw_text(c[0] + 4, c[1] + 4, str(i), color, scale=1)


def draw_marker_label(img, cam, r, t, text_lines, offset_z=0.0, color=WHITE):
    """Anchor a small text block just above the marker in 3-D."""
    anchor = cam.project_point(pose_mod.transform(r, t, [0.0, 0.0, offset_z]))
    if anchor is None:
        return
    x = int(anchor[0]) + 6
    y = int(anchor[1]) - 6 - 9 * len(text_lines)
    width = max(len(s) for s in text_lines) * 6 + 6
    img.fill_rect(x - 3, y - 3, x + width, y + 9 * len(text_lines), PANEL, 0.62)
    for i, line in enumerate(text_lines):
        img.draw_text(x, y + i * 9, line, color, scale=1)


def draw_hud(img, lines, x=8, y=8, scale=1, color=WHITE, title=None,
             title_color=CYAN, width=None):
    """Translucent info panel in the corner."""
    all_lines = ([title] if title else []) + lines
    if not all_lines:
        return
    lh = (7 * scale) + 3
    w = width or (max(len(s) for s in all_lines) * (6 * scale) + 10)
    img.fill_rect(x - 4, y - 4, x + w, y + lh * len(all_lines) + 1, PANEL, 0.66)
    cy = y
    if title:
        img.draw_text(x, cy, title, title_color, scale=scale)
        cy += lh
    for line in lines:
        img.draw_text(x, cy, line, color, scale=scale)
        cy += lh


def draw_progress_bar(img, frac, y=None, height=3, color=CYAN):
    y = (img.h - height - 2) if y is None else y
    img.fill_rect(0, y, img.w - 1, y + height, (30, 32, 44), 0.75)
    img.fill_rect(0, y, int((img.w - 1) * max(0.0, min(1.0, frac))), y + height, color, 0.95)


# ------------------------------------------------------- full AR composition


def augment(img, cam, detections, marker_side=0.10, phase=0.0, show_axes=True,
            show_cube=True, show_diamond=True, ground_truth=None, label=True):
    """Draw the complete AR overlay for every detection on a frame.

    ``ground_truth`` (optional, keyed by marker id) makes the figure
    self-documenting: the true marker outline is drawn as a thin white dashed
    quad so the estimate can be compared against it by eye.
    """
    for det in detections:
        if ground_truth is not None:
            g = ground_truth.get(det.marker_id)
            if g is not None:
                _draw_dashed_quad(img, g.corners, (255, 255, 255), 1, 0.55)

        draw_quad_outline(img, det.corners, CYAN, 1, 2)

        if show_cube:
            size = marker_side * 0.62
            verts, faces, colors = cube_mesh(size)
            draw_mesh(img, cam, det.r, det.t, verts, faces, colors,
                      alpha=0.5, wire=True, wire_color=WHITE)

        if show_diamond:
            # Spin about the marker's +Z and bob along it.
            spin = la.rodrigues([0.0, 0.0, phase])
            bob = 0.012 * math.sin(phase * 1.5)
            verts, faces, colors = octahedron_mesh(
                marker_side * 0.17, marker_side * 0.62 + marker_side * 0.3 + bob
            )
            verts = rotate_mesh(verts, spin)
            draw_mesh(img, cam, det.r, det.t, verts, faces, colors,
                      alpha=0.62, wire=True, wire_color=WHITE)

        if show_axes:
            draw_axes(img, cam, det.r, det.t, marker_side * 0.5, thickness=2)

        if label:
            lines = ["ID %02d  R%d" % (det.marker_id, det.rotation)]
            lines.append("%.0f MM  %.2F PX" % (det.distance * 1000.0, det.rmse))
            draw_marker_label(img, cam, det.r, det.t, lines,
                              offset_z=marker_side * 0.62 + marker_side * 0.62)
    return img


def _draw_dashed_quad(img, corners, color, thickness=1, alpha=1.0, dash=4):
    n = len(corners)
    for i in range(n):
        ax, ay = corners[i]
        bx, by = corners[(i + 1) % n]
        length = math.hypot(bx - ax, by - ay)
        if length < 1e-6:
            continue
        steps = max(1, int(length / dash))
        for s in range(steps):
            if s % 2:
                continue
            f0 = s / steps
            f1 = (s + 1) / steps
            img.draw_line(ax + (bx - ax) * f0, ay + (by - ay) * f0,
                          ax + (bx - ax) * f1, ay + (by - ay) * f1,
                          color, thickness, alpha)
