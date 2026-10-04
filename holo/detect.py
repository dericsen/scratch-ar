"""The marker detection pipeline.

Stages, in order:

1. **Grayscale** -- luma only; marker ink is achromatic so colour adds nothing.
2. **Adaptive threshold** -- local mean minus a bias. Survives the vignette and
   brightness ramp that a global threshold cannot.
3. **Morphological close** -- repairs border rings nicked by blur or noise.
4. **Connected components + boundary tracing** -- candidate dark blobs.
5. **Polygon approximation** -- Douglas-Peucker; keep convex 4-gons of
   plausible size and corner angles.
6. **Subpixel corner refinement** -- fit a line to each side, intersect them.
7. **Perspective unwarp + bit sampling** -- homography to a canonical square,
   Otsu threshold inside the patch, majority vote per cell.
8. **Border check + dictionary match** -- reject non-markers, recover ID and
   orientation.
9. **Pose** -- DLT homography, analytic decomposition, LM refinement.

Each stage can hand back its intermediate image for the pipeline figure.
"""

import math

from . import filters, geom, morph, pose
from . import marker as mk
from .contours import find_contours


class Detection:
    __slots__ = (
        "marker_id",
        "rotation",
        "bit_errors",
        "corners",
        "raw_corners",
        "r",
        "t",
        "rmse_initial",
        "rmse",
        "distance",
        "grid",
        "patch",
        "area",
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __repr__(self):
        return "Detection(id=%s, rot=%d, bit_errors=%d, dist=%.3fm, rmse=%.3fpx)" % (
            self.marker_id,
            self.rotation or 0,
            self.bit_errors or 0,
            self.distance or 0.0,
            self.rmse or 0.0,
        )


class Stages:
    """Intermediate images kept for visualisation."""

    def __init__(self):
        self.gray = None
        self.binary = None
        self.closed = None
        self.candidates = []  # quads that passed the shape filter
        self.rejected = []  # quads that failed decoding
        self.patches = []  # unwarped marker patches


# --------------------------------------------------------------- bit sampling


def unwarp_patch(gray, corners, cell_px=10):
    """Rectify the quad to a canonical ``GRID*cell_px`` square.

    Inverse mapping (canonical -> image) with bilinear sampling, which avoids
    the holes a forward mapping would leave.
    """
    n = mk.GRID * cell_px
    # Canonical corner order matches the detected quad order:
    # top-left, top-right, bottom-right, bottom-left.
    dst = [(0.0, 0.0), (float(n), 0.0), (float(n), float(n)), (0.0, float(n))]
    from . import homography

    h = homography.dlt(dst, [(float(c[0]), float(c[1])) for c in corners])
    if h is None:
        return None
    out = bytearray(n * n)
    gw, gh = gray.w, gray.h
    gpx = gray.px
    h00, h01, h02 = h[0]
    h10, h11, h12 = h[1]
    h20, h21, h22 = h[2]
    for yy in range(n):
        py = yy + 0.5
        base = yy * n
        for xx in range(n):
            px = xx + 0.5
            w = h20 * px + h21 * py + h22
            if abs(w) < 1e-12:
                continue
            sx = (h00 * px + h01 * py + h02) / w
            sy = (h10 * px + h11 * py + h12) / w
            x0 = int(sx)
            y0 = int(sy)
            if x0 < 0 or y0 < 0 or x0 >= gw - 1 or y0 >= gh - 1:
                out[base + xx] = 255
                continue
            fx = sx - x0
            fy = sy - y0
            i = y0 * gw + x0
            v = (
                gpx[i] * (1 - fx) * (1 - fy)
                + gpx[i + 1] * fx * (1 - fy)
                + gpx[i + gw] * (1 - fx) * fy
                + gpx[i + gw + 1] * fx * fy
            )
            out[base + xx] = int(v)
    from .image import Gray

    return Gray(n, n, px=out)


def sample_bits(patch, cell_px=10, margin=0.3):
    """Majority-vote each cell of the rectified patch into a bit.

    ``margin`` shrinks the sampling window inside each cell so that bleed from
    neighbouring cells (blur, slight corner error) does not flip a bit. The
    threshold comes from Otsu *on the patch itself* -- the patch contains both
    black and white ink by construction, so its own histogram is the most
    reliable reference available.
    """
    thresh = filters.otsu_threshold_value(patch)
    lo = int(cell_px * margin)
    hi = int(cell_px * (1.0 - margin))
    if hi <= lo:
        lo, hi = cell_px // 2, cell_px // 2 + 1
    grid = []
    px = patch.px
    n = patch.w
    for r in range(mk.GRID):
        row = []
        for c in range(mk.GRID):
            total = 0
            count = 0
            for yy in range(r * cell_px + lo, r * cell_px + hi):
                base = yy * n
                for xx in range(c * cell_px + lo, c * cell_px + hi):
                    total += px[base + xx]
                    count += 1
            row.append(1 if (total / count) >= thresh else 0)
        grid.append(row)
    return grid, thresh


def border_violations(full):
    """Count border cells that are not black. The border ring is the marker's
    signature; a random dark blob almost never has a clean one."""
    n = len(full)
    bad = 0
    for i in range(n):
        bad += full[0][i] != 0
        bad += full[n - 1][i] != 0
    for i in range(1, n - 1):
        bad += full[i][0] != 0
        bad += full[i][n - 1] != 0
    return bad


def payload_of(full):
    n = len(full)
    return [row[1 : n - 1] for row in full[1 : n - 1]]


# ------------------------------------------------------------------- pipeline


def detect(
    img,
    cam,
    dictionary,
    marker_side=0.10,
    block=25,
    bias=7,
    min_area=220,
    close_radius=1,
    max_border_bad=2,
    cell_px=10,
    refine_corners=True,
    collect_stages=False,
):
    """Find and localise every marker in ``img``.

    Returns ``(detections, stages)``; ``stages`` is ``None`` unless
    ``collect_stages`` is set.
    """
    st = Stages() if collect_stages else None
    gray = img.to_gray()
    binary = filters.adaptive_threshold(gray, block=block, c=bias, invert=True)
    closed = morph.close(binary, close_radius) if close_radius else binary
    if st:
        st.gray = gray
        st.binary = binary
        st.closed = closed

    contours, _labels = find_contours(closed, min_area=min_area)
    detections = []
    for comp, chain in contours:
        peri = geom.perimeter(chain)
        # RDP tolerance proportional to the contour size: a fixed pixel value
        # either over-simplifies big markers or under-simplifies small ones.
        eps = max(1.5, 0.035 * peri / 4.0)
        poly = geom.approx_polygon_closed(chain, eps)
        if len(poly) != 4:
            continue
        quad = geom.order_quad(poly)
        if not geom.is_convex(quad):
            continue
        if abs(geom.signed_area(quad)) < min_area * 0.5:
            continue
        if geom.shortest_side(quad) < 10.0:
            continue
        # Reject slivers and folded quads: a real marker seen under perspective
        # still has corners well away from 180 degrees.
        if geom.min_corner_angle_cos(quad) > 0.93:
            continue
        if st:
            st.candidates.append(list(quad))

        raw = [(float(p[0]), float(p[1])) for p in quad]
        if refine_corners:
            # Two-stage refinement: contour line fits remove the per-pixel
            # staircase, then grayscale gradient fits remove the threshold's
            # systematic shrink bias.
            corners = geom.refine_quad_corners(chain, quad)
            corners = geom.refine_quad_subpixel(gray, geom.order_quad(corners))
        else:
            corners = raw
        corners = geom.order_quad(corners)

        patch = unwarp_patch(gray, corners, cell_px)
        if patch is None:
            continue
        full, _thresh = sample_bits(patch, cell_px)
        if border_violations(full) > max_border_bad:
            if st:
                st.rejected.append(list(corners))
            continue
        decoded = dictionary.decode(payload_of(full))
        if decoded is None:
            if st:
                st.rejected.append(list(corners))
            continue
        marker_id, rot, errs = decoded
        if st:
            st.patches.append((marker_id, patch))

        # Re-index the corners into the dictionary's own orientation. The
        # sampled grid equals the dictionary grid rotated CCW by ``rot``, so
        # canonical corner i corresponds to dictionary corner (i + rot) mod 4.
        aligned = [corners[(j - rot) % 4] for j in range(4)]

        s = marker_side / 2.0
        obj = [(-s, s, 0.0), (s, s, 0.0), (s, -s, 0.0), (-s, -s, 0.0)]
        solved = pose.solve_marker_pose(cam, obj, aligned)
        if solved is None:
            continue
        r, t, rmse0, rmse1 = solved
        detections.append(
            Detection(
                marker_id=marker_id,
                rotation=rot,
                bit_errors=errs,
                corners=aligned,
                raw_corners=raw,
                r=r,
                t=t,
                rmse_initial=rmse0,
                rmse=rmse1,
                distance=math.sqrt(t[0] ** 2 + t[1] ** 2 + t[2] ** 2),
                grid=payload_of(full),
                patch=patch,
                area=comp.area,
            )
        )
    detections.sort(key=lambda d: d.distance)
    return detections, st
