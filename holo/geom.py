"""2-D geometry: polygon approximation, quad validation, and subpixel corner
refinement by edge-line intersection.

Corner accuracy is what ultimately limits pose accuracy, so the refinement in
:func:`refine_quad_corners` matters more than it looks: instead of trusting a
single boundary pixel per corner, it fits a total-least-squares line to each of
the four sides and intersects neighbouring lines. Every pixel along a side
contributes, which averages the pixel quantisation away.
"""

import math

from . import linalg as la


# ------------------------------------------------------- polygon approximation


def douglas_peucker(points, eps):
    """Ramer-Douglas-Peucker simplification of an *open* polyline."""
    if len(points) < 3:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        x0, y0 = points[i0]
        x1, y1 = points[i1]
        dx, dy = x1 - x0, y1 - y0
        seg = math.hypot(dx, dy)
        best_d, best_i = -1.0, -1
        if seg < 1e-12:
            for i in range(i0 + 1, i1):
                d = math.hypot(points[i][0] - x0, points[i][1] - y0)
                if d > best_d:
                    best_d, best_i = d, i
        else:
            for i in range(i0 + 1, i1):
                px, py = points[i]
                # Perpendicular distance from the point to the chord.
                d = abs(dy * (px - x0) - dx * (py - y0)) / seg
                if d > best_d:
                    best_d, best_i = d, i
        if best_d > eps:
            keep[best_i] = True
            stack.append((i0, best_i))
            stack.append((best_i, i1))
    return [p for p, k in zip(points, keep) if k]


def approx_polygon_closed(chain, eps, min_sep=3.0, collinear_cos=0.985):
    """RDP on a *closed* contour.

    A closed chain has no endpoints, so RDP needs two anchors. Both are chosen
    to be genuine extremities: the point furthest from the centroid (always a
    corner) and then the point furthest from *that* (for a quadrilateral, the
    opposite corner). Splitting at an arbitrary index instead -- say the
    antipode in chain order -- pins a vertex in the middle of a straight side
    and yields a spurious 5th corner a few pixels from a real one.

    The result is then cleaned: vertices closer than ``min_sep`` are merged and
    vertices that are nearly straight-through are dropped.
    """
    n = len(chain)
    if n < 4:
        return list(chain)
    cx = sum(p[0] for p in chain) / n
    cy = sum(p[1] for p in chain) / n
    i0 = max(range(n), key=lambda i: (chain[i][0] - cx) ** 2 + (chain[i][1] - cy) ** 2)
    p0 = chain[i0]
    i1 = max(range(n), key=lambda i: (chain[i][0] - p0[0]) ** 2 + (chain[i][1] - p0[1]) ** 2)
    if i1 == i0:
        i1 = (i0 + n // 2) % n
    # Two arcs between the anchors, each a proper open polyline.
    if i0 < i1:
        arc_a = chain[i0 : i1 + 1]
        arc_b = chain[i1:] + chain[: i0 + 1]
    else:
        arc_a = chain[i0:] + chain[: i1 + 1]
        arc_b = chain[i1 : i0 + 1]
    a = douglas_peucker(arc_a, eps)
    b = douglas_peucker(arc_b, eps)
    pts = a[:-1] + b[:-1]  # drop the duplicated anchors
    return cleanup_polygon(pts, min_sep, collinear_cos)


def cleanup_polygon(pts, min_sep=3.0, collinear_cos=0.985):
    """Merge near-duplicate vertices and drop near-collinear ones."""
    # 1. merge vertices that sit almost on top of each other
    out = []
    for p in pts:
        if out and math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) < min_sep:
            continue
        out.append(p)
    while (
        len(out) > 3
        and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) < min_sep
    ):
        out.pop()
    # 2. drop vertices where the boundary barely turns -- they are quantisation
    #    artefacts on a straight side, not corners
    changed = True
    while changed and len(out) > 3:
        changed = False
        for i in range(len(out)):
            a = out[(i - 1) % len(out)]
            b = out[i]
            c = out[(i + 1) % len(out)]
            u = (a[0] - b[0], a[1] - b[1])
            v = (c[0] - b[0], c[1] - b[1])
            nu = math.hypot(*u)
            nv = math.hypot(*v)
            if nu < 1e-9 or nv < 1e-9:
                out.pop(i)
                changed = True
                break
            cosv = (u[0] * v[0] + u[1] * v[1]) / (nu * nv)
            if cosv < -collinear_cos:  # interior angle close to 180 degrees
                out.pop(i)
                changed = True
                break
    return out


# ------------------------------------------------------------ polygon measures


def signed_area(pts):
    """Shoelace area.

    Note the sign convention: image coordinates have y pointing *down*, which
    flips orientation relative to a textbook plot. A positive shoelace value
    therefore means the vertices run **clockwise as seen on screen**.
    """
    s = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return 0.5 * s


def perimeter(pts, closed=True):
    n = len(pts)
    last = n if closed else n - 1
    return sum(
        math.hypot(pts[(i + 1) % n][0] - pts[i][0], pts[(i + 1) % n][1] - pts[i][1])
        for i in range(last)
    )


def is_convex(pts):
    n = len(pts)
    sign = 0
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        cx, cy = pts[(i + 2) % n]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def min_corner_angle_cos(pts):
    """Largest ``|cos(angle)|`` over the polygon's corners.

    Near 1 means some corner is nearly straight (180 degrees) or folded, i.e.
    the shape is a degenerate quad rather than a marker outline.
    """
    worst = 0.0
    n = len(pts)
    for i in range(n):
        a = pts[(i - 1) % n]
        b = pts[i]
        c = pts[(i + 1) % n]
        u = (a[0] - b[0], a[1] - b[1])
        v = (c[0] - b[0], c[1] - b[1])
        nu = math.hypot(*u)
        nv = math.hypot(*v)
        if nu < 1e-9 or nv < 1e-9:
            return 1.0
        cosv = abs((u[0] * v[0] + u[1] * v[1]) / (nu * nv))
        worst = max(worst, cosv)
    return worst


def shortest_side(pts):
    n = len(pts)
    return min(
        math.hypot(pts[(i + 1) % n][0] - pts[i][0], pts[(i + 1) % n][1] - pts[i][1])
        for i in range(n)
    )


def order_quad(pts):
    """Force a quad into positive-shoelace winding.

    Consistent winding is essential, not cosmetic: the decoder maps corner
    ``i`` of this list onto corner ``i`` of the canonical (un-warped) square.
    If half the detections arrived mirrored, their bit grids would come out
    transposed and decode to the wrong ID -- or to a valid but wrong marker.
    Positive shoelace is the same handedness as the canonical corner order
    (top-left, top-right, bottom-right, bottom-left), so no mirroring occurs.
    """
    return list(pts) if signed_area(pts) > 0 else list(reversed(pts))


# ----------------------------------------------------------------- line fitting


def fit_line(points):
    """Total-least-squares line fit.

    Returns ``(nx, ny, c)`` with ``nx*x + ny*y + c = 0`` and a unit normal. The
    normal is the eigenvector of the point scatter matrix with the smaller
    eigenvalue, i.e. the direction of least variance.
    """
    n = len(points)
    if n < 2:
        return None
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = syy = sxy = 0.0
    for px, py in points:
        dx = px - mx
        dy = py - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    vals, vecs = la.jacobi_eigen([[sxx, sxy], [sxy, syy]])
    nx, ny = vecs[0]  # smallest eigenvalue -> normal direction
    norm = math.hypot(nx, ny)
    if norm < 1e-12:
        return None
    nx /= norm
    ny /= norm
    return nx, ny, -(nx * mx + ny * my)


def line_intersect(l1, l2):
    """Intersection of two lines in ``(nx, ny, c)`` form, or ``None`` if they
    are (near) parallel."""
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    return ((b1 * c2 - b2 * c1) / det, (a2 * c1 - a1 * c2) / det)


def _chain_index_of(chain, pt):
    best, bi = None, 0
    for i, p in enumerate(chain):
        d = (p[0] - pt[0]) ** 2 + (p[1] - pt[1]) ** 2
        if best is None or d < best:
            best, bi = d, i
    return bi


def refine_quad_corners(chain, quad, trim=0.2, min_pts=5):
    """Subpixel corner refinement via per-side line fits.

    ``chain`` is the traced boundary, ``quad`` the four approximate corners
    (which must be vertices of that chain). For each side we take the boundary
    pixels strictly between two corners, drop a ``trim`` fraction at each end
    (those pixels are rounded by the corner itself), fit a line, and intersect
    consecutive lines to get the refined corner.

    Falls back to the input corner whenever a side has too few usable pixels.
    """
    n = len(chain)
    idx = [_chain_index_of(chain, q) for q in quad]
    lines = []
    for k in range(4):
        i0 = idx[k]
        i1 = idx[(k + 1) % 4]
        # Collect the chain pixels from corner k to corner k+1 (chain is cyclic).
        seg = []
        i = i0
        guard = 0
        while True:
            seg.append(chain[i])
            if i == i1:
                break
            i = (i + 1) % n
            guard += 1
            if guard > n:
                break
        cut = int(len(seg) * trim)
        inner = seg[cut : len(seg) - cut] if len(seg) - 2 * cut >= min_pts else seg
        lines.append(fit_line(inner) if len(inner) >= 2 else None)
    refined = []
    for k in range(4):
        l_prev = lines[(k - 1) % 4]
        l_cur = lines[k]
        p = line_intersect(l_prev, l_cur) if (l_prev and l_cur) else None
        if p is None:
            refined.append((float(quad[k][0]), float(quad[k][1])))
            continue
        # Reject a wild intersection (near-parallel sides) and keep the raw corner.
        if math.hypot(p[0] - quad[k][0], p[1] - quad[k][1]) > 6.0:
            refined.append((float(quad[k][0]), float(quad[k][1])))
        else:
            refined.append(p)
    return refined


# ------------------------------------------------- subpixel edge refinement


def _edge_crossing(gray, px, py, nx, ny, reach, step=0.4):
    """Locate the intensity edge nearest ``(px, py)`` along direction ``(nx, ny)``.

    Walks an intensity profile across the boundary, takes central differences,
    and fits a parabola to the three samples around the strongest gradient to
    get a subpixel offset. Returns the offset along the direction, or ``None``
    if the profile has no real edge in it.
    """
    n = int(reach / step)
    prof = []
    for k in range(-n, n + 1):
        s = k * step
        prof.append(gray.sample(px + nx * s, py + ny * s))
    if len(prof) < 5:
        return None
    grads = [abs(prof[i + 1] - prof[i - 1]) for i in range(1, len(prof) - 1)]
    best = max(range(len(grads)), key=lambda i: grads[i])
    # A flat profile means we are not looking at an edge at all.
    if grads[best] < 8.0:
        return None
    if best == 0 or best == len(grads) - 1:
        return (best + 1 - n) * step
    g0, g1, g2 = grads[best - 1], grads[best], grads[best + 1]
    denom = g0 - 2.0 * g1 + g2
    delta = 0.5 * (g0 - g2) / denom if abs(denom) > 1e-9 else 0.0
    if delta > 1.0:
        delta = 1.0
    elif delta < -1.0:
        delta = -1.0
    return (best + 1 + delta - n) * step


def refine_quad_subpixel(gray, quad, samples=14, trim=0.18, max_shift=4.0):
    """Refine quad corners against the *grayscale* image.

    Motivation: corners taken from a binarised contour inherit the threshold's
    bias. A mean-adaptive threshold sits slightly below the true 50% intensity
    crossing, so every black square comes out a fraction of a pixel too small
    in every direction -- which reads as the marker being systematically
    further away than it is. Measuring the edge where the *intensity gradient*
    peaks is bias-free, because the gradient peak of a blurred step edge is the
    edge, independent of any threshold.

    For each side we sample points along it, find the edge crossing along the
    side's normal, fit a total-least-squares line to those subpixel points, and
    intersect neighbouring lines. Returns refined corners, or the input quad if
    the evidence is too weak.
    """
    lines = []
    for k in range(4):
        ax, ay = quad[k]
        bx, by = quad[(k + 1) % 4]
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length < 8.0:
            return [(float(p[0]), float(p[1])) for p in quad]
        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux  # unit normal to this side
        # Limit the search so we cannot latch onto the next cell boundary: the
        # border ring is one cell wide, so stay well inside half a cell.
        reach = max(1.5, min(3.0, length / 7.0 * 0.4))
        pts = []
        for i in range(samples):
            f = trim + (1.0 - 2.0 * trim) * (i / (samples - 1.0))
            sx = ax + dx * f
            sy = ay + dy * f
            off = _edge_crossing(gray, sx, sy, nx, ny, reach)
            if off is None:
                continue
            pts.append((sx + nx * off, sy + ny * off))
        if len(pts) < 4:
            return [(float(p[0]), float(p[1])) for p in quad]
        lines.append(fit_line(pts))
    refined = []
    for k in range(4):
        l_prev = lines[(k - 1) % 4]
        l_cur = lines[k]
        p = line_intersect(l_prev, l_cur) if (l_prev and l_cur) else None
        if p is None or math.hypot(p[0] - quad[k][0], p[1] - quad[k][1]) > max_shift:
            refined.append((float(quad[k][0]), float(quad[k][1])))
        else:
            refined.append(p)
    return refined
