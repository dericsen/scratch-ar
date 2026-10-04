"""Planar homography estimation (normalised DLT).

``H`` maps source to destination in homogeneous coordinates::

    [u']       [x]
    [v']  =  H [y]        u = u'/w'   v = v'/w'
    [w']       [1]

Hartley normalisation is not optional cosmetics here. Raw pixel coordinates in
the hundreds make the 9x9 scatter matrix badly conditioned and the recovered
homography visibly wrong; translating to the centroid and scaling to a mean
distance of sqrt(2) fixes it.
"""

import math

from . import linalg as la


def _normalising_transform(pts):
    n = len(pts)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    mean_d = sum(math.hypot(p[0] - cx, p[1] - cy) for p in pts) / n
    s = math.sqrt(2.0) / mean_d if mean_d > 1e-12 else 1.0
    t = [[s, 0.0, -s * cx], [0.0, s, -s * cy], [0.0, 0.0, 1.0]]
    out = [((p[0] - cx) * s, (p[1] - cy) * s) for p in pts]
    return t, out


def dlt(src, dst):
    """Direct Linear Transform from >= 4 point correspondences.

    Returns the 3x3 homography normalised so that ``H[2][2] == 1``, or ``None``
    for a degenerate configuration.
    """
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("need at least 4 matched points")
    ts, nsrc = _normalising_transform(src)
    td, ndst = _normalising_transform(dst)
    a = []
    for (x, y), (u, v) in zip(nsrc, ndst):
        a.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        a.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    hv = la.null_vector(a)
    hn = [hv[0:3], hv[3:6], hv[6:9]]
    # Undo the normalisation: H = Td^-1 @ Hn @ Ts
    tdi = la.inv(td)
    if tdi is None:
        return None
    h = la.matmul(tdi, la.matmul(hn, ts))
    if abs(h[2][2]) < 1e-15:
        return None
    return la.mat_scale(h, 1.0 / h[2][2])


def apply(h, pt):
    """Map one point through a homography."""
    x, y = pt[0], pt[1]
    w = h[2][0] * x + h[2][1] * y + h[2][2]
    if abs(w) < 1e-15:
        w = 1e-15
    return (
        (h[0][0] * x + h[0][1] * y + h[0][2]) / w,
        (h[1][0] * x + h[1][1] * y + h[1][2]) / w,
    )


def transfer_error(h, src, dst):
    """Mean symmetric-free forward transfer error in pixels."""
    total = 0.0
    for s, d in zip(src, dst):
        p = apply(h, s)
        total += math.hypot(p[0] - d[0], p[1] - d[1])
    return total / len(src)
