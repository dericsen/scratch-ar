"""Pure-Python linear algebra.

Matrices are lists of row-lists, vectors are flat lists. No numpy, no C
extensions -- everything here is written from scratch because the whole point
of HOLO is to have zero dependencies.

The heavy lifters:
  * ``gauss_solve`` / ``inv``   -- Gauss-Jordan with partial pivoting.
  * ``jacobi_eigen``           -- cyclic Jacobi rotations for symmetric matrices.
  * ``null_vector``            -- smallest singular vector of A, obtained as the
                                  eigenvector of A^T A with the smallest
                                  eigenvalue. This is how the homography DLT
                                  gets solved without a full SVD.
  * ``rodrigues`` / ``rot_to_vec`` -- SO(3) <-> axis-angle, used to give the
                                  pose refiner a minimal 6-parameter space.
"""

import math

# ---------------------------------------------------------------- constructors


def zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


def eye(n):
    m = zeros(n, n)
    for i in range(n):
        m[i][i] = 1.0
    return m


def copy(a):
    return [row[:] for row in a]


# ------------------------------------------------------------- basic operators


def matmul(a, b):
    """Standard matrix product. ``a`` is (n x m), ``b`` is (m x p)."""
    n, m, p = len(a), len(b), len(b[0])
    out = [[0.0] * p for _ in range(n)]
    for i in range(n):
        ai = a[i]
        oi = out[i]
        for k in range(m):
            aik = ai[k]
            if aik == 0.0:
                continue
            bk = b[k]
            for j in range(p):
                oi[j] += aik * bk[j]
    return out


def matvec(a, v):
    return [sum(row[i] * v[i] for i in range(len(v))) for row in a]


def transpose(a):
    return [list(col) for col in zip(*a)]


def mat_scale(a, s):
    return [[x * s for x in row] for row in a]


def mat_sub(a, b):
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


# ------------------------------------------------------------- vector helpers


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def normalize(v):
    n = norm(v)
    if n < 1e-300:
        return list(v)
    return [x / n for x in v]


def cross(u, v):
    return [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]


def vsub(u, v):
    return [a - b for a, b in zip(u, v)]


def vadd(u, v):
    return [a + b for a, b in zip(u, v)]


def vscale(v, s):
    return [x * s for x in v]


# ------------------------------------------------------------- linear systems


def gauss_solve(a, b):
    """Solve ``a x = b`` by Gaussian elimination with partial pivoting.

    Returns ``None`` when the system is singular, so callers can fall back
    instead of crashing on a degenerate configuration.
    """
    n = len(a)
    m = [a[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-14:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        pv = m[col][col]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col] / pv
            if f == 0.0:
                continue
            mr, mc = m[r], m[col]
            for c in range(col, n + 1):
                mr[c] -= f * mc[c]
    return [m[i][n] / m[i][i] for i in range(n)]


def inv(a):
    """Matrix inverse via Gauss-Jordan. Returns ``None`` if singular."""
    n = len(a)
    m = [a[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-14:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        pv = m[col][col]
        mc = m[col]
        for c in range(2 * n):
            mc[c] /= pv
        for r in range(n):
            if r == col:
                continue
            f = m[r][col]
            if f == 0.0:
                continue
            mr = m[r]
            for c in range(2 * n):
                mr[c] -= f * mc[c]
    return [row[n:] for row in m]


# --------------------------------------------------------- eigen / null space


def jacobi_eigen(s, sweeps=100, eps=1e-16):
    """Eigen-decomposition of a *symmetric* matrix via cyclic Jacobi rotations.

    Returns ``(values, vectors)`` sorted by ascending eigenvalue, where
    ``vectors[i]`` is the unit eigenvector belonging to ``values[i]``.

    Jacobi is slower than a proper QR-based solver but it is short, numerically
    well behaved for symmetric input, and completely self-contained -- exactly
    the trade we want here.
    """
    n = len(s)
    a = copy(s)
    v = eye(n)
    for _ in range(sweeps):
        off = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                off += a[i][j] * a[i][j]
        if off <= eps:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) < 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * apq)
                sign = 1.0 if theta >= 0.0 else -1.0
                t = sign / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                sn = t * c
                for k in range(n):  # columns
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - sn * akq
                    a[k][q] = sn * akp + c * akq
                for k in range(n):  # rows
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - sn * aqk
                    a[q][k] = sn * apk + c * aqk
                for k in range(n):  # accumulate rotations
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - sn * vkq
                    v[k][q] = sn * vkp + c * vkq
    vals = [a[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: vals[i])
    out_vals = [vals[i] for i in order]
    out_vecs = [[v[r][i] for r in range(n)] for i in order]
    return out_vals, out_vecs


def null_vector(a):
    """Unit vector x minimising ||a x|| -- i.e. the right singular vector of
    ``a`` with the smallest singular value."""
    cols = len(a[0])
    ata = zeros(cols, cols)
    for row in a:
        for i in range(cols):
            ri = row[i]
            if ri == 0.0:
                continue
            target = ata[i]
            for j in range(cols):
                target[j] += ri * row[j]
    _, vecs = jacobi_eigen(ata)
    return normalize(vecs[0])


# ------------------------------------------------------------------ rotations


def rodrigues(v):
    """Axis-angle (3-vector, magnitude = angle in radians) -> 3x3 rotation."""
    theta = norm(v)
    if theta < 1e-12:
        # Second-order expansion is unnecessary at this magnitude; identity plus
        # the skew part is already accurate to float precision.
        kx, ky, kz = v
        return [[1.0, -kz, ky], [kz, 1.0, -kx], [-ky, kx, 1.0]]
    kx, ky, kz = v[0] / theta, v[1] / theta, v[2] / theta
    c, s = math.cos(theta), math.sin(theta)
    ci = 1.0 - c
    return [
        [c + kx * kx * ci, kx * ky * ci - kz * s, kx * kz * ci + ky * s],
        [ky * kx * ci + kz * s, c + ky * ky * ci, ky * kz * ci - kx * s],
        [kz * kx * ci - ky * s, kz * ky * ci + kx * s, c + kz * kz * ci],
    ]


def rot_to_vec(r):
    """3x3 rotation -> axis-angle 3-vector (inverse of ``rodrigues``)."""
    trace = r[0][0] + r[1][1] + r[2][2]
    cos_t = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    theta = math.acos(cos_t)
    if theta < 1e-9:
        return [0.0, 0.0, 0.0]
    if theta > math.pi - 1e-6:
        # Near 180 degrees sin(theta) vanishes, so recover the axis from the
        # symmetric part (R + I) instead, which stays well conditioned.
        best, bi = -1.0, 0
        for i in range(3):
            if r[i][i] > best:
                best, bi = r[i][i], i
        axis = [r[0][bi], r[1][bi], r[2][bi]]
        axis[bi] += 1.0
        axis = normalize(axis)
        return vscale(axis, theta)
    k = 1.0 / (2.0 * math.sin(theta))
    return [
        (r[2][1] - r[1][2]) * k * theta,
        (r[0][2] - r[2][0]) * k * theta,
        (r[1][0] - r[0][1]) * k * theta,
    ]


def orthonormalize(r):
    """Project an approximate rotation back onto SO(3) (Gram-Schmidt)."""
    c0 = normalize([r[0][0], r[1][0], r[2][0]])
    c1 = [r[0][1], r[1][1], r[2][1]]
    c1 = normalize(vsub(c1, vscale(c0, dot(c0, c1))))
    c2 = cross(c0, c1)
    return [
        [c0[0], c1[0], c2[0]],
        [c0[1], c1[1], c2[1]],
        [c0[2], c1[2], c2[2]],
    ]


def rotation_angle_between(ra, rb):
    """Geodesic angle (radians) between two rotations -- the standard metric for
    reporting pose error."""
    rel = matmul(transpose(ra), rb)
    trace = rel[0][0] + rel[1][1] + rel[2][2]
    cos_t = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.acos(cos_t)


def look_at(eye_pos, target, up=(0.0, 1.0, 0.0)):
    """Camera extrinsics from a look-at specification.

    Returns ``(r_cw, t_cw)`` mapping world points into the camera frame via
    ``x_cam = r_cw @ x_world + t_cw``, using the computer-vision convention:
    +X right, +Y down, +Z forward (into the scene).
    """
    fwd = normalize(vsub(target, eye_pos))
    right = cross(fwd, list(up))
    if norm(right) < 1e-9:
        right = cross(fwd, [0.0, 0.0, 1.0])
    right = normalize(right)
    down = cross(fwd, right)
    r_cw = [right, down, fwd]  # rows
    t_cw = [-dot(right, eye_pos), -dot(down, eye_pos), -dot(fwd, eye_pos)]
    return r_cw, t_cw
