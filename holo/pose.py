"""Camera model and 6-DoF pose estimation from a planar marker.

Two steps, the same ones a production AR tracker uses:

1. :func:`pose_from_homography` -- an analytic initial guess. For a planar
   target, ``H = K [r1 r2 t]`` up to scale, so the first two rotation columns
   and the translation fall straight out of ``K^-1 H``. The two columns come
   back neither unit-length nor orthogonal (noise), so they get symmetrically
   orthogonalised onto SO(3).
2. :func:`refine_pose` -- Levenberg-Marquardt on the real objective, the sum of
   squared reprojection errors, parameterised by a minimal 6-vector
   (axis-angle + translation). This is what turns a decent guess into a pose
   that holds still instead of jittering.
"""

import math

from . import linalg as la


class Camera:
    """Pinhole intrinsics. No distortion model -- the synthetic renderer is a
    perfect pinhole, so adding one would only fit noise."""

    __slots__ = ("fx", "fy", "cx", "cy", "width", "height")

    def __init__(self, fx, fy, cx, cy, width=0, height=0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height

    @staticmethod
    def from_fov(width, height, fov_x_deg=60.0):
        """Build intrinsics from a horizontal field of view."""
        fx = (width / 2.0) / math.tan(math.radians(fov_x_deg) / 2.0)
        return Camera(fx, fx, width / 2.0, height / 2.0, width, height)

    @property
    def matrix(self):
        return [
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ]

    def project_point(self, p_cam):
        """Project a camera-frame 3-D point. Returns ``None`` if behind the camera."""
        z = p_cam[2]
        if z <= 1e-6:
            return None
        return (self.fx * p_cam[0] / z + self.cx, self.fy * p_cam[1] / z + self.cy)

    def __repr__(self):
        return "Camera(fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f)" % (
            self.fx,
            self.fy,
            self.cx,
            self.cy,
        )


# ------------------------------------------------------------------ transforms


def transform(r, t, p):
    """Apply ``r @ p + t``."""
    return [
        r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0],
        r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1],
        r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2],
    ]


def project_points(cam, r, t, pts3d):
    """Project object-frame points given the object-to-camera pose."""
    out = []
    for p in pts3d:
        out.append(cam.project_point(transform(r, t, p)))
    return out


def reprojection_rmse(cam, r, t, obj_pts, img_pts):
    """Root-mean-square reprojection error in pixels."""
    total = 0.0
    n = 0
    for p, q in zip(obj_pts, img_pts):
        pr = cam.project_point(transform(r, t, p))
        if pr is None:
            return float("inf")
        total += (pr[0] - q[0]) ** 2 + (pr[1] - q[1]) ** 2
        n += 1
    return math.sqrt(total / n) if n else float("inf")


# ----------------------------------------------------------- initial estimate


def pose_from_homography(cam, h):
    """Decompose a marker-plane homography into ``(r, t)``.

    The marker's object frame is centred on the marker with +Z out of its face,
    so object points are ``(x, y, 0)`` and the third homography column carries
    the translation.
    """
    kinv = la.inv(cam.matrix)
    if kinv is None:
        return None
    m = la.matmul(kinv, h)
    m1 = [m[0][0], m[1][0], m[2][0]]
    m2 = [m[0][1], m[1][1], m[2][1]]
    m3 = [m[0][2], m[1][2], m[2][2]]
    n1, n2 = la.norm(m1), la.norm(m2)
    if n1 < 1e-12 or n2 < 1e-12:
        return None
    # H is known only up to scale; both rotation columns must be unit length, so
    # the geometric mean of their norms is the most stable scale estimate.
    scale = 1.0 / math.sqrt(n1 * n2)
    r1 = la.vscale(m1, scale)
    r2 = la.vscale(m2, scale)
    t = la.vscale(m3, scale)
    if t[2] < 0:
        # The marker must be in front of the camera; the opposite sign is the
        # mirror solution of the same homography.
        r1 = la.vscale(r1, -1.0)
        r2 = la.vscale(r2, -1.0)
        t = la.vscale(t, -1.0)

    # Symmetric orthogonalisation: (r1+r2) and (r1-r2) are orthogonal whenever
    # |r1| == |r2|, so rebuilding from that basis splits the correction evenly
    # between the two columns instead of favouring r1 (as Gram-Schmidt would).
    u = la.normalize(la.vadd(r1, r2))
    v = la.normalize(la.vsub(r1, r2))
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    c1 = la.vscale(la.vadd(u, v), inv_sqrt2)
    c2 = la.vscale(la.vsub(u, v), inv_sqrt2)
    c3 = la.cross(c1, c2)
    r = [
        [c1[0], c2[0], c3[0]],
        [c1[1], c2[1], c3[1]],
        [c1[2], c2[2], c3[2]],
    ]
    return la.orthonormalize(r), t


# --------------------------------------------------------- nonlinear refinement


def _residuals(cam, params, obj_pts, img_pts):
    r = la.rodrigues(params[0:3])
    t = params[3:6]
    res = []
    for p, q in zip(obj_pts, img_pts):
        pc = transform(r, t, p)
        z = pc[2]
        if z <= 1e-6:
            return None
        res.append(cam.fx * pc[0] / z + cam.cx - q[0])
        res.append(cam.fy * pc[1] / z + cam.cy - q[1])
    return res


def refine_pose(cam, obj_pts, img_pts, r0, t0, iters=40):
    """Levenberg-Marquardt minimisation of reprojection error.

    Jacobians are computed by central differences. Analytic derivatives would be
    faster, but with 8 residuals and 6 parameters the numeric version costs
    12 extra projections per iteration -- irrelevant next to the per-pixel work
    elsewhere -- and it cannot disagree with the forward model.

    Returns ``(r, t, rmse)``.
    """
    params = list(la.rot_to_vec(r0)) + list(t0)
    res = _residuals(cam, params, obj_pts, img_pts)
    if res is None:
        return r0, list(t0), float("inf")
    cost = sum(x * x for x in res)
    lam = 1e-3
    n_p = 6
    for _ in range(iters):
        # Numeric Jacobian (central differences, parameter-scaled step).
        jac = []
        ok = True
        for k in range(n_p):
            step = 1e-6 * max(1.0, abs(params[k]))
            pp = list(params)
            pm = list(params)
            pp[k] += step
            pm[k] -= step
            rp = _residuals(cam, pp, obj_pts, img_pts)
            rm = _residuals(cam, pm, obj_pts, img_pts)
            if rp is None or rm is None:
                ok = False
                break
            jac.append([(a - b) / (2.0 * step) for a, b in zip(rp, rm)])
        if not ok:
            break
        m = len(res)
        # Normal equations: (J^T J + lam * diag(J^T J)) delta = -J^T r
        jtj = [[sum(jac[i][k] * jac[j][k] for k in range(m)) for j in range(n_p)] for i in range(n_p)]
        jtr = [-sum(jac[i][k] * res[k] for k in range(m)) for i in range(n_p)]
        improved = False
        for _attempt in range(8):
            aug = [row[:] for row in jtj]
            for i in range(n_p):
                aug[i][i] += lam * (jtj[i][i] if jtj[i][i] > 1e-12 else 1.0)
            delta = la.gauss_solve(aug, jtr)
            if delta is None:
                lam *= 10.0
                continue
            cand = [params[i] + delta[i] for i in range(n_p)]
            cres = _residuals(cam, cand, obj_pts, img_pts)
            if cres is not None:
                ccost = sum(x * x for x in cres)
                if ccost < cost:
                    params, res, cost = cand, cres, ccost
                    lam = max(lam * 0.3, 1e-12)
                    improved = True
                    break
            lam *= 10.0
            if lam > 1e12:
                break
        if not improved:
            break
        if math.sqrt(cost / (len(res) / 2)) < 1e-9:
            break
    r = la.orthonormalize(la.rodrigues(params[0:3]))
    t = params[3:6]
    return r, t, math.sqrt(cost / (len(res) / 2))


def solve_marker_pose(cam, obj_pts, img_pts, refine=True):
    """Full planar-marker pose solve: DLT homography -> analytic pose -> LM.

    Returns ``(r, t, rmse_initial, rmse_refined)`` or ``None``.
    """
    from . import homography

    src = [(p[0], p[1]) for p in obj_pts]  # marker plane is z = 0
    h = homography.dlt(src, img_pts)
    if h is None:
        return None
    guess = pose_from_homography(cam, h)
    if guess is None:
        return None
    r0, t0 = guess
    rmse0 = reprojection_rmse(cam, r0, t0, obj_pts, img_pts)
    if not refine:
        return r0, t0, rmse0, rmse0
    r, t, rmse1 = refine_pose(cam, obj_pts, img_pts, r0, t0)
    if rmse1 > rmse0:  # never return something worse than the analytic guess
        return r0, t0, rmse0, rmse0
    return r, t, rmse0, rmse1


def camera_position_in_object_frame(r, t):
    """Where the camera sits in the marker's own frame (useful for HUD readouts)."""
    rt = la.transpose(r)
    return la.vscale(la.matvec(rt, t), -1.0)
