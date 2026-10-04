"""Quantitative evaluation against ground truth.

One design decision worth spelling out: detections are matched to ground-truth
markers **by geometry, not by ID**. Matching on ID would quietly hide the most
embarrassing possible failure -- finding the marker in the right place but
decoding the wrong number -- because a mis-decoded detection would simply be
counted as a miss plus a false positive. Matching on image position first and
*then* comparing IDs is what makes "ID accuracy" a real measurement.

Markers that are out of frame, too small, or viewed edge-on are excluded from
the detection-rate denominator via :func:`is_detectable`. A marker turned
almost perpendicular to the camera carries no recoverable signal, and counting
it as a failure would measure the scene, not the detector.
"""

import math

from . import linalg as la


def centroid(pts):
    n = len(pts)
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)


def quad_area(pts):
    s = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return abs(s) / 2.0


def view_angle_deg(gt):
    """Angle between the marker's surface normal and the line of sight."""
    normal = [gt.r_cm[0][2], gt.r_cm[1][2], gt.r_cm[2][2]]
    view = la.normalize(list(gt.t_cm))
    c = abs(la.dot(normal, view))
    return math.degrees(math.acos(max(0.0, min(1.0, c))))


def is_detectable(gt, cam, min_area=250.0, max_view_angle=72.0, margin=2.0):
    # 250 px^2 is roughly a 16x16 px marker, i.e. about 2.3 px per cell. The
    # threshold is set from what the detector is actually observed to manage
    # (see the distance sweep) rather than picked to flatter it.
    """Could this marker reasonably be found at all?"""
    for x, y in gt.corners:
        if not (-margin <= x < cam.width + margin and -margin <= y < cam.height + margin):
            return False
    if quad_area(gt.corners) < min_area:
        return False
    if view_angle_deg(gt) > max_view_angle:
        return False
    return True


def errors_for(det, gt):
    """Full error record for one matched detection/ground-truth pair."""
    rot = math.degrees(la.rotation_angle_between(gt.r_cm, det.r))
    dt = la.vsub(list(gt.t_cm), list(det.t))
    pos = la.norm(dt)
    corner = [math.dist(a, b) for a, b in zip(gt.corners, det.corners)]
    est_dist = la.norm(list(det.t))
    return {
        "marker_id": det.marker_id,
        "gt_id": gt.marker_id,
        "id_correct": det.marker_id == gt.marker_id,
        "rot_deg": rot,
        "pos_mm": pos * 1000.0,
        "pos_pct": pos / gt.distance * 100.0,
        "depth_bias_pct": (est_dist - gt.distance) / gt.distance * 100.0,
        "corner_mean_px": sum(corner) / len(corner),
        "corner_max_px": max(corner),
        "rmse_px": det.rmse,
        "rmse_initial_px": det.rmse_initial,
        "distance_m": gt.distance,
        "view_angle_deg": view_angle_deg(gt),
        "apparent_area_px": quad_area(gt.corners),
        "bit_errors": det.bit_errors,
    }


def match(detections, ground_truths, max_center_dist=40.0):
    """Greedy nearest-centroid matching.

    Returns ``(pairs, missed, false_positives)`` where ``pairs`` is a list of
    ``(detection, ground_truth)``.
    """
    remaining = list(ground_truths)
    pairs = []
    false_pos = []
    for det in sorted(detections, key=lambda d: d.distance):
        dc = centroid(det.corners)
        best = None
        for gt in remaining:
            d = math.dist(dc, centroid(gt.corners))
            if d <= max_center_dist and (best is None or d < best[0]):
                best = (d, gt)
        if best is None:
            false_pos.append(det)
        else:
            remaining.remove(best[1])
            pairs.append((det, best[1]))
    return pairs, remaining, false_pos


def evaluate_frame(detections, ground_truths, cam, **detectable_kw):
    """Evaluate one frame. Returns a dict with per-pair errors and counts."""
    detectable = [g for g in ground_truths if is_detectable(g, cam, **detectable_kw)]
    pairs, missed, false_pos = match(detections, ground_truths)
    detectable_ids = {id(g) for g in detectable}
    recs = [errors_for(d, g) for d, g in pairs]
    return {
        "errors": recs,
        "n_detectable": len(detectable),
        "n_detected": len(pairs),
        "n_detectable_detected": sum(1 for _d, g in pairs if id(g) in detectable_ids),
        "n_missed_detectable": sum(1 for g in missed if id(g) in detectable_ids),
        "n_false_positive": len(false_pos),
        "n_id_wrong": sum(1 for r in recs if not r["id_correct"]),
    }


# ------------------------------------------------------------------ statistics


def _pct(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q / 100.0 * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def summarize(frame_results, keys=("rot_deg", "pos_mm", "pos_pct", "corner_mean_px",
                                   "rmse_px", "rmse_initial_px", "depth_bias_pct")):
    """Aggregate a list of :func:`evaluate_frame` results."""
    recs = [r for fr in frame_results for r in fr["errors"]]
    out = {
        "frames": len(frame_results),
        "detectable": sum(fr["n_detectable"] for fr in frame_results),
        "detected": sum(fr["n_detectable_detected"] for fr in frame_results),
        "missed": sum(fr["n_missed_detectable"] for fr in frame_results),
        "false_positives": sum(fr["n_false_positive"] for fr in frame_results),
        "id_wrong": sum(fr["n_id_wrong"] for fr in frame_results),
        "matched": len(recs),
    }
    out["detection_rate_pct"] = (
        100.0 * out["detected"] / out["detectable"] if out["detectable"] else float("nan")
    )
    out["id_accuracy_pct"] = (
        100.0 * (out["matched"] - out["id_wrong"]) / out["matched"] if out["matched"] else float("nan")
    )
    stats = {}
    for k in keys:
        vals = [r[k] for r in recs if r[k] == r[k]]  # drop NaN
        if not vals:
            continue
        stats[k] = {
            "mean": sum(vals) / len(vals),
            "median": _pct(vals, 50),
            "p95": _pct(vals, 95),
            "max": max(vals),
            "min": min(vals),
        }
    out["stats"] = stats
    return out


def format_report(summary, title="ACCURACY vs GROUND TRUTH"):
    """Human-readable console table."""
    lines = []
    lines.append("=" * 68)
    lines.append(title)
    lines.append("=" * 68)
    lines.append(
        "frames %d | detectable markers %d | detected %d (%.1f%%) | missed %d"
        % (
            summary["frames"],
            summary["detectable"],
            summary["detected"],
            summary["detection_rate_pct"],
            summary["missed"],
        )
    )
    lines.append(
        "false positives %d | ID accuracy %.2f%% (%d wrong of %d matched)"
        % (
            summary["false_positives"],
            summary["id_accuracy_pct"],
            summary["id_wrong"],
            summary["matched"],
        )
    )
    lines.append("-" * 68)
    lines.append("%-22s %9s %9s %9s %9s" % ("metric", "mean", "median", "p95", "max"))
    labels = [
        ("rot_deg", "rotation error (deg)"),
        ("pos_mm", "position error (mm)"),
        ("pos_pct", "position error (%)"),
        ("depth_bias_pct", "depth bias (%)"),
        ("corner_mean_px", "corner error (px)"),
        ("rmse_initial_px", "reproj before LM (px)"),
        ("rmse_px", "reproj after LM (px)"),
    ]
    for key, label in labels:
        s = summary["stats"].get(key)
        if not s:
            continue
        lines.append(
            "%-22s %9.4f %9.4f %9.4f %9.4f" % (label, s["mean"], s["median"], s["p95"], s["max"])
        )
    lines.append("=" * 68)
    return "\n".join(lines)
