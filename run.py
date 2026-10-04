#!/usr/bin/env python3
"""HOLO command line -- generate every demo artefact.

    python3 run.py all          # everything below
    python3 run.py markers      # printable marker sheet
    python3 run.py stages       # pipeline figure, stage by stage
    python3 run.py frame        # one annotated frame + error numbers
    python3 run.py orbit        # animated AR GIF over a camera orbit
    python3 run.py bench        # accuracy sweeps (distance / tilt / noise)
    python3 run.py selftest     # unit tests

Requires nothing but CPython >= 3.9. No third-party packages, by design.
"""

import argparse
import json
import math
import os
import sys
import time

from holo import detect as detect_mod
from holo import evaluate, gif, linalg as la
from holo import marker as mk
from holo import render as rr
from holo import scene as scene_mod
from holo import filters
from holo.image import Image, montage
from holo.pose import Camera

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

WIDTH, HEIGHT = 480, 320
FOV = 58.0
MARKER_SIDE = 0.10  # metres, side of the 7x7 black square


def setup():
    dictionary = mk.generate_dictionary(count=16, min_dist=8)
    cam = Camera.from_fov(WIDTH, HEIGHT, FOV)
    placements = scene_mod.default_scene(dictionary, side=MARKER_SIDE)
    return dictionary, cam, placements


def ensure_out():
    os.makedirs(OUT, exist_ok=True)
    return OUT


def _banner(msg):
    print("\n" + "-" * 68)
    print(msg)
    print("-" * 68)


# --------------------------------------------------------------- marker sheet


def cmd_markers(args):
    """Render the dictionary as a printable sheet."""
    ensure_out()
    dictionary, _cam, _pl = setup()
    stats = mk.dictionary_stats(dictionary)
    cell = 12
    tiles = []
    for i, grid in enumerate(dictionary.grids):
        tiles.append(("ID %02d" % i, mk.render_marker(grid, cell_px=cell)))
    sheet = montage(tiles, cols=4, pad=10, label_h=14, bg=(24, 26, 34))
    path = os.path.join(OUT, "markers.png")
    sheet.save(path)
    _banner("MARKER DICTIONARY")
    print("markers                      : %d" % stats["markers"])
    print("min inter-marker distance    : %d bits (all 4 rotations)" % stats["min_inter_distance"])
    print("min self-rotation distance   : %d bits" % stats["min_self_rotation_distance"])
    print("correctable bit errors       : %d of %d" % (dictionary.max_correctable, mk.PAYLOAD ** 2))
    print("wrote %s (%dx%d)" % (path, sheet.w, sheet.h))
    return {"dictionary": stats, "correctable_bits": dictionary.max_correctable}


# ------------------------------------------------------------ pipeline figure


def cmd_stages(args):
    """Render the detector pipeline, one panel per stage."""
    ensure_out()
    dictionary, cam, placements = setup()
    r_cw, t_cw = scene_mod.orbit_trajectory(1, start_deg=-115.0)[0]
    img, gts = scene_mod.render_frame(cam, r_cw, t_cw, placements, supersample=2, seed=1)
    dets, st = detect_mod.detect(
        img, cam, dictionary, marker_side=MARKER_SIDE, collect_stages=True
    )

    panels = [("1. SYNTHETIC CAMERA FRAME", img)]
    panels.append(("2. GRAYSCALE", st.gray.to_rgb()))
    panels.append(("3. ADAPTIVE THRESHOLD", st.binary.to_rgb()))
    panels.append(("4. MORPHOLOGICAL CLOSE", st.closed.to_rgb()))

    # 5. candidate quads: everything the shape filter let through
    cand = img.clone()
    for quad in st.candidates:
        cand.draw_poly([(p[0], p[1]) for p in quad], (255, 80, 80), 1)
    for quad in st.rejected:
        cand.draw_poly([(p[0], p[1]) for p in quad], (255, 180, 40), 1)
    for det in dets:
        cand.draw_poly([(c[0], c[1]) for c in det.corners], (0, 255, 160), 2)
    rr.draw_hud(
        cand,
        ["RED   %d SHAPE CANDIDATES" % len(st.candidates),
         "AMBER %d FAILED DECODE" % len(st.rejected),
         "GREEN %d ACCEPTED" % len(dets)],
        scale=1,
    )
    panels.append(("5. QUAD CANDIDATES / DECODE", cand))

    # 6. rectified patches
    patch_panel = Image(img.w, img.h, fill=(18, 18, 24))
    x = 6
    for mid, patch in st.patches[:3]:
        up = patch.to_rgb().scaled(2)
        patch_panel.paste(up, x, 30)
        patch_panel.draw_text(x, 14, "ID %02d" % mid, (0, 229, 255), scale=1)
        x += up.w + 10
    patch_panel.draw_text(6, 30 + 150, "PERSPECTIVE-RECTIFIED 7X7 CELL GRID", (200, 200, 215), scale=1)
    patch_panel.draw_text(6, 30 + 162, "SAMPLED WITH OTSU PER PATCH", (200, 200, 215), scale=1)
    panels.append(("6. UNWARPED MARKER PATCHES", patch_panel))

    # 7. final AR overlay
    final = img.clone()
    gtmap = {g.marker_id: g for g in gts}
    rr.augment(final, cam, dets, MARKER_SIDE, phase=1.0, ground_truth=gtmap)
    rr.draw_hud(final, ["%d MARKERS TRACKED" % len(dets), "WHITE DASH = GROUND TRUTH"],
                title="HOLO AR", scale=1)
    panels.append(("7. AR OVERLAY (6-DOF POSE)", final))

    # 8. sobel, purely illustrative
    panels.append(("8. SOBEL GRADIENT (DIAGNOSTIC)", filters.sobel_magnitude(st.gray).to_rgb()))

    fig = montage(panels, cols=4, pad=10, label_h=14)
    path = os.path.join(OUT, "pipeline_stages.png")
    fig.save(path)
    _banner("PIPELINE FIGURE")
    print("shape candidates %d -> decode-rejected %d -> accepted %d"
          % (len(st.candidates), len(st.rejected), len(dets)))
    print("wrote %s (%dx%d)" % (path, fig.w, fig.h))
    return {"candidates": len(st.candidates), "rejected": len(st.rejected), "accepted": len(dets)}


# -------------------------------------------------------------- single frame


def cmd_frame(args):
    """One frame, annotated, with the full error breakdown printed."""
    ensure_out()
    dictionary, cam, placements = setup()
    r_cw, t_cw = scene_mod.orbit_trajectory(1, start_deg=args.angle)[0]
    t0 = time.time()
    img, gts = scene_mod.render_frame(cam, r_cw, t_cw, placements, supersample=2, seed=5)
    t_render = time.time() - t0
    t0 = time.time()
    dets, _ = detect_mod.detect(img, cam, dictionary, marker_side=MARKER_SIDE)
    t_detect = time.time() - t0

    res = evaluate.evaluate_frame(dets, gts, cam)
    gtmap = {g.marker_id: g for g in gts}
    out = img.clone()
    rr.augment(out, cam, dets, MARKER_SIDE, phase=1.2, ground_truth=gtmap)
    rr.draw_hud(
        out,
        [
            "DETECTED %d / %d" % (res["n_detected"], res["n_detectable"]),
            "RENDER  %5.2F S" % t_render,
            "DETECT  %5.2F S" % t_detect,
            "WHITE DASH = GROUND TRUTH",
        ],
        title="HOLO  ZERO-DEP AR",
    )
    path = os.path.join(OUT, "frame_annotated.png")
    out.save(path)

    _banner("SINGLE FRAME vs GROUND TRUTH")
    print("render %.2fs | detect %.2fs | %d/%d detectable markers found"
          % (t_render, t_detect, res["n_detected"], res["n_detectable"]))
    print()
    print("%-4s %-4s %8s %9s %9s %9s %9s %8s" %
          ("id", "ok", "rot deg", "pos mm", "pos %", "corner px", "rmse px", "view deg"))
    for r in sorted(res["errors"], key=lambda x: x["marker_id"]):
        print("%-4d %-4s %8.3f %9.2f %9.3f %9.3f %9.3f %8.1f" % (
            r["marker_id"], "yes" if r["id_correct"] else "NO!",
            r["rot_deg"], r["pos_mm"], r["pos_pct"],
            r["corner_mean_px"], r["rmse_px"], r["view_angle_deg"]))
    print("\nwrote %s" % path)
    return {"frame": res, "render_s": t_render, "detect_s": t_detect}


# ---------------------------------------------------------------- orbit / gif


def cmd_orbit(args):
    """Full camera orbit: AR frames, animated GIF, and aggregate accuracy."""
    ensure_out()
    dictionary, cam, placements = setup()
    n = args.frames
    traj = scene_mod.orbit_trajectory(n)
    frames = []
    frame_results = []
    keep_png = args.save_frames
    if keep_png:
        os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)

    _banner("ORBIT: %d FRAMES" % n)
    t_start = time.time()
    t_render_total = t_detect_total = 0.0
    for i, (r_cw, t_cw) in enumerate(traj):
        t0 = time.time()
        img, gts = scene_mod.render_frame(
            cam, r_cw, t_cw, placements, supersample=args.supersample, seed=1000 + i
        )
        t_render_total += time.time() - t0
        t0 = time.time()
        dets, _ = detect_mod.detect(img, cam, dictionary, marker_side=MARKER_SIDE)
        t_detect_total += time.time() - t0

        res = evaluate.evaluate_frame(dets, gts, cam)
        frame_results.append(res)

        gtmap = {g.marker_id: g for g in gts}
        phase = 2.0 * math.pi * i / max(1, n) * 3.0
        rr.augment(img, cam, dets, MARKER_SIDE, phase=phase, ground_truth=gtmap)
        mean_rot = (sum(r["rot_deg"] for r in res["errors"]) / len(res["errors"])
                    if res["errors"] else 0.0)
        rr.draw_hud(
            img,
            [
                "FRAME %02d / %02d" % (i + 1, n),
                "TRACKED %d / %d" % (res["n_detected"], res["n_detectable"]),
                "ROT ERR %5.2F DEG" % mean_rot,
            ],
            title="HOLO  ZERO-DEP AR",
        )
        rr.draw_progress_bar(img, (i + 1) / n)
        frames.append(img)
        if keep_png:
            img.save(os.path.join(OUT, "frames", "frame_%03d.png" % i))
        print("  frame %02d/%02d  tracked %d/%d  mean rot err %.3f deg"
              % (i + 1, n, res["n_detected"], res["n_detectable"], mean_rot))

    t0 = time.time()
    path = os.path.join(OUT, "ar_orbit.gif")
    _, size = gif.write_animation(path, frames, delay_cs=args.delay, max_colors=args.colors)
    t_gif = time.time() - t0

    summary = evaluate.summarize(frame_results)
    print()
    print(evaluate.format_report(summary, "ORBIT ACCURACY vs GROUND TRUTH (%d frames)" % n))
    print()
    print("timing: render %.1fs | detect %.1fs | gif encode %.1fs | total %.1fs"
          % (t_render_total, t_detect_total, t_gif, time.time() - t_start))
    print("        per frame: render %.2fs, detect %.2fs"
          % (t_render_total / n, t_detect_total / n))
    print("wrote %s (%.2f MB, %d frames, %d colours)"
          % (path, size / 1e6, len(frames), args.colors))

    summary["timing"] = {
        "render_s_per_frame": t_render_total / n,
        "detect_s_per_frame": t_detect_total / n,
        "gif_encode_s": t_gif,
        "gif_bytes": size,
    }
    return summary


# ----------------------------------------------------------------- benchmarks


def _sweep(dictionary, cam, label, variants, noise=3.0, supersample=2):
    """Run one controlled sweep; ``variants`` yields (x_value, placements)."""
    rows = []
    for xval, placements, cam_pose in variants:
        r_cw, t_cw = cam_pose
        img, gts = scene_mod.render_frame(
            cam, r_cw, t_cw, placements, supersample=supersample, noise=noise, seed=int(xval * 977) & 0xFFFF
        )
        dets, _ = detect_mod.detect(img, cam, dictionary, marker_side=MARKER_SIDE)
        res = evaluate.evaluate_frame(dets, gts, cam)
        s = evaluate.summarize([res])
        rows.append((xval, res, s))
        st = s["stats"]
        print("  %-18s %-12s det %d/%d  rot %7s deg  pos %8s mm  corner %6s px" % (
            label, "%.3g" % xval,
            res["n_detected"], res["n_detectable"],
            ("%.3f" % st["rot_deg"]["mean"]) if "rot_deg" in st else "-",
            ("%.2f" % st["pos_mm"]["mean"]) if "pos_mm" in st else "-",
            ("%.3f" % st["corner_mean_px"]["mean"]) if "corner_mean_px" in st else "-",
        ))
    return rows


def cmd_bench(args):
    """Controlled sweeps: distance, tilt, and sensor noise."""
    ensure_out()
    dictionary, cam, _pl = setup()
    report = {}

    _banner("SWEEP 1: DISTANCE (single marker, face-on-ish)")
    variants = []
    for dist in (0.22, 0.30, 0.40, 0.55, 0.75, 1.00, 1.35, 1.80, 2.30, 2.80):
        pl = [scene_mod.marker_on_floor(0, dictionary.grids[0], 0.0, 0.0, 12.0, MARKER_SIDE)]
        # Look straight at the marker from 'dist' away, 35 degrees above the floor.
        elev = math.radians(35.0)
        eye = [0.0, dist * math.sin(elev), -dist * math.cos(elev)]
        variants.append((dist, pl, la.look_at(eye, [0.0, 0.0, 0.0])))
    report["distance"] = [
        {"distance_m": x, "detected": r["n_detected"], "detectable": r["n_detectable"],
         "stats": s["stats"]}
        for x, r, s in _sweep(dictionary, cam, "distance (m)", variants)
    ]

    _banner("SWEEP 2: VIEWING OBLIQUITY (camera elevation above a flat marker)")
    # Sweeping the camera's elevation is the clean way to vary viewing angle:
    # the marker stays flat so its apparent size and the angle off its normal
    # change together and monotonically, which is exactly the quantity of
    # interest. (Tilting the marker instead also swings it towards or away from
    # the camera, confounding the x axis with self-occlusion.)
    variants = []
    for view_angle in (5.0, 20.0, 35.0, 50.0, 62.0, 72.0, 80.0):
        pl = [scene_mod.marker_on_floor(0, dictionary.grids[0], 0.0, 0.0, 12.0, MARKER_SIDE)]
        elev = math.radians(90.0 - view_angle)
        dist = 0.45
        eye = [0.0, dist * math.sin(elev), -dist * math.cos(elev)]
        variants.append((view_angle, pl, la.look_at(eye, [0.0, 0.0, 0.0])))
    report["view_angle"] = [
        {"view_angle_deg": x, "detected": r["n_detected"], "detectable": r["n_detectable"],
         "stats": s["stats"]}
        for x, r, s in _sweep(dictionary, cam, "view angle (deg)", variants)
    ]

    _banner("SWEEP 3: SENSOR NOISE")
    rows = []
    for sigma in (0.0, 2.0, 5.0, 10.0, 16.0, 24.0, 34.0):
        pl = [scene_mod.marker_on_floor(0, dictionary.grids[0], 0.0, 0.0, 12.0, MARKER_SIDE)]
        elev = math.radians(35.0)
        eye = [0.0, 0.45 * math.sin(elev), -0.45 * math.cos(elev)]
        variants = [(sigma, pl, la.look_at(eye, [0.0, 0.0, 0.0]))]
        rows += _sweep(dictionary, cam, "noise sigma", variants, noise=sigma)
    report["noise"] = [
        {"noise_sigma": x, "detected": r["n_detected"], "detectable": r["n_detectable"],
         "stats": s["stats"]}
        for x, r, s in rows
    ]

    chart = _draw_bench_chart(report)
    cpath = os.path.join(OUT, "robustness.png")
    chart.save(cpath)
    print("\nwrote %s" % cpath)
    return report


def _draw_bench_chart(report):
    """Plot the sweeps using only our own drawing primitives."""
    panels = []
    specs = [
        ("distance", "distance_m", "DISTANCE (M)", "rot_deg", "ROT ERR (DEG)"),
        ("distance", "distance_m", "DISTANCE (M)", "pos_pct", "POS ERR (%)"),
        ("view_angle", "view_angle_deg", "VIEW ANGLE OFF NORMAL (DEG)", "rot_deg", "ROT ERR (DEG)"),
        ("noise", "noise_sigma", "NOISE SIGMA (GRAY LEVELS)", "rot_deg", "ROT ERR (DEG)"),
    ]
    for key, xk, xlabel, yk, ylabel in specs:
        rows = report.get(key, [])
        pts = [(r[xk], r["stats"][yk]["mean"]) for r in rows
               if r["stats"].get(yk) and r["detected"]]
        panels.append((xlabel + "  ->  " + ylabel, _line_plot(pts, xlabel, ylabel, rows, xk)))
    return montage(panels, cols=2, pad=12, label_h=14)


def _line_plot(pts, xlabel, ylabel, rows, xk, w=300, h=200):
    img = Image(w, h, fill=(20, 22, 30))
    m_l, m_r, m_t, m_b = 42, 10, 12, 26
    pw, ph = w - m_l - m_r, h - m_t - m_b
    img.fill_rect(m_l, m_t, m_l + pw, m_t + ph, (28, 30, 40))
    if not pts:
        img.draw_text(m_l + 8, m_t + ph // 2, "NO DATA", (200, 80, 80))
        return img
    # The x range must span *all* sweep points, including the ones where
    # detection failed -- those are plotted as red "X" ticks and are the most
    # informative part of the chart, so they must not fall off the axis.
    xs = [r[xk] for r in rows] or [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = 0.0, max(ys) * 1.15 + 1e-6
    if x1 - x0 < 1e-9:
        x1 = x0 + 1.0

    def sx(v):
        return m_l + (v - x0) / (x1 - x0) * pw

    def sy(v):
        return m_t + ph - (v - y0) / (y1 - y0) * ph

    # grid + y labels
    for k in range(5):
        gy = m_t + ph - k * ph / 4
        img.draw_line(m_l, gy, m_l + pw, gy, (44, 46, 58), 1)
        img.draw_text(2, gy - 3, "%.2f" % (y0 + k * (y1 - y0) / 4), (150, 155, 170), scale=1)
    # markers that were not detected at all get a red tick on the x axis
    for r in rows:
        if not r["detected"]:
            img.draw_line(sx(r[xk]), m_t, sx(r[xk]), m_t + ph, (120, 40, 50), 1)
            img.draw_text(sx(r[xk]) - 6, m_t + 4, "X", (220, 70, 70), scale=1)
    for i in range(len(pts) - 1):
        img.draw_line(sx(pts[i][0]), sy(pts[i][1]), sx(pts[i + 1][0]), sy(pts[i + 1][1]),
                      (0, 229, 255), 2)
    for px, py in pts:
        img.fill_circle(sx(px), sy(py), 3, (255, 196, 40))
    img.draw_text(m_l, h - 10, "%.3g" % x0, (170, 175, 190), scale=1)
    img.draw_text(m_l + pw - 24, h - 10, "%.3g" % x1, (170, 175, 190), scale=1)
    return img


# ------------------------------------------------------------------ selftest


def cmd_selftest(args):
    from tests import run_tests

    return {"passed": run_tests.main()}


# ---------------------------------------------------------------------- main


def cmd_all(args):
    report = {}
    report["markers"] = cmd_markers(args)
    report["stages"] = cmd_stages(args)
    report["frame"] = cmd_frame(args)
    report["orbit"] = cmd_orbit(args)
    report["bench"] = cmd_bench(args)
    path = os.path.join(OUT, "metrics.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print("\nwrote %s" % path)
    _banner("ARTEFACTS IN outputs/")
    for name in sorted(os.listdir(OUT)):
        full = os.path.join(OUT, name)
        if os.path.isfile(full):
            print("  %-26s %8.1f KB" % (name, os.path.getsize(full) / 1024))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="HOLO zero-dependency AR engine")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("markers", help="printable marker sheet")
    sub.add_parser("stages", help="detector pipeline figure")

    p = sub.add_parser("frame", help="single annotated frame")
    p.add_argument("--angle", type=float, default=-115.0, help="orbit angle in degrees")

    p = sub.add_parser("orbit", help="animated AR GIF over a camera orbit")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--delay", type=int, default=8, help="frame delay in centiseconds")
    p.add_argument("--colors", type=int, default=128)
    p.add_argument("--supersample", type=int, default=2)
    p.add_argument("--save-frames", action="store_true", help="also write each frame as PNG")

    sub.add_parser("bench", help="accuracy sweeps")
    sub.add_parser("selftest", help="run unit tests")

    p = sub.add_parser("all", help="everything")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--delay", type=int, default=8)
    p.add_argument("--colors", type=int, default=128)
    p.add_argument("--supersample", type=int, default=2)
    p.add_argument("--save-frames", action="store_true")
    p.add_argument("--angle", type=float, default=-115.0)

    args = ap.parse_args(argv)
    if not args.cmd:
        ap.print_help()
        return 0
    handlers = {
        "markers": cmd_markers,
        "stages": cmd_stages,
        "frame": cmd_frame,
        "orbit": cmd_orbit,
        "bench": cmd_bench,
        "selftest": cmd_selftest,
        "all": cmd_all,
    }
    handlers[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
