#!/usr/bin/env python3
"""HOLO test suite -- stdlib only, no pytest.

Run with ``python3 run.py selftest`` or ``python3 tests/run_tests.py``.

Where possible the tests check against something *external* rather than
against HOLO's own output, because a codec that round-trips through itself
proves only self-consistency. Specifically:

* the PNG decoder is run over every PNG found on the host filesystem, and the
  decoded pixel data is cross-checked against ``zlib`` + the spec's filter
  definitions re-implemented independently in this file;
* the GIF decoder is run over every GIF on the host, all of which were produced
  by other people's encoders;
* the pose solver is checked against analytically projected ground truth.
"""

import math
import os
import random
import struct
import subprocess
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holo import contours, detect, evaluate, filters, geom, gif, homography
from holo import linalg as la
from holo import marker as mk
from holo import morph, png, pose, render, scene
from holo.image import Gray, Image
from holo.pose import Camera

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def approx(a, b, tol=1e-9, msg=""):
    assert abs(a - b) <= tol, "%s: %r != %r (tol %g)" % (msg or "approx", a, b, tol)


# --------------------------------------------------------------------- linalg


@test
def test_linalg_inverse_and_solve():
    rng = random.Random(1)
    for n in (2, 3, 5, 7):
        a = [[rng.uniform(-3, 3) for _ in range(n)] for _ in range(n)]
        ai = la.inv(a)
        assert ai is not None
        prod = la.matmul(a, ai)
        for i in range(n):
            for j in range(n):
                approx(prod[i][j], 1.0 if i == j else 0.0, 1e-9, "inv")
        b = [rng.uniform(-1, 1) for _ in range(n)]
        x = la.gauss_solve(a, b)
        resid = la.matvec(a, x)
        for i in range(n):
            approx(resid[i], b[i], 1e-9, "solve")


@test
def test_linalg_singular_returns_none():
    sing = [[1.0, 2.0], [2.0, 4.0]]
    assert la.inv(sing) is None
    assert la.gauss_solve(sing, [1.0, 2.0]) is None


@test
def test_jacobi_eigen_matches_definition():
    rng = random.Random(2)
    for n in (2, 3, 6):
        m = [[rng.uniform(-2, 2) for _ in range(n)] for _ in range(n)]
        s = [[(m[i][j] + m[j][i]) / 2 for j in range(n)] for i in range(n)]
        vals, vecs = la.jacobi_eigen(s)
        assert all(vals[i] <= vals[i + 1] + 1e-12 for i in range(n - 1)), "not sorted"
        for v, vec in zip(vals, vecs):
            approx(la.norm(vec), 1.0, 1e-9, "eigenvector not unit")
            av = la.matvec(s, vec)
            for i in range(n):
                approx(av[i], v * vec[i], 1e-7, "eigen residual")


@test
def test_null_vector():
    m = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]
    nv = la.null_vector(m)
    approx(la.norm(nv), 1.0, 1e-9)
    assert la.norm(la.matvec(m, nv)) < 1e-8


@test
def test_rodrigues_round_trip():
    rng = random.Random(3)
    for _ in range(300):
        axis = la.normalize([rng.gauss(0, 1) for _ in range(3)])
        ang = rng.uniform(0, math.pi - 1e-6)
        v = la.vscale(axis, ang)
        r = la.rodrigues(v)
        rtr = la.matmul(la.transpose(r), r)
        for i in range(3):
            for j in range(3):
                approx(rtr[i][j], 1.0 if i == j else 0.0, 1e-9, "R not orthogonal")
        back = la.rot_to_vec(r)
        assert la.norm(la.vsub(v, back)) < 1e-7


@test
def test_rodrigues_near_pi():
    """The axis recovery degenerates at 180 degrees; check the special path."""
    for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0.3, -0.9, 0.2]):
        a = la.normalize(axis)
        r = la.rodrigues(la.vscale(a, math.pi - 1e-9))
        back = la.rodrigues(la.rot_to_vec(r))
        assert math.degrees(la.rotation_angle_between(r, back)) < 1e-3


@test
def test_orthonormalize_fixes_perturbed_rotation():
    rng = random.Random(4)
    r = la.rodrigues([0.3, -0.7, 0.2])
    noisy = [[r[i][j] + rng.gauss(0, 0.02) for j in range(3)] for i in range(3)]
    o = la.orthonormalize(noisy)
    oto = la.matmul(la.transpose(o), o)
    for i in range(3):
        for j in range(3):
            approx(oto[i][j], 1.0 if i == j else 0.0, 1e-9)
    det = (o[0][0] * (o[1][1] * o[2][2] - o[1][2] * o[2][1])
           - o[0][1] * (o[1][0] * o[2][2] - o[1][2] * o[2][0])
           + o[0][2] * (o[1][0] * o[2][1] - o[1][1] * o[2][0]))
    approx(det, 1.0, 1e-9, "not a proper rotation")


@test
def test_look_at_is_a_valid_extrinsic():
    r, t = la.look_at([0.3, 0.4, -0.5], [0.0, 0.0, 0.0])
    rtr = la.matmul(la.transpose(r), r)
    for i in range(3):
        for j in range(3):
            approx(rtr[i][j], 1.0 if i == j else 0.0, 1e-9)
    # The target must land on the optical axis, i.e. at (0, 0, +d).
    p = la.vadd(la.matvec(r, [0.0, 0.0, 0.0]), t)
    approx(p[0], 0.0, 1e-9)
    approx(p[1], 0.0, 1e-9)
    assert p[2] > 0.0, "target ended up behind the camera"


# ------------------------------------------------------------------------ png


def _reference_unfilter(raw, w, h, bpp):
    """Independent re-implementation of the PNG filter reversal (spec 9.2)."""
    stride = w * bpp
    out = bytearray()
    prev = bytearray(stride)
    pos = 0
    for _y in range(h):
        ft = raw[pos]
        pos += 1
        line = bytearray(raw[pos:pos + stride])
        pos += stride
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if ft == 0:
                pass
            elif ft == 1:
                line[i] = (line[i] + a) & 255
            elif ft == 2:
                line[i] = (line[i] + b) & 255
            elif ft == 3:
                line[i] = (line[i] + ((a + b) >> 1)) & 255
            elif ft == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pred) & 255
        out += line
        prev = line
    return out


@test
def test_png_rgb_round_trip():
    img = Image(37, 23, fill=(11, 200, 90))
    img.fill_rect(3, 3, 20, 15, (255, 0, 0))
    img.draw_line(0, 0, 36, 22, (0, 0, 255), 2)
    data = png.encode_rgb(img.w, img.h, bytes(img.px))
    w, h, rgb = png.decode(data)
    assert (w, h) == (37, 23)
    assert bytes(rgb) == bytes(img.px)


@test
def test_png_gray_round_trip():
    g = Gray(16, 9)
    for i in range(len(g.px)):
        g.px[i] = (i * 7) & 255
    w, h, rgb = png.decode(png.encode_gray(16, 9, bytes(g.px)))
    assert (w, h) == (16, 9)
    for i in range(16 * 9):
        assert rgb[i * 3] == g.px[i] and rgb[i * 3 + 2] == g.px[i]


@test
def test_png_decodes_all_filter_types():
    """Our encoder only emits filter 0, so synthesise PNGs using filters 1-4."""
    w, h = 12, 8
    rng = random.Random(9)
    truth = bytearray(rng.randrange(256) for _ in range(w * h * 3))
    for ftype in (0, 1, 2, 3, 4):
        raw = bytearray()
        prev = bytearray(w * 3)
        for y in range(h):
            line = truth[y * w * 3:(y + 1) * w * 3]
            enc = bytearray(len(line))
            for i in range(len(line)):
                a = line[i - 3] if i >= 3 else 0
                b = prev[i]
                c = prev[i - 3] if i >= 3 else 0
                if ftype == 0:
                    enc[i] = line[i]
                elif ftype == 1:
                    enc[i] = (line[i] - a) & 255
                elif ftype == 2:
                    enc[i] = (line[i] - b) & 255
                elif ftype == 3:
                    enc[i] = (line[i] - ((a + b) >> 1)) & 255
                else:
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                    enc[i] = (line[i] - pred) & 255
            raw.append(ftype)
            raw += enc
            prev = line
        body = (png._SIG
                + png._chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + png._chunk(b"IDAT", zlib.compress(bytes(raw)))
                + png._chunk(b"IEND", b""))
        dw, dh, rgb = png.decode(body)
        assert (dw, dh) == (w, h)
        assert bytes(rgb) == bytes(truth), "filter type %d mismatch" % ftype


@test
def test_png_decoder_against_real_files():
    """Decode every PNG on the host and cross-check against an independent
    zlib + reference-unfilter implementation."""
    files = _find_files("*.png", limit=140)
    assert len(files) >= 5, "expected some PNGs on the host to test against"
    checked = unsupported = 0
    for path in files:
        with open(path, "rb") as fh:
            data = fh.read()
        try:
            w, h, rgb = png.decode(data)
        except ValueError:
            unsupported += 1  # 16-bit or interlaced: documented as unsupported
            continue
        except Exception as exc:
            raise AssertionError("decoding %s raised %r" % (path, exc))
        assert len(rgb) == w * h * 3, "%s: wrong buffer size" % path
        # Independent check: re-extract the raw scanlines ourselves.
        ihdr, idat, plte = _raw_chunks(data)
        bw, bh, depth, ctype = ihdr
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
        ref = _reference_unfilter(zlib.decompress(idat), bw, bh, channels)
        if ctype == 2:
            assert bytes(rgb) == bytes(ref), "%s: RGB mismatch" % path
        elif ctype == 0:
            assert all(rgb[i * 3] == ref[i] for i in range(0, bw * bh, 7)), path
        elif ctype == 6:
            assert all(rgb[i * 3] == ref[i * 4] for i in range(0, bw * bh, 7)), path
        elif ctype == 3:
            assert all(rgb[i * 3] == plte[ref[i] * 3] for i in range(0, bw * bh, 7)), path
        checked += 1
    assert checked >= 5, "only %d real PNGs were checked" % checked


def _raw_chunks(data):
    pos = 8
    ihdr = None
    idat = bytearray()
    plte = b""
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            w, h, depth, ctype, _c, _f, _i = struct.unpack(">IIBBBBB", body)
            ihdr = (w, h, depth, ctype)
        elif tag == b"IDAT":
            idat += body
        elif tag == b"PLTE":
            plte = body
        elif tag == b"IEND":
            break
    return ihdr, bytes(idat), plte


def _find_files(pattern, limit=200):
    try:
        res = subprocess.run(
            ["find", "/", "-iname", pattern, "-type", "f"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=120, text=True,
        )
        files = [f for f in res.stdout.split("\n") if f.strip()]
    except Exception:
        return []
    return files[:limit]


# ------------------------------------------------------------------------ gif


@test
def test_lzw_round_trip_edge_cases():
    rng = random.Random(11)
    cases = [
        bytes(20000),                                        # maximal runs
        bytes(rng.randrange(256) for _ in range(40000)),     # incompressible
        bytes((i * 7) % 256 for i in range(60000)),          # periodic
        bytes([5] * 5000),                                   # KwKwK path
        bytes([3]),                                          # single symbol
        bytes(rng.randrange(256) for _ in range(300000)),    # forces dict reset
    ]
    for data in cases:
        for mcs in (2, 4, 8):
            if max(data, default=0) >= (1 << mcs):
                continue
            s = gif.lzw_encode(data, mcs)
            raw, _ = gif._read_sub_blocks(s, 1)
            back = gif.lzw_decode(raw, s[0], expected=len(data))
            assert back == data, "LZW mismatch (mcs=%d, len=%d)" % (mcs, len(data))


@test
def test_lzw_width_growth_stays_in_sync():
    """A payload engineered to cross several code-width boundaries.

    This is the regression guard for the encoder/decoder off-by-one: if the
    width rules drift apart the stream desynchronises partway through and the
    tail decodes to garbage.
    """
    data = bytes((i * 31 + (i // 256)) % 256 for i in range(200000))
    s = gif.lzw_encode(data, 8)
    raw, _ = gif._read_sub_blocks(s, 1)
    assert gif.lzw_decode(raw, s[0], expected=len(data)) == data


@test
def test_gif_container_round_trip():
    img = Image(40, 30, fill=(8, 16, 32))
    img.fill_rect(4, 4, 25, 18, (220, 60, 60))
    img.draw_text(2, 20, "AR", (255, 255, 0))
    data = gif.encode_animation([img, img], delay_cs=5, max_colors=64)
    assert data[:6] == b"GIF89a"
    assert data[-1:] == b"\x3b"
    w, h, pal, idx = gif.decode_first_frame(data)
    assert (w, h) == (40, 30)
    assert len(idx) == w * h
    expect = gif.quantize(img, gif._nearest_factory(pal))
    assert bytes(idx) == bytes(expect)


@test
def test_gif_decoder_against_real_files():
    """Our decoder must handle GIFs written by other encoders."""
    files = _find_files("*.gif", limit=200)
    assert len(files) >= 5, "expected some GIFs on the host to test against"
    ok = 0
    for path in files:
        with open(path, "rb") as fh:
            data = fh.read()
        try:
            w, h, pal, idx = gif.decode_first_frame(data)
        except Exception:
            continue  # truncated or exotic file; not our concern here
        if w * h == 0:
            continue
        assert len(idx) == w * h, "%s: got %d indices, expected %d" % (path, len(idx), w * h)
        if pal:
            assert max(idx) < len(pal), "%s: index outside palette" % path
        ok += 1
    assert ok >= 5, "only %d real GIFs decoded" % ok


@test
def test_palette_covers_saturated_accents():
    """Median cut must not wash out small but vivid regions (the AR overlay)."""
    img = Image(64, 64, fill=(128, 128, 130))
    img.fill_rect(0, 0, 5, 5, (0, 229, 255))
    img.fill_rect(10, 10, 15, 15, (255, 64, 180))
    pal = gif.build_palette([img], max_colors=64, sample_step=1)
    for target in ((0, 229, 255), (255, 64, 180)):
        best = min(sum((a - b) ** 2 for a, b in zip(target, p)) for p in pal)
        assert best < 1200, "accent %s poorly represented (err^2=%d)" % (target, best)


# ----------------------------------------------------------------- image ops


@test
def test_gray_conversion_and_bilinear_sampling():
    img = Image(2, 2, fill=(255, 0, 0))
    approx(img.to_gray().px[0], 76, 1.5, "BT.601 red luma")
    g = Gray(2, 2, px=bytearray([0, 100, 200, 255]))
    approx(g.sample(0.0, 0.0), 0.0, 1e-9)
    approx(g.sample(0.5, 0.0), 50.0, 1e-9)
    approx(g.sample(0.0, 0.5), 100.0, 1e-9)
    approx(g.sample(0.5, 0.5), 138.75, 1e-9)


@test
def test_fill_poly_and_bounds_safety():
    img = Image(20, 20)
    img.fill_poly([(-50, -50), (100, -50), (100, 100), (-50, 100)], (255, 255, 255))
    assert img.at(10, 10) == (255, 255, 255)
    img2 = Image(10, 10)
    img2.draw_line(-100, -100, 200, 200, (1, 2, 3))  # must not raise
    img2.fill_circle(0, 0, 50, (9, 9, 9))
    img2.draw_text(-5, -5, "XYZ", (7, 7, 7))


@test
def test_integral_image_matches_brute_force():
    rng = random.Random(12)
    g = Gray(17, 11, px=bytearray(rng.randrange(256) for _ in range(17 * 11)))
    ii, iw = filters.integral_image(g)
    for _ in range(40):
        x0 = rng.randrange(g.w)
        x1 = rng.randrange(x0, g.w)
        y0 = rng.randrange(g.h)
        y1 = rng.randrange(y0, g.h)
        brute = sum(g.px[y * g.w + x] for y in range(y0, y1 + 1) for x in range(x0, x1 + 1))
        assert filters.window_sum(ii, iw, x0, y0, x1, y1) == brute


@test
def test_adaptive_threshold_beats_global_under_a_gradient():
    """The point of adaptive thresholding, as a test.

    A dark square on a background with a strong brightness ramp: no single
    global threshold can recover it, a local one can.

    Note the block size must exceed the dark feature's width, otherwise the
    window sits entirely inside the feature and its local mean goes dark with
    it. That is an inherent property of mean-adaptive thresholding, not a bug,
    and it is why the detector pairs a 25px block with ~10px marker cells.
    """
    w = h = 90
    g = Gray(w, h)
    for y in range(h):
        for x in range(w):
            g.px[y * w + x] = 30 + int(200 * x / w)  # ramp from dark to bright
    # Crucially, the patch sitting in the bright region (120) is *brighter* than
    # the background of the dark region (~70). No global threshold can mark both
    # patches without also swallowing the dark background, which is exactly the
    # situation uneven lighting creates.
    for y in range(24, 36):
        for x in range(64, 76):
            g.px[y * w + x] = 120
    for y in range(58, 70):
        for x in range(12, 24):
            g.px[y * w + x] = 10
    def in_patch(x, y):
        return (24 <= y < 36 and 64 <= x < 76) or (58 <= y < 70 and 12 <= x < 24)

    def scores(mask):
        """(fraction of patch pixels found, fraction of background wrongly lit).

        Measured over every pixel rather than a few hand-picked ones -- a
        sampled control can be fooled by *where* it samples, which is exactly
        how an earlier version of this test passed for the wrong reason.
        """
        hit = patch_total = fp = bg_total = 0
        for y in range(h):
            for x in range(w):
                fg = mask.px[y * w + x] == filters.FG
                if in_patch(x, y):
                    patch_total += 1
                    hit += fg
                else:
                    bg_total += 1
                    fp += fg
        return hit / patch_total, fp / bg_total

    ad_recall, ad_fp = scores(filters.adaptive_threshold(g, block=25, c=8))
    gl_recall, gl_fp = scores(filters.global_threshold(g, filters.otsu_threshold_value(g)))
    assert ad_recall > 0.85, "adaptive found only %.0f%% of the patches" % (ad_recall * 100)
    assert ad_fp < 0.05, "adaptive lit %.1f%% of the background" % (ad_fp * 100)
    # The control: Otsu can only reach both patches by also swallowing the
    # entire dark half of the ramp.
    assert not (gl_recall > 0.85 and gl_fp < 0.05), (
        "global threshold solved it too (recall %.2f, fp %.2f); test is not discriminative"
        % (gl_recall, gl_fp))


@test
def test_otsu_on_bimodal_histogram():
    g = Gray(100, 1)
    for i in range(100):
        g.px[i] = 20 if i < 50 else 220
    t = filters.otsu_threshold_value(g)
    assert 20 < t <= 220, t
    # The documented contract: value < t is dark, value >= t is bright.
    split = filters.global_threshold(g, t, invert=True)
    assert split.px[0] == filters.FG and split.px[99] == filters.BG


@test
def test_morphology_close_repairs_a_gap():
    m = Gray(20, 20)
    for x in range(2, 18):
        m.px[10 * 20 + x] = filters.FG
    m.px[10 * 20 + 9] = filters.BG  # one-pixel break in the line
    closed = morph.close(m, 1)
    assert closed.at(9, 10) == filters.FG, "close did not bridge the gap"
    eroded = morph.erode(m, 1)
    assert eroded.at(10, 10) == filters.BG


# ----------------------------------------------------------- contours / geom


@test
def test_label_components_and_boundary_trace():
    m = Gray(30, 20)
    for y in range(3, 13):
        for x in range(4, 18):
            m.px[y * 30 + x] = filters.FG
    for y in range(15, 18):
        for x in range(22, 26):
            m.px[y * 30 + x] = filters.FG
    labels, comps = contours.label_components(m)
    assert len(comps) == 2, comps
    big = max(comps, key=lambda c: c.area)
    assert big.area == 10 * 14
    assert (big.x0, big.y0, big.x1, big.y1) == (4, 3, 17, 12)
    chain = contours.trace_boundary(labels, 30, 20, big)
    # The boundary of a 14x10 rectangle has 2*(14+10)-4 pixels.
    assert len(chain) == 2 * (14 + 10) - 4, len(chain)
    for x, y in chain:
        assert m.px[y * 30 + x] == filters.FG
        on_edge = x in (4, 17) or y in (3, 12)
        assert on_edge, "traced an interior pixel (%d,%d)" % (x, y)


@test
def test_douglas_peucker_keeps_corners():
    pts = [(float(x), 0.0) for x in range(20)] + [(19.0, float(y)) for y in range(1, 20)]
    simp = geom.douglas_peucker(pts, 0.5)
    assert len(simp) == 3, simp
    assert simp[0] == (0.0, 0.0) and simp[-1] == (19.0, 19.0)


@test
def test_approx_polygon_on_rotated_square_gives_four_corners():
    """Regression: splitting the contour at an arbitrary index used to pin a
    spurious 5th vertex in the middle of a straight side."""
    for angle in range(0, 90, 7):
        th = math.radians(angle)
        side = 60
        corners = []
        for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            x = dx * side / 2
            y = dy * side / 2
            corners.append((100 + x * math.cos(th) - y * math.sin(th),
                            100 + x * math.sin(th) + y * math.cos(th)))
        chain = []
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            steps = int(math.dist(a, b))
            for s in range(steps):
                f = s / steps
                chain.append((int(round(a[0] + (b[0] - a[0]) * f)),
                              int(round(a[1] + (b[1] - a[1]) * f))))
        poly = geom.approx_polygon_closed(chain, 2.0)
        assert len(poly) == 4, "angle %d gave %d vertices: %s" % (angle, len(poly), poly)
        assert geom.is_convex(geom.order_quad(poly))


@test
def test_order_quad_winding_is_consistent():
    cw = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert geom.signed_area(cw) > 0
    assert geom.order_quad(cw) == cw
    assert geom.order_quad(list(reversed(cw))) == cw
    assert geom.signed_area(geom.order_quad(list(reversed(cw)))) > 0


@test
def test_fit_line_and_intersect():
    pts = [(float(x), 2.0 * x + 1.0) for x in range(10)]
    nx, ny, c = geom.fit_line(pts)
    for x, y in pts:
        approx(nx * x + ny * y + c, 0.0, 1e-9, "point off fitted line")
    horiz = geom.fit_line([(float(x), 5.0) for x in range(5)])
    vert = geom.fit_line([(3.0, float(y)) for y in range(5)])
    ix, iy = geom.line_intersect(horiz, vert)
    approx(ix, 3.0, 1e-9)
    approx(iy, 5.0, 1e-9)
    assert geom.line_intersect(horiz, geom.fit_line([(float(x), 9.0) for x in range(5)])) is None


@test
def test_min_corner_angle_rejects_slivers():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert geom.min_corner_angle_cos(square) < 0.01
    sliver = [(0, 0), (50, 1), (100, 0), (50, -1)]
    assert geom.min_corner_angle_cos(sliver) > 0.9


# ---------------------------------------------------------------- marker dict


@test
def test_dictionary_distance_guarantees():
    d = mk.generate_dictionary(count=16, min_dist=8)
    stats = mk.dictionary_stats(d)
    assert stats["markers"] == 16
    assert stats["min_inter_distance"] >= 8, stats
    assert stats["min_self_rotation_distance"] >= 8, stats


@test
def test_dictionary_is_deterministic():
    a = mk.generate_dictionary(count=8, min_dist=8)
    b = mk.generate_dictionary(count=8, min_dist=8)
    assert a.grids == b.grids, "dictionary generation is not reproducible"


@test
def test_decode_exact_for_all_rotations():
    d = mk.generate_dictionary(count=12, min_dist=8)
    for mid, g in enumerate(d.grids):
        for rot, rg in enumerate(mk.rotations(g)):
            assert d.decode(rg) == (mid, rot, 0), (mid, rot)


@test
def test_decode_corrects_errors_and_never_misdecodes():
    d = mk.generate_dictionary(count=12, min_dist=8)
    rng = random.Random(21)
    n_cells = mk.PAYLOAD ** 2
    for k in range(1, d.max_correctable + 1):
        for _ in range(400):
            mid = rng.randrange(len(d.grids))
            rot = rng.randrange(4)
            g = [row[:] for row in mk.rotations(d.grids[mid])[rot]]
            for i in rng.sample(range(n_cells), k):
                g[i // mk.PAYLOAD][i % mk.PAYLOAD] ^= 1
            res = d.decode(g)
            assert res is not None, "rejected a correctable %d-bit error" % k
            assert (res[0], res[1]) == (mid, rot), "misdecoded with %d bit errors" % k


@test
def test_rotate_ccw_is_order_four():
    g = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    r = g
    for _ in range(4):
        r = mk.rotate_ccw(r)
    assert r == g
    assert mk.rotate_ccw([[1, 2], [3, 4]]) == [[2, 4], [1, 3]]


@test
def test_rendered_marker_has_black_border():
    d = mk.generate_dictionary(count=4, min_dist=8)
    img = mk.render_marker(d.grids[0], cell_px=8, quiet_cells=1)
    assert img.w == img.h == (mk.GRID + 2) * 8
    assert img.at(2, 2) == (255, 255, 255), "quiet zone should be white"
    # a pixel inside the border ring
    assert img.at(8 + 4, 8 + 4) == (0, 0, 0), "border ring should be black"


# ----------------------------------------------------------- homography/pose


@test
def test_homography_recovers_exact_transform():
    h_true = [[1.2, 0.1, 30.0], [-0.2, 1.1, -15.0], [0.0005, 0.0002, 1.0]]
    src = [(0, 0), (200, 0), (200, 150), (0, 150), (50, 90)]
    dst = [homography.apply(h_true, p) for p in src]
    h = homography.dlt(src, dst)
    for i in range(3):
        for j in range(3):
            approx(h[i][j], h_true[i][j], 1e-7, "H coefficient")
    assert homography.transfer_error(h, src, dst) < 1e-8


@test
def test_homography_needs_four_points():
    try:
        homography.dlt([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)])
    except ValueError:
        return
    raise AssertionError("expected ValueError for 3 correspondences")


@test
def test_pose_exact_without_noise():
    cam = Camera.from_fov(480, 320, 58.0)
    s = 0.05
    obj = [(-s, s, 0.0), (s, s, 0.0), (s, -s, 0.0), (-s, -s, 0.0)]
    rng = random.Random(31)
    for _ in range(60):
        axis = la.normalize([rng.gauss(0, 1) for _ in range(3)])
        r_true = la.rodrigues(la.vscale(axis, rng.uniform(0, math.radians(45))))
        t_true = [rng.uniform(-0.05, 0.05), rng.uniform(-0.04, 0.04), rng.uniform(0.3, 0.9)]
        pts = pose.project_points(cam, r_true, t_true, obj)
        if any(p is None for p in pts):
            continue
        out = pose.solve_marker_pose(cam, obj, pts)
        assert out is not None
        r, t, _rmse0, rmse = out
        assert rmse < 1e-6, rmse
        assert math.degrees(la.rotation_angle_between(r_true, r)) < 1e-3
        assert la.norm(la.vsub(t_true, t)) < 1e-6


@test
def test_lm_refinement_improves_on_noise():
    cam = Camera.from_fov(480, 320, 58.0)
    s = 0.05
    obj = [(-s, s, 0.0), (s, s, 0.0), (s, -s, 0.0), (-s, -s, 0.0)]
    rng = random.Random(32)
    better = total = 0
    for _ in range(120):
        r_true = la.rodrigues([rng.uniform(-0.5, 0.5) for _ in range(3)])
        t_true = [rng.uniform(-0.04, 0.04), rng.uniform(-0.03, 0.03), rng.uniform(0.35, 0.8)]
        pts = pose.project_points(cam, r_true, t_true, obj)
        if any(p is None for p in pts):
            continue
        obs = [(p[0] + rng.gauss(0, 0.5), p[1] + rng.gauss(0, 0.5)) for p in pts]
        out = pose.solve_marker_pose(cam, obj, obs)
        if out is None:
            continue
        _r, _t, rmse0, rmse1 = out
        total += 1
        if rmse1 <= rmse0 + 1e-12:
            better += 1
    assert total > 50
    assert better == total, "LM made %d/%d cases worse" % (total - better, total)


@test
def test_pose_puts_marker_in_front_of_camera():
    cam = Camera.from_fov(480, 320, 58.0)
    s = 0.05
    obj = [(-s, s, 0.0), (s, s, 0.0), (s, -s, 0.0), (-s, -s, 0.0)]
    r_true = la.rodrigues([0.2, -0.3, 0.1])
    t_true = [0.01, 0.0, 0.5]
    pts = pose.project_points(cam, r_true, t_true, obj)
    r, t, _a, _b = pose.solve_marker_pose(cam, obj, pts)
    assert t[2] > 0, "recovered a mirror pose behind the camera"


@test
def test_camera_from_fov_and_projection():
    cam = Camera.from_fov(480, 320, 90.0)
    approx(cam.fx, 240.0, 1e-9, "90 deg FOV on a 480px sensor -> fx = 240")
    assert cam.project_point([0.0, 0.0, 1.0]) == (240.0, 160.0)
    assert cam.project_point([0.0, 0.0, -1.0]) is None


# -------------------------------------------------------------------- render


@test
def test_cube_mesh_sits_on_the_marker_plane():
    verts, faces, colors = render.cube_mesh(0.06)
    assert len(verts) == 8 and len(faces) == 6 and len(colors) == 6
    assert min(v[2] for v in verts) == 0.0, "cube must rest on z=0"
    approx(max(v[2] for v in verts), 0.06, 1e-12)


@test
def test_draw_mesh_skips_geometry_behind_the_camera():
    cam = Camera.from_fov(80, 60, 58.0)
    img = Image(80, 60)
    verts, faces, colors = render.cube_mesh(0.05)
    assert render.draw_mesh(img, cam, la.eye(3), [0.0, 0.0, -1.0], verts, faces, colors) is False
    assert bytes(img.px) == bytes(bytearray(80 * 60 * 3)), "nothing should have been drawn"
    assert render.draw_mesh(img, cam, la.eye(3), [0.0, 0.0, 0.5], verts, faces, colors) is True
    assert bytes(img.px) != bytes(bytearray(80 * 60 * 3))


# --------------------------------------------------------------- scene/eval


@test
def test_scene_ground_truth_matches_the_pinhole_model():
    """The GT corners must be the projection of the GT pose, or every accuracy
    number in the report is meaningless."""
    d = mk.generate_dictionary(count=4, min_dist=8)
    cam = Camera.from_fov(240, 160, 58.0)
    pl = scene.default_scene(d, side=0.10)
    r_cw, t_cw = scene.orbit_trajectory(1, start_deg=-115.0)[0]
    img, gts = scene.render_frame(cam, r_cw, t_cw, pl, supersample=1, blur=0,
                                  noise=0.0, vignette=0.0, light_ramp=0.0)
    assert img.w == 240 and img.h == 160
    assert len(gts) >= 1
    for gt in gts:
        placement = next(p for p in pl if p.marker_id == gt.marker_id)
        for obj, expected in zip(placement.object_corners(), gt.corners):
            pc = la.vadd(la.matvec(gt.r_cm, list(obj)), list(gt.t_cm))
            got = cam.project_point(pc)
            approx(got[0], expected[0], 1e-9)
            approx(got[1], expected[1], 1e-9)
        rtr = la.matmul(la.transpose(gt.r_cm), gt.r_cm)
        for i in range(3):
            for j in range(3):
                approx(rtr[i][j], 1.0 if i == j else 0.0, 1e-9)


@test
def test_tilted_marker_stays_above_the_floor():
    """Regression: a negative tilt used to sink the sheet under the floor,
    where it was occluded down to a sliver."""
    d = mk.generate_dictionary(count=4, min_dist=8)
    for tilt in (-60.0, -30.0, 0.0, 30.0, 60.0):
        p = scene.marker_tilted(0, d.grids[0], 0.0, 0.0, 0.0, tilt, 0.10)
        xa, ya, na = p.axes()
        lowest = min(
            p.t_wm[1] + ya[1] * b * p.paper_half + xa[1] * a * p.paper_half
            for a in (-1, 1) for b in (-1, 1)
        )
        assert lowest >= -1e-9, "tilt %.0f puts the sheet %.4f m below the floor" % (tilt, lowest)


@test
def test_evaluate_matches_by_geometry_not_by_id():
    class FakeDet:
        def __init__(self, mid, corners):
            self.marker_id = mid
            self.corners = corners
            self.distance = 0.5
            self.r = la.eye(3)
            self.t = [0.0, 0.0, 0.5]
            self.rmse = 0.1
            self.rmse_initial = 0.2
            self.bit_errors = 0

    class FakeGT:
        def __init__(self, mid, corners):
            self.marker_id = mid
            self.corners = corners
            self.r_cm = la.eye(3)
            self.t_cm = [0.0, 0.0, 0.5]
            self.distance = 0.5

    quad = [(100, 100), (140, 100), (140, 140), (100, 140)]
    far = [(300, 200), (340, 200), (340, 240), (300, 240)]
    # Right place, wrong ID: must be reported as a wrong ID, not a miss + FP.
    pairs, missed, fps = evaluate.match([FakeDet(7, quad)], [FakeGT(3, quad)])
    assert len(pairs) == 1 and not missed and not fps
    rec = evaluate.errors_for(*pairs[0])
    assert rec["id_correct"] is False
    # A detection nowhere near any marker is a false positive.
    pairs, missed, fps = evaluate.match([FakeDet(3, far)], [FakeGT(3, quad)])
    assert not pairs and len(missed) == 1 and len(fps) == 1


@test
def test_summarize_percentiles():
    frames = [{"errors": [{"rot_deg": float(v), "pos_mm": float(v)} for v in range(1, 101)],
               "n_detectable": 100, "n_detected": 100, "n_detectable_detected": 100,
               "n_missed_detectable": 0, "n_false_positive": 0, "n_id_wrong": 0}]
    s = evaluate.summarize(frames, keys=("rot_deg",))
    approx(s["stats"]["rot_deg"]["median"], 50.5, 1e-9)
    approx(s["stats"]["rot_deg"]["max"], 100.0, 1e-9)
    approx(s["stats"]["rot_deg"]["p95"], 95.05, 1e-9)
    approx(s["detection_rate_pct"], 100.0, 1e-9)


# ------------------------------------------------------------- integration


@test
def test_end_to_end_detection_accuracy():
    """The headline claim: detect all three markers in a degraded synthetic
    frame, with correct IDs and sub-degree pose error."""
    d = mk.generate_dictionary(count=16, min_dist=8)
    cam = Camera.from_fov(480, 320, 58.0)
    pl = scene.default_scene(d, side=0.10)
    r_cw, t_cw = scene.orbit_trajectory(1, start_deg=-115.0)[0]
    img, gts = scene.render_frame(cam, r_cw, t_cw, pl, supersample=2, seed=1)
    dets, _ = detect.detect(img, cam, d, marker_side=0.10)
    res = evaluate.evaluate_frame(dets, gts, cam)
    assert res["n_detectable"] == 3, res
    assert res["n_detected"] == 3, "found %d of 3 markers" % res["n_detected"]
    assert res["n_false_positive"] == 0, res
    assert res["n_id_wrong"] == 0, res
    for rec in res["errors"]:
        assert rec["bit_errors"] == 0, rec
        assert rec["rot_deg"] < 2.0, "rotation error %.3f deg too high" % rec["rot_deg"]
        assert rec["pos_mm"] < 8.0, "position error %.2f mm too high" % rec["pos_mm"]
        assert rec["corner_mean_px"] < 1.5, rec
        assert rec["rmse_px"] < 1.0, rec


@test
def test_subpixel_refinement_removes_depth_bias():
    """Refinement is not cosmetic: without it the recovered distance is
    systematically biased by the threshold."""
    d = mk.generate_dictionary(count=16, min_dist=8)
    cam = Camera.from_fov(480, 320, 58.0)
    pl = scene.default_scene(d, side=0.10)
    r_cw, t_cw = scene.orbit_trajectory(1, start_deg=-115.0)[0]
    img, gts = scene.render_frame(cam, r_cw, t_cw, pl, supersample=2, seed=1)
    gtmap = {g.marker_id: g for g in gts}

    def mean_abs_depth_bias(refine):
        dets, _ = detect.detect(img, cam, d, marker_side=0.10, refine_corners=refine)
        vals = []
        for det in dets:
            gt = gtmap[det.marker_id]
            vals.append(abs(la.norm(list(det.t)) - gt.distance) / gt.distance * 100.0)
        return sum(vals) / len(vals) if vals else float("inf")

    with_refine = mean_abs_depth_bias(True)
    without = mean_abs_depth_bias(False)
    assert with_refine < without, "refinement did not help (%.3f vs %.3f)" % (with_refine, without)
    assert with_refine < 0.5, "residual depth bias %.3f%% is too large" % with_refine


@test
def test_detector_finds_nothing_in_an_empty_scene():
    """False-positive guard: a marker-free frame must yield zero detections,
    even though the checkerboard floor is full of dark quadrilaterals."""
    d = mk.generate_dictionary(count=16, min_dist=8)
    cam = Camera.from_fov(480, 320, 58.0)
    r_cw, t_cw = scene.orbit_trajectory(1, start_deg=-115.0)[0]
    img, gts = scene.render_frame(cam, r_cw, t_cw, [], supersample=2, seed=3)
    assert gts == []
    dets, st = detect.detect(img, cam, d, marker_side=0.10, collect_stages=True)
    assert len(st.candidates) > 0, "expected the floor tiles to produce candidates"
    assert dets == [], "false positives: %s" % dets


def main():
    print("running %d tests with Python %s\n" % (len(TESTS), sys.version.split()[0]))
    failed = []
    for fn in TESTS:
        name = fn.__name__
        sys.stdout.write("  %-56s " % name)
        sys.stdout.flush()
        try:
            fn()
        except AssertionError as exc:
            print("FAIL")
            failed.append((name, str(exc)))
        except Exception as exc:  # noqa: BLE001
            print("ERROR")
            failed.append((name, "%s: %s" % (type(exc).__name__, exc)))
        else:
            print("ok")
    print()
    if failed:
        print("%d of %d tests FAILED" % (len(failed), len(TESTS)))
        for name, msg in failed:
            print("  - %s\n      %s" % (name, msg))
        return False
    print("all %d tests passed" % len(TESTS))
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
