"""Classic image filters, hand rolled.

The important one is :func:`adaptive_threshold`: markers have to survive
vignetting and uneven lighting, and a single global threshold simply does not
work for that. We compute a summed-area table once and then every pixel gets
compared against the mean of its local window in O(1).
"""

from .image import Gray, Image

FG = 255  # binary foreground ("ink") value
BG = 0


# ----------------------------------------------------------- integral images


def integral_image(g):
    """Summed-area table with a zero row/column border.

    ``ii[(y+1)*(w+1) + (x+1)]`` is the sum of all pixels in ``[0..y, 0..x]``,
    so a window sum is four lookups regardless of window size.
    """
    w, h = g.w, g.h
    iw = w + 1
    ii = [0] * (iw * (h + 1))
    px = g.px
    for y in range(h):
        row_sum = 0
        base_src = y * w
        base_cur = (y + 1) * iw
        base_prev = y * iw
        for x in range(w):
            row_sum += px[base_src + x]
            ii[base_cur + x + 1] = ii[base_prev + x + 1] + row_sum
    return ii, iw


def window_sum(ii, iw, x0, y0, x1, y1):
    """Inclusive window sum using the summed-area table."""
    return (
        ii[(y1 + 1) * iw + (x1 + 1)]
        - ii[y0 * iw + (x1 + 1)]
        - ii[(y1 + 1) * iw + x0]
        + ii[y0 * iw + x0]
    )


# ---------------------------------------------------------------- thresholds


def adaptive_threshold(g, block=25, c=7, invert=True):
    """Mean-adaptive threshold.

    A pixel becomes foreground when it is at least ``c`` levels *darker* than
    the mean of its ``block x block`` neighbourhood (``invert=True``, which is
    what we want since markers are dark ink on a light background).
    """
    w, h = g.w, g.h
    ii, iw = integral_image(g)
    r = block // 2
    out = bytearray(w * h)
    px = g.px
    for y in range(h):
        y0 = y - r
        y1 = y + r
        if y0 < 0:
            y0 = 0
        if y1 > h - 1:
            y1 = h - 1
        rowbase = y * w
        for x in range(w):
            x0 = x - r
            x1 = x + r
            if x0 < 0:
                x0 = 0
            if x1 > w - 1:
                x1 = w - 1
            area = (x1 - x0 + 1) * (y1 - y0 + 1)
            s = (
                ii[(y1 + 1) * iw + (x1 + 1)]
                - ii[y0 * iw + (x1 + 1)]
                - ii[(y1 + 1) * iw + x0]
                + ii[y0 * iw + x0]
            )
            mean = s / area
            v = px[rowbase + x]
            if invert:
                out[rowbase + x] = FG if v < mean - c else BG
            else:
                out[rowbase + x] = FG if v > mean + c else BG
    return Gray(w, h, px=out)


def global_threshold(g, t=128, invert=True):
    out = bytearray(g.w * g.h)
    for i, v in enumerate(g.px):
        dark = v < t
        out[i] = FG if (dark if invert else not dark) else BG
    return Gray(g.w, g.h, px=out)


def otsu_threshold_value(g):
    """Otsu's method: pick the threshold maximising between-class variance.

    Contract: the returned value ``t`` splits the histogram so that pixels with
    ``value < t`` are the dark class and ``value >= t`` the bright class. The
    search finds the last level belonging to the dark class, so the returned
    value is that level plus one -- keeping the comparison consistent with
    :func:`global_threshold` and with the cell sampling in ``detect``.
    """
    hist = [0] * 256
    for v in g.px:
        hist[v] += 1
    total = len(g.px)
    sum_all = sum(i * hist[i] for i in range(256))
    w_b = 0
    sum_b = 0
    best_var = -1.0
    best_t = 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var = var
            best_t = t
    return best_t + 1


# -------------------------------------------------------------------- blurs


def box_blur_gray(g, radius=1):
    """Separable box blur on a single channel."""
    w, h = g.w, g.h
    src = g.px
    tmp = bytearray(w * h)
    k = 2 * radius + 1
    for y in range(h):
        base = y * w
        for x in range(w):
            x0 = x - radius
            x1 = x + radius
            if x0 < 0:
                x0 = 0
            if x1 > w - 1:
                x1 = w - 1
            s = 0
            for xx in range(x0, x1 + 1):
                s += src[base + xx]
            tmp[base + x] = s // (x1 - x0 + 1)
    out = bytearray(w * h)
    for y in range(h):
        y0 = y - radius
        y1 = y + radius
        if y0 < 0:
            y0 = 0
        if y1 > h - 1:
            y1 = h - 1
        base = y * w
        cnt = y1 - y0 + 1
        for x in range(w):
            s = 0
            for yy in range(y0, y1 + 1):
                s += tmp[yy * w + x]
            out[base + x] = s // cnt
        _ = k
    return Gray(w, h, px=out)


def blur_rgb(img, radius=1):
    """Separable box blur over RGB -- used to fake lens softness in the
    synthetic scene so the detector has to cope with blurry edges."""
    w, h = img.w, img.h
    src = img.px
    tmp = bytearray(w * h * 3)
    for y in range(h):
        rb = y * w
        for x in range(w):
            x0 = max(0, x - radius)
            x1 = min(w - 1, x + radius)
            n = x1 - x0 + 1
            r = g = b = 0
            for xx in range(x0, x1 + 1):
                i = (rb + xx) * 3
                r += src[i]
                g += src[i + 1]
                b += src[i + 2]
            o = (rb + x) * 3
            tmp[o] = r // n
            tmp[o + 1] = g // n
            tmp[o + 2] = b // n
    out = bytearray(w * h * 3)
    for y in range(h):
        y0 = max(0, y - radius)
        y1 = min(h - 1, y + radius)
        n = y1 - y0 + 1
        for x in range(w):
            r = g = b = 0
            for yy in range(y0, y1 + 1):
                i = (yy * w + x) * 3
                r += tmp[i]
                g += tmp[i + 1]
                b += tmp[i + 2]
            o = (y * w + x) * 3
            out[o] = r // n
            out[o + 1] = g // n
            out[o + 2] = b // n
    return Image(w, h, px=out)


# -------------------------------------------------------------------- edges


def sobel_magnitude(g):
    """Sobel gradient magnitude, clamped to 8 bits. Diagnostic/visualisation
    only -- the detector itself works off the binarised image."""
    w, h = g.w, g.h
    src = g.px
    out = bytearray(w * h)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            i = y * w + x
            tl, t, tr = src[i - w - 1], src[i - w], src[i - w + 1]
            l, r = src[i - 1], src[i + 1]
            bl, b, br = src[i + w - 1], src[i + w], src[i + w + 1]
            gx = (tr + 2 * r + br) - (tl + 2 * l + bl)
            gy = (bl + 2 * b + br) - (tl + 2 * t + tr)
            m = int((gx * gx + gy * gy) ** 0.5) >> 2
            out[i] = 255 if m > 255 else m
    return Gray(w, h, px=out)
