"""Binary morphology on :class:`holo.image.Gray` masks (0 / 255).

Only what the detector actually needs: a square structuring element, plus
``close`` to repair marker borders broken by blur or noise.
"""

from .filters import BG, FG
from .image import Gray


def _morph(mask, radius, erode):
    w, h = mask.w, mask.h
    src = mask.px
    # Separable: a square structuring element is the Minkowski sum of a
    # horizontal and a vertical segment, so two 1-D passes suffice.
    tmp = bytearray(w * h)
    for y in range(h):
        base = y * w
        for x in range(w):
            x0 = x - radius
            x1 = x + radius
            if x0 < 0:
                x0 = 0
            if x1 > w - 1:
                x1 = w - 1
            if erode:
                v = FG
                for xx in range(x0, x1 + 1):
                    if src[base + xx] == BG:
                        v = BG
                        break
            else:
                v = BG
                for xx in range(x0, x1 + 1):
                    if src[base + xx] == FG:
                        v = FG
                        break
            tmp[base + x] = v
    out = bytearray(w * h)
    for y in range(h):
        y0 = y - radius
        y1 = y + radius
        if y0 < 0:
            y0 = 0
        if y1 > h - 1:
            y1 = h - 1
        base = y * w
        for x in range(w):
            if erode:
                v = FG
                for yy in range(y0, y1 + 1):
                    if tmp[yy * w + x] == BG:
                        v = BG
                        break
            else:
                v = BG
                for yy in range(y0, y1 + 1):
                    if tmp[yy * w + x] == FG:
                        v = FG
                        break
            out[base + x] = v
    return Gray(w, h, px=out)


def erode(mask, radius=1):
    return _morph(mask, radius, True)


def dilate(mask, radius=1):
    return _morph(mask, radius, False)


def close(mask, radius=1):
    """Dilate then erode -- fills pinholes and reconnects thin breaks."""
    return erode(dilate(mask, radius), radius)


def open_(mask, radius=1):
    """Erode then dilate -- removes salt noise."""
    return dilate(erode(mask, radius), radius)
