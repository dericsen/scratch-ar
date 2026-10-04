"""Raster image containers and drawing primitives.

``Image`` holds 8-bit RGB in a flat ``bytearray``; ``Gray`` holds single-channel
luminance. Everything (lines, polygon fill, alpha blending, text) is rasterised
by hand -- including a 5x7 bitmap font defined at the bottom of this module,
since there is no font library available.
"""

from . import png

# --------------------------------------------------------------------- 5x7 font
# Each glyph is 7 rows of 5 columns. '#' = ink, anything else = transparent.
_FONT_SRC = {
    "0": ".###.|#...#|#...#|#...#|#...#|#...#|.###.",
    "1": "..#..|.##..|..#..|..#..|..#..|..#..|.###.",
    "2": ".###.|#...#|....#|...#.|..#..|.#...|#####",
    "3": "#####|...#.|..#..|...#.|....#|#...#|.###.",
    "4": "...#.|..##.|.#.#.|#..#.|#####|...#.|...#.",
    "5": "#####|#....|####.|....#|....#|#...#|.###.",
    "6": "..##.|.#...|#....|####.|#...#|#...#|.###.",
    "7": "#####|....#|...#.|..#..|.#...|.#...|.#...",
    "8": ".###.|#...#|#...#|.###.|#...#|#...#|.###.",
    "9": ".###.|#...#|#...#|.####|....#|...#.|.##..",
    "A": "..#..|.#.#.|#...#|#...#|#####|#...#|#...#",
    "B": "####.|#...#|#...#|####.|#...#|#...#|####.",
    "C": ".###.|#...#|#....|#....|#....|#...#|.###.",
    "D": "####.|#...#|#...#|#...#|#...#|#...#|####.",
    "E": "#####|#....|#....|####.|#....|#....|#####",
    "F": "#####|#....|#....|####.|#....|#....|#....",
    "G": ".###.|#...#|#....|#.###|#...#|#...#|.###.",
    "H": "#...#|#...#|#...#|#####|#...#|#...#|#...#",
    "I": ".###.|..#..|..#..|..#..|..#..|..#..|.###.",
    "J": "..###|...#.|...#.|...#.|...#.|#..#.|.##..",
    "K": "#...#|#..#.|#.#..|##...|#.#..|#..#.|#...#",
    "L": "#....|#....|#....|#....|#....|#....|#####",
    "M": "#...#|##.##|#.#.#|#...#|#...#|#...#|#...#",
    "N": "#...#|##..#|#.#.#|#..##|#...#|#...#|#...#",
    "O": ".###.|#...#|#...#|#...#|#...#|#...#|.###.",
    "P": "####.|#...#|#...#|####.|#....|#....|#....",
    "Q": ".###.|#...#|#...#|#...#|#.#.#|#..#.|.##.#",
    "R": "####.|#...#|#...#|####.|#.#..|#..#.|#...#",
    "S": ".###.|#...#|#....|.###.|....#|#...#|.###.",
    "T": "#####|..#..|..#..|..#..|..#..|..#..|..#..",
    "U": "#...#|#...#|#...#|#...#|#...#|#...#|.###.",
    "V": "#...#|#...#|#...#|#...#|#...#|.#.#.|..#..",
    "W": "#...#|#...#|#...#|#.#.#|#.#.#|##.##|#...#",
    "X": "#...#|#...#|.#.#.|..#..|.#.#.|#...#|#...#",
    "Y": "#...#|#...#|.#.#.|..#..|..#..|..#..|..#..",
    "Z": "#####|....#|...#.|..#..|.#...|#....|#####",
    " ": ".....|.....|.....|.....|.....|.....|.....",
    ".": ".....|.....|.....|.....|.....|.##..|.##..",
    ",": ".....|.....|.....|.....|.##..|.##..|.#...",
    ":": ".....|.##..|.##..|.....|.##..|.##..|.....",
    "-": ".....|.....|.....|#####|.....|.....|.....",
    "+": ".....|..#..|..#..|#####|..#..|..#..|.....",
    "/": "....#|....#|...#.|..#..|.#...|#....|#....",
    "%": "#...#|#..#.|...#.|..#..|.#...|.#..#|#...#",
    "(": "..##.|.#...|.#...|.#...|.#...|.#...|..##.",
    ")": ".##..|...#.|...#.|...#.|...#.|...#.|.##..",
    "[": ".###.|.#...|.#...|.#...|.#...|.#...|.###.",
    "]": ".###.|...#.|...#.|...#.|...#.|...#.|.###.",
    "=": ".....|.....|#####|.....|#####|.....|.....",
    "#": ".#.#.|.#.#.|#####|.#.#.|#####|.#.#.|.#.#.",
    "*": ".....|#.#.#|.###.|#####|.###.|#.#.#|.....",
    "!": "..#..|..#..|..#..|..#..|..#..|.....|..#..",
    "?": ".###.|#...#|....#|...#.|..#..|.....|..#..",
    "<": "...#.|..#..|.#...|#....|.#...|..#..|...#.",
    ">": ".#...|..#..|...#.|....#|...#.|..#..|.#...",
    "_": ".....|.....|.....|.....|.....|.....|#####",
    "^": ".###.|.#.#.|.###.|.....|.....|.....|.....",  # degree sign
}

GLYPH_W, GLYPH_H = 5, 7


def _build_font():
    out = {}
    for ch, spec in _FONT_SRC.items():
        rows = spec.split("|")
        assert len(rows) == GLYPH_H, ch
        out[ch] = [[c == "#" for c in row] for row in rows]
    return out


FONT = _build_font()


def text_width(s, scale=1, spacing=1):
    return len(s) * (GLYPH_W + spacing) * scale - spacing * scale


def text_height(scale=1):
    return GLYPH_H * scale


# --------------------------------------------------------------------- Gray


class Gray:
    """Single channel 8-bit image."""

    __slots__ = ("w", "h", "px")

    def __init__(self, w, h, fill=0, px=None):
        self.w = w
        self.h = h
        self.px = px if px is not None else bytearray([fill]) * (w * h)

    def at(self, x, y):
        return self.px[y * self.w + x]

    def sample(self, x, y):
        """Bilinear sample at a real-valued position, clamped at the border.

        Subpixel edge refinement lives or dies on this: nearest-neighbour
        sampling would quantise the intensity profile and destroy exactly the
        fractional-pixel information we are trying to measure.
        """
        if x < 0.0:
            x = 0.0
        elif x > self.w - 1.001:
            x = self.w - 1.001
        if y < 0.0:
            y = 0.0
        elif y > self.h - 1.001:
            y = self.h - 1.001
        x0 = int(x)
        y0 = int(y)
        fx = x - x0
        fy = y - y0
        px = self.px
        i = y0 * self.w + x0
        top = px[i] * (1.0 - fx) + px[i + 1] * fx
        bot = px[i + self.w] * (1.0 - fx) + px[i + self.w + 1] * fx
        return top * (1.0 - fy) + bot * fy

    def clone(self):
        return Gray(self.w, self.h, px=bytearray(self.px))

    def to_rgb(self):
        out = bytearray(self.w * self.h * 3)
        px = self.px
        for i in range(self.w * self.h):
            v = px[i]
            j = i * 3
            out[j] = v
            out[j + 1] = v
            out[j + 2] = v
        return Image(self.w, self.h, px=out)

    def save(self, path):
        return png.write_gray(path, self.w, self.h, bytes(self.px))


# --------------------------------------------------------------------- Image


class Image:
    """8-bit RGB image stored as ``w*h*3`` bytes."""

    __slots__ = ("w", "h", "px")

    def __init__(self, w, h, fill=(0, 0, 0), px=None):
        self.w = w
        self.h = h
        if px is not None:
            self.px = px
        else:
            self.px = bytearray(bytes(fill) * (w * h))

    # -- io -----------------------------------------------------------------
    @staticmethod
    def load(path):
        w, h, rgb = png.read(path)
        return Image(w, h, px=bytearray(rgb))

    def save(self, path, level=6):
        return png.write_rgb(path, self.w, self.h, bytes(self.px), level)

    def clone(self):
        return Image(self.w, self.h, px=bytearray(self.px))

    # -- pixel access -------------------------------------------------------
    def at(self, x, y):
        i = (y * self.w + x) * 3
        return self.px[i], self.px[i + 1], self.px[i + 2]

    def set(self, x, y, color):
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y * self.w + x) * 3
            self.px[i] = color[0]
            self.px[i + 1] = color[1]
            self.px[i + 2] = color[2]

    def blend(self, x, y, color, alpha):
        """Alpha-composite a single pixel. ``alpha`` in [0, 1]."""
        if alpha >= 0.999:
            self.set(x, y, color)
            return
        if alpha <= 0.001 or not (0 <= x < self.w and 0 <= y < self.h):
            return
        i = (y * self.w + x) * 3
        px = self.px
        inv = 1.0 - alpha
        px[i] = int(px[i] * inv + color[0] * alpha)
        px[i + 1] = int(px[i + 1] * inv + color[1] * alpha)
        px[i + 2] = int(px[i + 2] * inv + color[2] * alpha)

    # -- conversion ---------------------------------------------------------
    def to_gray(self):
        """ITU-R BT.601 luma, done with integer math for speed."""
        n = self.w * self.h
        out = bytearray(n)
        px = self.px
        for i in range(n):
            j = i * 3
            out[i] = (px[j] * 77 + px[j + 1] * 150 + px[j + 2] * 29) >> 8
        return Gray(self.w, self.h, px=out)

    # -- shapes -------------------------------------------------------------
    def fill_rect(self, x0, y0, x1, y1, color, alpha=1.0):
        x0, x1 = max(0, min(x0, x1)), min(self.w - 1, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(self.h - 1, max(y0, y1))
        if alpha >= 0.999:
            row = bytes(color) * (x1 - x0 + 1)
            for y in range(y0, y1 + 1):
                i = (y * self.w + x0) * 3
                self.px[i : i + len(row)] = row
        else:
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    self.blend(x, y, color, alpha)

    def draw_line(self, x0, y0, x1, y1, color, thickness=1, alpha=1.0):
        """Bresenham line, thickened by stamping a square brush."""
        x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        r = thickness // 2
        while True:
            if thickness <= 1:
                self.blend(x0, y0, color, alpha)
            else:
                for oy in range(-r, r + 1):
                    for ox in range(-r, r + 1):
                        self.blend(x0 + ox, y0 + oy, color, alpha)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def draw_poly(self, pts, color, thickness=1, alpha=1.0, closed=True):
        n = len(pts)
        last = n if closed else n - 1
        for i in range(last):
            a, b = pts[i], pts[(i + 1) % n]
            self.draw_line(a[0], a[1], b[0], b[1], color, thickness, alpha)

    def fill_poly(self, pts, color, alpha=1.0):
        """Scanline polygon fill (even-odd rule), works for convex and simple
        concave polygons alike."""
        if len(pts) < 3:
            return
        ys = [p[1] for p in pts]
        y_lo = max(0, int(min(ys)))
        y_hi = min(self.h - 1, int(max(ys)) + 1)
        n = len(pts)
        for y in range(y_lo, y_hi + 1):
            yc = y + 0.5
            xs = []
            for i in range(n):
                x0, y0 = pts[i][0], pts[i][1]
                x1, y1 = pts[(i + 1) % n][0], pts[(i + 1) % n][1]
                if (y0 <= yc < y1) or (y1 <= yc < y0):
                    t = (yc - y0) / (y1 - y0)
                    xs.append(x0 + t * (x1 - x0))
            if not xs:
                continue
            xs.sort()
            for k in range(0, len(xs) - 1, 2):
                xa = max(0, int(xs[k] + 0.5))
                xb = min(self.w - 1, int(xs[k + 1] - 0.5))
                if xb < xa:
                    continue
                if alpha >= 0.999:
                    row = bytes(color) * (xb - xa + 1)
                    i = (y * self.w + xa) * 3
                    self.px[i : i + len(row)] = row
                else:
                    for x in range(xa, xb + 1):
                        self.blend(x, y, color, alpha)

    def fill_circle(self, cx, cy, r, color, alpha=1.0):
        cx, cy = int(round(cx)), int(round(cy))
        r2 = r * r
        for y in range(max(0, cy - r), min(self.h - 1, cy + r) + 1):
            dy = y - cy
            span = int((r2 - dy * dy) ** 0.5) if r2 >= dy * dy else -1
            if span < 0:
                continue
            for x in range(max(0, cx - span), min(self.w - 1, cx + span) + 1):
                self.blend(x, y, color, alpha)

    def draw_circle(self, cx, cy, r, color, thickness=1, alpha=1.0):
        steps = max(12, int(r * 6))
        import math

        pts = [
            (
                cx + r * math.cos(2 * math.pi * i / steps),
                cy + r * math.sin(2 * math.pi * i / steps),
            )
            for i in range(steps)
        ]
        self.draw_poly(pts, color, thickness, alpha)

    # -- text ---------------------------------------------------------------
    def draw_text(
        self, x, y, s, color=(255, 255, 255), scale=1, spacing=1, bg=None, alpha=1.0
    ):
        """Draw ``s`` with the built-in 5x7 font, top-left anchored at (x, y)."""
        s = s.upper()
        if bg is not None:
            pad = 2 * scale
            self.fill_rect(
                int(x) - pad,
                int(y) - pad,
                int(x) + text_width(s, scale, spacing) + pad,
                int(y) + text_height(scale) + pad,
                bg,
                alpha if alpha < 1.0 else 0.72,
            )
        cx = int(x)
        for ch in s:
            glyph = FONT.get(ch)
            if glyph is None:
                glyph = FONT["?"]
            for gy in range(GLYPH_H):
                row = glyph[gy]
                for gx in range(GLYPH_W):
                    if not row[gx]:
                        continue
                    px0 = cx + gx * scale
                    py0 = int(y) + gy * scale
                    if scale == 1:
                        self.blend(px0, py0, color, alpha)
                    else:
                        for oy in range(scale):
                            for ox in range(scale):
                                self.blend(px0 + ox, py0 + oy, color, alpha)
            cx += (GLYPH_W + spacing) * scale
        return cx

    # -- composition --------------------------------------------------------
    def paste(self, other, x, y):
        for yy in range(other.h):
            ty = y + yy
            if not (0 <= ty < self.h):
                continue
            src = (yy * other.w) * 3
            width = min(other.w, self.w - x)
            if width <= 0:
                continue
            dst = (ty * self.w + x) * 3
            self.px[dst : dst + width * 3] = other.px[src : src + width * 3]

    def scaled(self, factor):
        """Integer nearest-neighbour upscale (keeps marker pixels crisp)."""
        nw, nh = self.w * factor, self.h * factor
        out = Image(nw, nh)
        for y in range(nh):
            sy = y // factor
            for x in range(nw):
                out.set(x, y, self.at(x // factor, sy))
        return out

    def resized(self, nw, nh):
        """Nearest-neighbour resize to arbitrary dimensions."""
        out = Image(nw, nh)
        for y in range(nh):
            sy = min(self.h - 1, y * self.h // nh)
            for x in range(nw):
                out.set(x, y, self.at(min(self.w - 1, x * self.w // nw), sy))
        return out


# ------------------------------------------------------------------- montage


def montage(panels, cols=3, pad=8, label_h=16, bg=(18, 18, 24)):
    """Lay out ``(title, Image)`` pairs on a labelled grid.

    Used to produce the pipeline-stages figure, which is the clearest way to
    show what each step of the detector actually does.
    """
    if not panels:
        raise ValueError("no panels")
    cw = max(p[1].w for p in panels)
    ch = max(p[1].h for p in panels)
    rows = (len(panels) + cols - 1) // cols
    out = Image(
        pad + cols * (cw + pad),
        pad + rows * (ch + label_h + pad),
        fill=bg,
    )
    for idx, (title, img) in enumerate(panels):
        r, c = divmod(idx, cols)
        x = pad + c * (cw + pad)
        y = pad + r * (ch + label_h + pad)
        out.draw_text(x, y + 4, title, (235, 235, 245), scale=1)
        out.paste(img, x, y + label_h)
    return out
