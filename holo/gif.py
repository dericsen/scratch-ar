"""Animated GIF encoder: median-cut quantisation + LZW compression.

Two pieces of real work here.

**Palette.** GIF allows 256 colours per frame, our renders are 24-bit, so we
build one global palette by median cut: treat the colour cloud as a box, keep
splitting the box with the largest weighted volume along its longest axis at the
median, and average each final box. Median cut adapts to where colours actually
are, unlike a fixed colour cube, which matters because most of our pixels are
grey floor and a handful are saturated overlay accents that would otherwise be
flattened.

**LZW.** The GIF variant, with one notorious trap: the encoder assigns a
dictionary entry on every emitted code, while the decoder only starts assigning
from its *second* code, so the decoder's counter trails the encoder's by one.
The two therefore need code-width rules that differ by exactly one --
``next_code > (1 << width)`` when encoding and ``next_code == (1 << width)``
when decoding -- or the stream desynchronises the first time the width grows and
the image dissolves into noise. :func:`decode_first_frame` implements the
decoder side so the round trip is actually tested rather than assumed.
"""

import struct

MAX_CODE = 4096


# --------------------------------------------------------------- quantisation


class _Box:
    __slots__ = ("colors", "count", "rmin", "rmax", "gmin", "gmax", "bmin", "bmax")

    def __init__(self, colors):
        self.colors = colors  # list of ((r, g, b), count)
        self.count = sum(c for _, c in colors)
        self._bounds()

    def _bounds(self):
        rs = [c[0][0] for c in self.colors]
        gs = [c[0][1] for c in self.colors]
        bs = [c[0][2] for c in self.colors]
        self.rmin, self.rmax = min(rs), max(rs)
        self.gmin, self.gmax = min(gs), max(gs)
        self.bmin, self.bmax = min(bs), max(bs)

    def longest_axis(self):
        dr = self.rmax - self.rmin
        dg = self.gmax - self.gmin
        db = self.bmax - self.bmin
        if dg >= dr and dg >= db:
            return 1, dg
        if dr >= db:
            return 0, dr
        return 2, db

    def priority(self):
        """Split the box that contributes the most total error."""
        _, extent = self.longest_axis()
        return self.count * (extent + 1)

    def split(self):
        axis, extent = self.longest_axis()
        if extent == 0 or len(self.colors) < 2:
            return None
        self.colors.sort(key=lambda c: c[0][axis])
        half = self.count // 2
        acc = 0
        cut = 0
        for i, (_, cnt) in enumerate(self.colors):
            acc += cnt
            if acc >= half and i < len(self.colors) - 1:
                cut = i + 1
                break
        if cut <= 0 or cut >= len(self.colors):
            cut = len(self.colors) // 2
        return _Box(self.colors[:cut]), _Box(self.colors[cut:])

    def average(self):
        r = g = b = 0
        for (cr, cg, cb), cnt in self.colors:
            r += cr * cnt
            g += cg * cnt
            b += cb * cnt
        n = max(1, self.count)
        return (r // n, g // n, b // n)


def build_palette(images, max_colors=256, sample_step=3):
    """Global palette across all frames via median cut.

    Colours are bucketed to 5 bits per channel before histogramming. That caps
    the number of distinct inputs to median cut at 32768 instead of the ~10^5
    a noisy render produces, which keeps the repeated sorting inside the split
    loop affordable. It costs nothing in practice because the nearest-colour
    lookup is cached at exactly the same 5-bit granularity.
    """
    hist = {}
    for img in images:
        px = img.px
        step = sample_step * 3
        for i in range(0, len(px) - 2, step):
            key = ((px[i] & 0xF8) | 4, (px[i + 1] & 0xF8) | 4, (px[i + 2] & 0xF8) | 4)
            hist[key] = hist.get(key, 0) + 1
    colors = list(hist.items())
    if len(colors) <= max_colors:
        pal = [c for c, _ in colors]
    else:
        boxes = [_Box(colors)]
        while len(boxes) < max_colors:
            boxes.sort(key=lambda b: -b.priority())
            for idx in range(len(boxes)):
                parts = boxes[idx].split()
                if parts:
                    boxes.pop(idx)
                    boxes.extend(parts)
                    break
            else:
                break  # nothing left can be split
        pal = [b.average() for b in boxes]
    # GIF colour tables must be a power-of-two length.
    size = 2
    while size < max(2, len(pal)):
        size <<= 1
    pal = pal + [(0, 0, 0)] * (size - len(pal))
    return pal


def _nearest_factory(palette):
    """Nearest-palette-index lookup, cached on a 5-bit-per-channel cell.

    A linear scan over 256 palette entries per pixel would dominate the whole
    encode. Caching on 32x32x32 colour cells means we pay that scan once per
    distinct cell actually present in the animation -- a few thousand times
    rather than millions.
    """
    cache = {}
    pal = palette

    def nearest(r, g, b):
        key = (r >> 3) << 10 | (g >> 3) << 5 | (b >> 3)
        idx = cache.get(key)
        if idx is not None:
            return idx
        # Compare against the cell centre so every colour in the cell agrees.
        cr = (r & 0xF8) | 4
        cg = (g & 0xF8) | 4
        cb = (b & 0xF8) | 4
        best = 0
        best_d = 1 << 30
        for i, (pr, pg, pb) in enumerate(pal):
            dr = pr - cr
            dg = pg - cg
            db = pb - cb
            d = dr * dr + dg * dg + db * db
            if d < best_d:
                best_d = d
                best = i
                if d == 0:
                    break
        cache[key] = best
        return best

    return nearest


def quantize(img, nearest):
    """Map an RGB image to palette indices."""
    px = img.px
    n = img.w * img.h
    out = bytearray(n)
    for i in range(n):
        j = i * 3
        out[i] = nearest(px[j], px[j + 1], px[j + 2])
    return out


# ----------------------------------------------------------------- LZW codec


class _BitWriter:
    """LSB-first bit packer that emits GIF sub-blocks."""

    def __init__(self):
        self.out = bytearray()
        self.acc = 0
        self.nbits = 0

    def write(self, code, width):
        self.acc |= code << self.nbits
        self.nbits += width
        while self.nbits >= 8:
            self.out.append(self.acc & 0xFF)
            self.acc >>= 8
            self.nbits -= 8

    def flush(self):
        if self.nbits > 0:
            self.out.append(self.acc & 0xFF)
            self.acc = 0
            self.nbits = 0

    def blocks(self):
        """Chop the byte stream into GIF's 255-byte sub-blocks."""
        self.flush()
        data = bytearray()
        pos = 0
        while pos < len(self.out):
            chunk = self.out[pos : pos + 255]
            data.append(len(chunk))
            data += chunk
            pos += len(chunk)
        data.append(0)  # block terminator
        return bytes(data)


def lzw_encode(indices, min_code_size):
    """Compress palette indices into a GIF image data stream."""
    clear = 1 << min_code_size
    eoi = clear + 1
    width = min_code_size + 1
    next_code = eoi + 1
    table = {}
    bw = _BitWriter()
    bw.write(clear, width)
    if not indices:
        bw.write(eoi, width)
        return bytes([min_code_size]) + bw.blocks()

    prefix = indices[0]
    for k in indices[1:]:
        key = (prefix << 12) | k  # cheaper than a tuple key
        found = table.get(key)
        if found is not None:
            prefix = found
            continue
        bw.write(prefix, width)
        if next_code < MAX_CODE:
            table[key] = next_code
            next_code += 1
            # See module docstring: '>' here, '==' in the decoder.
            if next_code > (1 << width) and width < 12:
                width += 1
        else:
            bw.write(clear, width)
            table = {}
            next_code = eoi + 1
            width = min_code_size + 1
        prefix = k
    bw.write(prefix, width)
    bw.write(eoi, width)
    return bytes([min_code_size]) + bw.blocks()


def _read_sub_blocks(data, pos):
    out = bytearray()
    while True:
        n = data[pos]
        pos += 1
        if n == 0:
            return bytes(out), pos
        out += data[pos : pos + n]
        pos += n


def lzw_decode(stream, min_code_size, expected=None):
    """Reference LZW decoder, used to verify the encoder.

    Deliberately written from the decoder's own perspective (including the
    KwKwK special case and the ``==`` width rule) rather than mirroring the
    encoder, so that a round-trip success is real evidence.
    """
    clear = 1 << min_code_size
    eoi = clear + 1
    width = min_code_size + 1
    table = {i: bytes([i]) for i in range(clear)}
    next_code = eoi + 1
    out = bytearray()
    acc = 0
    nbits = 0
    pos = 0
    old = None
    while True:
        while nbits < width:
            if pos >= len(stream):
                return bytes(out)
            acc |= stream[pos] << nbits
            nbits += 8
            pos += 1
        code = acc & ((1 << width) - 1)
        acc >>= width
        nbits -= width
        if code == clear:
            table = {i: bytes([i]) for i in range(clear)}
            next_code = eoi + 1
            width = min_code_size + 1
            old = None
            continue
        if code == eoi:
            return bytes(out)
        if old is None:
            entry = table[code]
            out += entry
            old = code
            continue
        if code in table:
            entry = table[code]
        else:
            # KwKwK: the code refers to the entry we are about to create.
            entry = table[old] + table[old][:1]
        out += entry
        if next_code < MAX_CODE:
            table[next_code] = table[old] + entry[:1]
            next_code += 1
            if next_code == (1 << width) and width < 12:
                width += 1
        old = code
        if expected is not None and len(out) >= expected:
            return bytes(out)


# ------------------------------------------------------------- GIF container


def encode_animation(images, delay_cs=8, loop=0, max_colors=256, palette=None):
    """Encode a list of :class:`holo.image.Image` frames as an animated GIF.

    ``delay_cs`` is the inter-frame delay in hundredths of a second.
    """
    if not images:
        raise ValueError("no frames")
    w, h = images[0].w, images[0].h
    for im in images:
        if im.w != w or im.h != h:
            raise ValueError("all frames must share the same dimensions")
    pal = palette or build_palette(images, max_colors)
    bits = 1
    while (1 << bits) < len(pal):
        bits += 1
    min_code_size = max(2, bits)
    nearest = _nearest_factory(pal)

    out = bytearray(b"GIF89a")
    # Logical screen descriptor: global colour table present, depth = bits
    packed = 0x80 | ((bits - 1) & 0x07)
    out += struct.pack("<HHBBB", w, h, packed, 0, 0)
    for r, g, b in pal:
        out += bytes((r, g, b))
    # Netscape extension -> loop forever
    out += b"\x21\xff\x0bNETSCAPE2.0\x03\x01" + struct.pack("<H", loop) + b"\x00"

    for img in images:
        idx = quantize(img, nearest)
        # Graphic control extension: disposal "do not dispose", no transparency
        out += b"\x21\xf9\x04\x04" + struct.pack("<H", delay_cs) + b"\x00\x00"
        # Image descriptor: full frame, no local colour table, not interlaced
        out += b"\x2c" + struct.pack("<HHHHB", 0, 0, w, h, 0)
        out += lzw_encode(idx, min_code_size)
    out += b"\x3b"  # trailer
    return bytes(out)


def write_animation(path, images, delay_cs=8, loop=0, max_colors=256):
    data = encode_animation(images, delay_cs, loop, max_colors)
    with open(path, "wb") as fh:
        fh.write(data)
    return path, len(data)


def decode_first_frame(data):
    """Parse a GIF far enough to return ``(w, h, palette, indices)`` of frame 1.

    This exists for the test suite: it is the independent check that our LZW
    stream is really decodable, not just self-consistent.
    """
    if data[:6] not in (b"GIF89a", b"GIF87a"):
        raise ValueError("not a GIF")
    w, h, packed, _bg, _ar = struct.unpack("<HHBBB", data[6:13])
    pos = 13
    palette = []
    if packed & 0x80:
        size = 2 << (packed & 0x07)
        for i in range(size):
            palette.append(tuple(data[pos + i * 3 : pos + i * 3 + 3]))
        pos += size * 3
    while pos < len(data):
        block = data[pos]
        if block == 0x21:  # extension
            pos += 2  # skip label
            _body, pos = _read_sub_blocks(data, pos)
        elif block == 0x2C:  # image descriptor
            _x, _y, iw, ih, ipacked = struct.unpack("<HHHHB", data[pos + 1 : pos + 10])
            pos += 10
            if ipacked & 0x80:
                lsize = 2 << (ipacked & 0x07)
                pos += lsize * 3
            min_code_size = data[pos]
            pos += 1
            stream, pos = _read_sub_blocks(data, pos)
            indices = lzw_decode(stream, min_code_size, expected=iw * ih)
            return iw, ih, palette, indices
        elif block == 0x3B:
            break
        else:
            raise ValueError("unexpected GIF block 0x%02x at %d" % (block, pos))
    raise ValueError("no image data found")
