"""PNG encoder and decoder written from scratch.

Only ``zlib`` (stdlib) is used for the DEFLATE layer; the container format,
chunk CRCs, scanline filtering and colour-type handling are all implemented
here.

Supported on decode: bit depth 8, non-interlaced, colour types 0 (gray),
2 (RGB), 3 (palette), 4 (gray+alpha) and 6 (RGBA). That covers essentially
every PNG a normal tool emits, so you can feed HOLO your own photos.
"""

import struct
import zlib

_SIG = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------- encode


def _chunk(tag, data):
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _encode(width, height, pixels, color_type, channels, level=6):
    raw = bytearray()
    stride = width * channels
    for y in range(height):
        raw.append(0)  # filter type 0 (None): simple and fast to produce
        start = y * stride
        raw += pixels[start : start + stride]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (
        _SIG
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), level))
        + _chunk(b"IEND", b"")
    )


def encode_rgb(width, height, pixels, level=6):
    """``pixels``: ``width*height*3`` bytes of 8-bit RGB."""
    return _encode(width, height, pixels, 2, 3, level)


def encode_gray(width, height, pixels, level=6):
    """``pixels``: ``width*height`` bytes of 8-bit luminance."""
    return _encode(width, height, pixels, 0, 1, level)


def write_rgb(path, width, height, pixels, level=6):
    with open(path, "wb") as fh:
        fh.write(encode_rgb(width, height, pixels, level))
    return path


def write_gray(path, width, height, pixels, level=6):
    with open(path, "wb") as fh:
        fh.write(encode_gray(width, height, pixels, level))
    return path


# ---------------------------------------------------------------------- decode


def _unfilter(raw, width, height, bpp):
    """Reverse the per-scanline PNG filters (spec section 9.2)."""
    stride = width * bpp
    out = bytearray(stride * height)
    pos = 0
    for y in range(height):
        ftype = raw[pos]
        pos += 1
        line = raw[pos : pos + stride]
        pos += stride
        base = y * stride
        prev = base - stride
        if ftype == 0:
            out[base : base + stride] = line
        elif ftype == 1:  # Sub
            for i in range(stride):
                left = out[base + i - bpp] if i >= bpp else 0
                out[base + i] = (line[i] + left) & 0xFF
        elif ftype == 2:  # Up
            if y == 0:
                out[base : base + stride] = line
            else:
                for i in range(stride):
                    out[base + i] = (line[i] + out[prev + i]) & 0xFF
        elif ftype == 3:  # Average
            for i in range(stride):
                left = out[base + i - bpp] if i >= bpp else 0
                up = out[prev + i] if y > 0 else 0
                out[base + i] = (line[i] + ((left + up) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                left = out[base + i - bpp] if i >= bpp else 0
                up = out[prev + i] if y > 0 else 0
                ul = out[prev + i - bpp] if (y > 0 and i >= bpp) else 0
                p = left + up - ul
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - ul)
                if pa <= pb and pa <= pc:
                    pred = left
                elif pb <= pc:
                    pred = up
                else:
                    pred = ul
                out[base + i] = (line[i] + pred) & 0xFF
        else:
            raise ValueError("unknown PNG filter type %d" % ftype)
    return out


def decode(data):
    """Decode PNG bytes into ``(width, height, rgb_bytearray)``."""
    if data[:8] != _SIG:
        raise ValueError("not a PNG file")
    pos = 8
    width = height = depth = color_type = None
    idat = bytearray()
    palette = None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        pos += 12 + length  # 4 len + 4 tag + body + 4 crc
        if tag == b"IHDR":
            width, height, depth, color_type, _comp, _filt, interlace = struct.unpack(
                ">IIBBBBB", body
            )
            if depth != 8:
                raise ValueError("only 8-bit PNGs are supported (got %d)" % depth)
            if interlace:
                raise ValueError("interlaced PNGs are not supported")
        elif tag == b"PLTE":
            palette = body
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
    if width is None:
        raise ValueError("PNG has no IHDR chunk")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    raw = _unfilter(zlib.decompress(bytes(idat)), width, height, channels)

    n = width * height
    rgb = bytearray(n * 3)
    if color_type == 2:
        return width, height, raw
    if color_type == 0:
        for i in range(n):
            v = raw[i]
            j = i * 3
            rgb[j] = rgb[j + 1] = rgb[j + 2] = v
    elif color_type == 4:
        for i in range(n):
            v = raw[i * 2]
            j = i * 3
            rgb[j] = rgb[j + 1] = rgb[j + 2] = v
    elif color_type == 6:
        for i in range(n):
            s, j = i * 4, i * 3
            rgb[j : j + 3] = raw[s : s + 3]
    elif color_type == 3:
        if palette is None:
            raise ValueError("palette PNG without PLTE chunk")
        for i in range(n):
            s = raw[i] * 3
            j = i * 3
            rgb[j : j + 3] = palette[s : s + 3]
    return width, height, rgb


def read(path):
    with open(path, "rb") as fh:
        return decode(fh.read())
