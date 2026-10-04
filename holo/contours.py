"""Connected components and boundary tracing.

Two stages:

1. :func:`label_components` -- iterative flood fill (4-connectivity) over the
   binary mask. Iterative on purpose: a recursive fill blows the Python stack
   on a region of any real size.
2. :func:`trace_boundary` -- Moore-neighbour radial sweep that walks the outer
   boundary of one component and returns it as an ordered pixel chain, which is
   what the polygon approximator consumes.
"""

from .filters import FG

# 8-neighbourhood in clockwise order, starting East.
_OFF = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))


class Component:
    __slots__ = ("label", "area", "x0", "y0", "x1", "y1", "seed")

    def __init__(self, label, seed):
        self.label = label
        self.area = 0
        self.seed = seed  # first pixel in raster order: topmost, then leftmost
        self.x0 = self.x1 = seed[0]
        self.y0 = self.y1 = seed[1]

    @property
    def width(self):
        return self.x1 - self.x0 + 1

    @property
    def height(self):
        return self.y1 - self.y0 + 1

    def __repr__(self):
        return "Component(label=%d, area=%d, bbox=(%d,%d,%d,%d))" % (
            self.label,
            self.area,
            self.x0,
            self.y0,
            self.x1,
            self.y1,
        )


def label_components(mask, fg=FG):
    """Label 4-connected foreground regions.

    Returns ``(labels, components)`` where ``labels`` is a flat list of ints
    (0 = background, n >= 1 = component id) and ``components`` is a list of
    :class:`Component` metadata.
    """
    w, h = mask.w, mask.h
    px = mask.px
    labels = [0] * (w * h)
    comps = []
    next_label = 0
    for sy in range(h):
        base = sy * w
        for sx in range(w):
            i = base + sx
            if px[i] != fg or labels[i]:
                continue
            next_label += 1
            comp = Component(next_label, (sx, sy))
            labels[i] = next_label
            stack = [i]
            area = 0
            x0 = x1 = sx
            y0 = y1 = sy
            while stack:
                p = stack.pop()
                area += 1
                py, pxx = divmod(p, w)
                if pxx < x0:
                    x0 = pxx
                elif pxx > x1:
                    x1 = pxx
                if py > y1:
                    y1 = py
                # left
                if pxx > 0:
                    q = p - 1
                    if px[q] == fg and not labels[q]:
                        labels[q] = next_label
                        stack.append(q)
                if pxx < w - 1:
                    q = p + 1
                    if px[q] == fg and not labels[q]:
                        labels[q] = next_label
                        stack.append(q)
                if py > 0:
                    q = p - w
                    if px[q] == fg and not labels[q]:
                        labels[q] = next_label
                        stack.append(q)
                if py < h - 1:
                    q = p + w
                    if px[q] == fg and not labels[q]:
                        labels[q] = next_label
                        stack.append(q)
            comp.area = area
            comp.x0, comp.x1, comp.y0, comp.y1 = x0, x1, y0, y1
            comps.append(comp)
    return labels, comps


def trace_boundary(labels, w, h, comp, max_steps=None):
    """Walk the outer boundary of ``comp`` clockwise, returning ``[(x, y), ...]``.

    Starts from the component's raster-order seed pixel. Because that seed is
    the topmost-then-leftmost pixel of the component, its western neighbour is
    guaranteed to be outside the component, which gives the sweep a valid
    initial backtrack direction.
    """
    lab = comp.label
    start = comp.seed
    if max_steps is None:
        # Generous bound: a boundary cannot be longer than the component's
        # perimeter in the worst (comb-shaped) case.
        max_steps = 4 * (comp.width + comp.height) + 4 * comp.area + 64
    chain = [start]
    cur = start
    prev_dir = 4  # we conceptually arrived from the West
    steps = 0
    while True:
        found = False
        for k in range(1, 9):
            d = (prev_dir + k) & 7
            ox, oy = _OFF[d]
            nx, ny = cur[0] + ox, cur[1] + oy
            if 0 <= nx < w and 0 <= ny < h and labels[ny * w + nx] == lab:
                cur = (nx, ny)
                prev_dir = (d + 4) & 7  # direction pointing back to where we came from
                found = True
                break
        if not found:
            break  # isolated single pixel
        if cur == start:
            break
        chain.append(cur)
        steps += 1
        if steps >= max_steps:
            break
    return chain


def find_contours(mask, min_area=100, max_area_frac=0.9, fg=FG):
    """Convenience wrapper: label, filter by area, trace each survivor.

    Returns a list of ``(Component, chain)`` sorted by descending area.
    """
    w, h = mask.w, mask.h
    labels, comps = label_components(mask, fg)
    max_area = int(w * h * max_area_frac)
    out = []
    for c in comps:
        if c.area < min_area or c.area > max_area:
            continue
        # A marker is a quad; a component thinner than 3px in either direction
        # cannot be one.
        if c.width < 8 or c.height < 8:
            continue
        chain = trace_boundary(labels, w, h, c)
        if len(chain) >= 8:
            out.append((c, chain))
    out.sort(key=lambda t: -t[0].area)
    return out, labels
