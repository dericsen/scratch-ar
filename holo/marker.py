"""Fiducial marker dictionary, rendering, and decoding.

Layout of a HOLO marker (``PAYLOAD = 5``, so 7x7 cells total)::

    +-------------------+   <- 1-cell black border on all sides, which is what
    | # # # # # # #     |      the detector actually finds as a dark quad
    | #             #   |
    | #   payload   #   |      inner 5x5 cells carry 25 bits
    | #    5 x 5    #   |
    | #             #   |
    | # # # # # # #     |
    +-------------------+

The dictionary is built by randomised greedy search under two constraints:

* every pair of markers differs by at least ``min_dist`` bits **in all four
  rotations**, so a few misread cells cannot turn one ID into another;
* every marker differs from its *own* rotations by at least ``min_dist`` bits,
  which is what makes orientation recoverable -- without it a rotationally
  symmetric pattern would leave the pose 90-degree ambiguous.

That is the same design principle behind ArUco dictionaries, implemented from
scratch here.
"""

import random

from .image import Image

PAYLOAD = 5  # payload cells per side
GRID = PAYLOAD + 2  # including the black border ring


# ----------------------------------------------------------------- bit grids


def rotate_ccw(grid):
    """Rotate a square bit grid 90 degrees counter-clockwise."""
    n = len(grid)
    return [[grid[c][n - 1 - r] for c in range(n)] for r in range(n)]


def rotations(grid):
    out = [grid]
    for _ in range(3):
        out.append(rotate_ccw(out[-1]))
    return out


def flatten(grid):
    return [b for row in grid for b in row]


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def grid_hamming(g1, g2):
    return hamming(flatten(g1), flatten(g2))


# -------------------------------------------------------------- dictionary


class Dictionary:
    """A set of marker bit grids plus the decoding logic."""

    def __init__(self, grids, payload=PAYLOAD, min_dist=0):
        self.grids = grids
        self.payload = payload
        self.min_dist = min_dist
        # Pre-expand all four rotations of every marker: decoding is then a
        # plain nearest-neighbour search in Hamming space.
        self._rot_bits = []
        for g in grids:
            self._rot_bits.append([flatten(r) for r in rotations(g)])

    def __len__(self):
        return len(self.grids)

    @property
    def max_correctable(self):
        """Errors we can correct while staying unambiguous: floor((d-1)/2)."""
        return max(0, (self.min_dist - 1) // 2)

    def decode(self, grid, max_errors=None):
        """Match a sampled payload grid against the dictionary.

        Returns ``(marker_id, rotation, errors)`` where ``rotation`` is the
        number of 90-degree CCW rotations that maps the dictionary grid onto the
        sampled one, or ``None`` if nothing is close enough.
        """
        if max_errors is None:
            max_errors = self.max_correctable
        bits = flatten(grid)
        best = None
        for mid, rots in enumerate(self._rot_bits):
            for rot, rb in enumerate(rots):
                d = hamming(bits, rb)
                if best is None or d < best[2]:
                    best = (mid, rot, d)
        if best is None or best[2] > max_errors:
            return None
        return best


def generate_dictionary(count=12, payload=PAYLOAD, min_dist=7, seed=20261004, max_tries=400000):
    """Greedy randomised construction of a marker dictionary.

    Deterministic for a given ``seed``, so the dictionary is reproducible and
    markers printed today still decode tomorrow.
    """
    rng = random.Random(seed)
    accepted = []
    accepted_rots = []
    tries = 0
    n_cells = payload * payload
    while len(accepted) < count and tries < max_tries:
        tries += 1
        flat = [1 if rng.random() < 0.5 else 0 for _ in range(n_cells)]
        ones = sum(flat)
        # Reject near-blank or near-solid payloads: they carry little signal and
        # a mostly-white payload weakens the dark blob the detector looks for.
        if not (0.3 * n_cells <= ones <= 0.7 * n_cells):
            continue
        grid = [flat[r * payload : (r + 1) * payload] for r in range(payload)]
        rots = [flatten(r) for r in rotations(grid)]
        # self-distance across rotations -> orientation is recoverable
        if min(hamming(rots[0], rots[k]) for k in (1, 2, 3)) < min_dist:
            continue
        ok = True
        for other in accepted_rots:
            for rb in other:
                if hamming(rots[0], rb) < min_dist:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        accepted.append(grid)
        accepted_rots.append(rots)
    return Dictionary(accepted, payload, min_dist)


def dictionary_stats(d):
    """Minimum inter-marker and self-rotation Hamming distances (sanity check)."""
    inter = None
    for i in range(len(d.grids)):
        for j in range(len(d.grids)):
            if i >= j:
                continue
            for rb in d._rot_bits[j]:
                h = hamming(d._rot_bits[i][0], rb)
                inter = h if inter is None else min(inter, h)
    self_d = None
    for i in range(len(d.grids)):
        for k in (1, 2, 3):
            h = hamming(d._rot_bits[i][0], d._rot_bits[i][k])
            self_d = h if self_d is None else min(self_d, h)
    return {"markers": len(d.grids), "min_inter_distance": inter, "min_self_rotation_distance": self_d}


# ---------------------------------------------------------------- rendering


def full_grid(grid):
    """Wrap a payload grid in its black border, giving the full GRID x GRID cells.

    Convention: 1 = white cell, 0 = black cell. Row 0 is the top row.
    """
    p = len(grid)
    n = p + 2
    out = [[0] * n for _ in range(n)]
    for r in range(p):
        for c in range(p):
            out[r + 1][c + 1] = grid[r][c]
    return out


def render_marker(grid, cell_px=16, quiet_cells=1):
    """Render a printable marker: black border, payload, white quiet zone."""
    fg = full_grid(grid)
    n = len(fg)
    side = (n + 2 * quiet_cells) * cell_px
    img = Image(side, side, fill=(255, 255, 255))
    off = quiet_cells * cell_px
    for r in range(n):
        for c in range(n):
            if fg[r][c] == 0:
                img.fill_rect(
                    off + c * cell_px,
                    off + r * cell_px,
                    off + (c + 1) * cell_px - 1,
                    off + (r + 1) * cell_px - 1,
                    (0, 0, 0),
                )
    return img
