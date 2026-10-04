# HOLO — an augmented reality engine written from scratch

Marker detection → 6-DoF camera pose → 3D objects anchored to the marker, plus
the PNG and animated-GIF codecs used to write the results.

**Zero dependencies.** No OpenCV, no NumPy, no Pillow. The only imports in the
whole project are `math`, `zlib`, `struct`, `random`, `os`, `sys`, `json`,
`time`, `argparse` and `subprocess` — all Python standard library. Every pixel
is touched by Python code written here.

![AR overlay](outputs/frame_annotated.png)

*Three markers detected in a synthetic frame. Cyan outline + amber dot = the
detected quad and its recovered orientation. White dashed quad = ground truth.
Each label shows the decoded ID, the distance in millimetres, and the
reprojection error.*

## It is measured, not just demonstrated

"The cube looks stuck to the paper" is not a result. So the input frames are
ray-cast from a scene with **known** camera and marker placements, which makes
the pose the detector *should* recover exactly computable. Every number below is
a comparison against that ground truth, over a 30-frame 360° camera orbit
(3 markers, 90 marker-sightings):

| metric | mean | median | p95 | max |
|---|---|---|---|---|
| rotation error (deg) | 0.379 | 0.304 | 0.875 | 1.386 |
| position error (mm) | 1.54 | 1.37 | 3.00 | 4.25 |
| position error (% of distance) | 0.263 | 0.229 | 0.561 | 0.721 |
| depth bias (%) | **−0.04** | −0.05 | 0.30 | 0.69 |
| corner localisation (px) | 0.736 | 0.727 | 0.855 | 0.973 |
| reprojection before LM (px) | 0.928 | 0.687 | 2.449 | 4.126 |
| reprojection after LM (px) | **0.153** | 0.108 | 0.400 | 0.482 |

**Detection rate 98.9%** (89 of 90), **ID accuracy 100%**, **zero false
positives** — in frames that also contain a checkerboard floor contributing ~20
dark quadrilaterals each, every one of which has to be rejected.

Markers are 10 cm wide, the camera is 480×320 with a 58° field of view, so a
marker spans roughly 65–85 px. Recovering position to 1.5 mm at ~0.6 m from
~75 px of marker is a scale accuracy of about 0.26%.

Operating envelope, from `run.py bench`:

| condition | works | fails |
|---|---|---|
| distance (10 cm marker) | 0.22 m – 1.8 m (marker down to ~24 px) | ≥ 2.3 m |
| viewing angle off normal | up to 72° | ≥ 80° |
| Gaussian sensor noise | σ up to 34 grey levels | — (still detected at σ=34) |

![robustness](outputs/robustness.png)

Red `X` marks a sweep point where detection failed. These charts are drawn with
the same hand-rolled primitives as everything else — there is no plotting
library either.

## The pipeline

![pipeline](outputs/pipeline_stages.png)

1. **Synthesise** a frame: ray-cast the floor and marker sheets, then degrade it
   deliberately — lens blur, Gaussian sensor noise, vignetting and a horizontal
   brightness ramp. A detector that only works on clean synthetic pixels is not
   a detector.
2. **Grayscale** (BT.601 luma, integer math).
3. **Adaptive threshold**: local mean minus a bias, via a summed-area table so
   the window cost is O(1) per pixel. The brightness ramp is specifically there
   to break a global threshold; there is a test asserting that it does.
4. **Morphological close** to repair border rings nicked by blur and noise.
5. **Connected components → boundary tracing** (iterative flood fill, then a
   Moore-neighbour radial sweep) → **Douglas–Peucker** polygon approximation →
   keep convex quads with plausible size and corner angles.
6. **Subpixel corner refinement**, twice: line fits to the contour sides, then
   line fits to *grayscale gradient peaks* across each edge.
7. **Perspective unwarp** to a canonical 7×7 cell square, Otsu threshold inside
   the patch, majority-vote each cell to a bit.
8. **Border check + dictionary match** over all four rotations, recovering both
   the ID and the orientation.
9. **Pose**: normalised DLT homography → analytic decomposition → Levenberg–
   Marquardt on reprojection error.
10. **Augment**: draw a cube and a spinning octahedron in the marker's own
    coordinate frame, pushed through the estimated pose. If the pose is wrong by
    a degree, the cube visibly slides off the paper.

## Four decisions worth reading the code for

**Subpixel refinement is not cosmetic — it removes a systematic bias.** Corners
taken from a binarised contour inherit the threshold's bias: a mean-adaptive
threshold sits slightly below the true 50% intensity crossing, so every black
square comes out a fraction of a pixel too small *in every direction*, which
reads as the marker being further away than it is. Measured on a fixed frame:

| | corner error | depth bias | position error |
|---|---|---|---|
| raw contour corners | 1.261 px | +1.46% | 9.21 mm |
| + contour line fits | 1.017 px | +1.60% | 9.77 mm |
| + grayscale gradient fits | **0.750 px** | **+0.20%** | **1.82 mm** |

The gradient peak of a blurred step edge *is* the edge, independent of any
threshold — so the bias goes away rather than shrinking. There is a regression
test (`test_subpixel_refinement_removes_depth_bias`) that fails if it returns.

**The GIF LZW off-by-one.** The encoder assigns a dictionary entry on every
emitted code, but a decoder only starts assigning from its *second* code, so the
decoder's counter permanently trails the encoder's by one. The two therefore need
code-width rules that differ by exactly one — `next_code > (1 << width)` when
encoding, `next_code == (1 << width)` when decoding. Get it wrong and the stream
desynchronises the first time the width grows, part-way into the image.

Rather than assume, the codecs are validated against *other people's* files:
`holo/gif.py`'s decoder is run over every GIF on the host (192 of 193 decode,
the one failure being a truncated file), and `holo/png.py`'s decoder over every
host PNG, cross-checked against an independently written re-implementation of
the spec's scanline filters. A codec that only round-trips through itself proves
nothing but self-consistency.

**Marker dictionaries need rotation-aware distance.** Markers are generated by
randomised greedy search under two constraints: every pair differs by ≥ 8 bits
*in all four rotations*, and every marker differs from *its own* rotations by
≥ 8 bits. The second constraint is what makes orientation recoverable — without
it a rotationally symmetric pattern would leave the pose 90°-ambiguous. With 16
markers of 25 payload bits this corrects any 3 bit errors; the test suite
flips bits at random 400 times per error count and asserts zero misdecodes.

**Evaluation matches detections to ground truth by geometry, not by ID.**
Matching on ID would quietly hide the most embarrassing possible failure —
finding a marker in the right place but decoding the wrong number — by counting
it as a miss plus a false positive. Matching on image position first and
comparing IDs second is what makes "ID accuracy: 100%" a real measurement.

## Known limitations

These are real, and measured rather than hidden:

- **Planar pose ambiguity.** A planar target viewed near fronto-parallel admits
  two poses with almost identical reprojections. Over 1500 random synthetic
  poses with 0.5 px corner noise, 7.2% came back with >20° rotation error — and
  those cases kept a *low* reprojection error (median 0.26 px) on *small*
  markers (median 31 px), which is the signature of the ambiguity rather than a
  solver failure. Translation stays accurate (~1% error) when it happens.
  Resolving it properly needs temporal tracking or a second marker.
- **Adaptive threshold block size vs marker scale.** The 25 px window must be
  larger than the marker's dark features (~10 px cells here). A very close-up
  marker whose cells exceed the window would need a scaled block size; the
  detector does not currently adapt it.
- **No lens distortion model.** The synthetic camera is a perfect pinhole, so
  fitting distortion coefficients would only fit noise. Real camera footage
  would need one.
- **Validated on synthetic imagery only.** No physical camera was available in
  this environment. The pipeline reads real PNGs, so feeding it photographs is
  possible, but the accuracy figures here are from rendered frames.
- **Speed.** Pure Python costs about 2.7 s to render and 0.57 s to detect one
  480×320 frame. The detector is the realistic part: ~1.8 fps, far from
  realtime, and dominated by per-pixel Python loops in the threshold stage.

## Running it

Needs only CPython ≥ 3.9.

```bash
python3 run.py selftest   # 50 tests, no pytest needed
python3 run.py all        # every artefact below (~3 min)

python3 run.py markers    # printable marker sheet
python3 run.py stages     # the pipeline figure
python3 run.py frame      # one annotated frame + error table
python3 run.py orbit      # animated AR GIF + accuracy report
python3 run.py bench      # distance / angle / noise sweeps
```

Artefacts land in `outputs/`:

| file | what |
|---|---|
| `ar_orbit.gif` | 30-frame 360° orbit with live AR overlay |
| `frame_annotated.png` | single frame, estimate vs ground truth |
| `pipeline_stages.png` | all eight pipeline stages side by side |
| `markers.png` | the 16-marker dictionary, printable |
| `robustness.png` | the sweep charts |
| `metrics.json` | every number, machine-readable |
| `run_all.log` | full console output of the last run |

## Layout

| module | lines | contents |
|---|---|---|
| `holo/linalg.py` | 319 | Gauss–Jordan, Jacobi eigensolver, null space, Rodrigues, look-at |
| `holo/png.py` | 181 | PNG encoder + decoder (colour types 0/2/3/4/6, all 5 filters) |
| `holo/gif.py` | 406 | median-cut quantiser, LZW encoder *and* decoder, GIF container |
| `holo/image.py` | 419 | RGB/grey buffers, lines, polygon fill, alpha, bilinear sampling, 5×7 font |
| `holo/filters.py` | 239 | summed-area tables, adaptive threshold, Otsu, blur, Sobel |
| `holo/morph.py` | 80 | separable erode/dilate/open/close |
| `holo/contours.py` | 170 | connected components, Moore-neighbour boundary tracing |
| `holo/geom.py` | 403 | Douglas–Peucker, quad validation, TLS line fits, subpixel edges |
| `holo/homography.py` | 75 | normalised DLT |
| `holo/pose.py` | 261 | pinhole camera, homography decomposition, Levenberg–Marquardt |
| `holo/marker.py` | 198 | dictionary generation, rendering, rotation-aware decoding |
| `holo/detect.py` | 289 | the detection pipeline |
| `holo/render.py` | 242 | 3D meshes, painter's algorithm, HUD, AR composition |
| `holo/scene.py` | 405 | ray-cast renderer, sensor artefacts, ground truth, trajectories |
| `holo/evaluate.py` | 221 | geometric matching, error metrics, percentile reports |
| `run.py` | 496 | CLI, demo generation, chart drawing |
| `tests/run_tests.py` | 980 | 50 tests |

The bitmap font in `image.py` is hand-drawn glyph by glyph, because there was no
font library to load either.
