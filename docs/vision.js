/* HOLO LAVA -- silhouette vision core.
 *
 * No libraries. The job: find the player's body in a webcam frame, every
 * frame, whether they are moving or standing perfectly still.
 *
 * Why background subtraction and not frame differencing
 * ----------------------------------------------------
 * Frame differencing (compare this frame to the previous one) is the easy way
 * to find movement, and it is useless for this game: the moment the player
 * holds still they vanish, and "the floor is lava" needs to know where the
 * body *is*, not where it last moved. So instead the detector memorises the
 * empty room once, and then every frame asks each pixel "do you still look
 * like the room, or does something cover you?". A motionless player stays
 * visible indefinitely.
 *
 * Three things this handles that a textbook one-liner does not:
 *
 *   1. Sensor noise. Webcam pixels flicker even on a dead-still scene, far
 *      more in a dim room. The threshold is not a magic constant -- it is
 *      measured from the camera's own noise during calibration
 *      (mean + k * stddev of the frame-to-frame variation).
 *   2. Shadows. A person casts a shadow, the shadow differs from the learned
 *      background, and naively it becomes part of the body -- which in this
 *      game means your shadow falls in the lava and kills you. A shadowed
 *      pixel keeps roughly its surroundings' *ratio* while dropping in
 *      brightness, so pixels that are merely scaled-down versions of the
 *      background are classified as shadow, not body.
 *   3. Lighting drift. Rooms slowly change brightness. Pixels that look like
 *      background get blended into the background model very slowly, so the
 *      model tracks drift without ever absorbing a stationary player.
 *
 * Exposed as a global, so the same file runs from https://, from file://, and
 * inside Node for the tests -- no bundler, no module loader, no build step.
 */
(function (root) {
  'use strict';

  var DEFAULTS = {
    width: 160,                 // processing resolution (not display size)
    height: 120,
    gridCols: 40,               // coarse grid used for fast collision queries
    gridRows: 30,
    noiseK: 6.0,                // threshold = noiseMean + noiseK * noiseStd
    minThreshold: 12,
    maxThreshold: 80,
    shadowSuppression: true,
    shadowRatioLo: 0.55,        // darker than this -> real object, not shadow
    shadowRatioHi: 0.93,        // between lo and hi, and darker -> shadow
    backgroundAdapt: 0.004,     // per-frame blend for pixels seen as background
    openingPasses: 1,           // morphological cleanup of the mask
    sensitivity: 1.0,
    minBlobCells: 4
  };

  function createDetector(options) {
    var o = {};
    for (var k in DEFAULTS) o[k] = DEFAULTS[k];
    if (options) for (var j in options) if (options[j] !== undefined) o[j] = options[j];

    var W = o.width, H = o.height;
    var nPixels = W * H;
    var cols = o.gridCols, rows = o.gridRows;
    var nCells = cols * rows;

    var background = new Float32Array(nPixels);
    var mask = new Uint8Array(nPixels);
    var maskTmp = new Uint8Array(nPixels);
    var occupancy = new Float32Array(nCells);
    var cellCount = new Int32Array(nCells);
    var cellPixels = new Int32Array(nCells);
    var cellOf = new Int32Array(nPixels);

    var cellW = W / cols, cellH = H / rows;
    for (var y = 0; y < H; y++) {
      var gy = Math.min(rows - 1, (y / cellH) | 0);
      for (var x = 0; x < W; x++) {
        var gx = Math.min(cols - 1, (x / cellW) | 0);
        var ci = gy * cols + gx;
        cellOf[y * W + x] = ci;
        cellPixels[ci]++;
      }
    }

    // --- calibration state
    var calibFrames = 0;
    var calibSum = new Float32Array(nPixels);
    var prevFrame = null;
    var noiseSum = 0, noiseSumSq = 0, noiseSamples = 0;
    var threshold = o.minThreshold;
    var ready = false;

    var visited = new Uint8Array(nCells);
    var stack = new Int32Array(nCells);

    /* ---------------------------------------------------------------- calibrate
     * Feed frames of the *empty* scene. Accumulates the mean background and,
     * from consecutive frames, the camera's noise statistics. */
    function learnBackground(gray) {
      requireFrame(gray);
      for (var i = 0; i < nPixels; i++) calibSum[i] += gray[i];
      if (prevFrame) {
        for (var p = 0; p < nPixels; p++) {
          var d = gray[p] - prevFrame[p];
          if (d < 0) d = -d;
          noiseSum += d;
          noiseSumSq += d * d;
        }
        noiseSamples += nPixels;
      } else {
        prevFrame = new Uint8Array(nPixels);
      }
      prevFrame.set(gray);
      calibFrames++;
      return calibFrames;
    }

    function finishCalibration() {
      if (calibFrames === 0) return false;
      for (var i = 0; i < nPixels; i++) background[i] = calibSum[i] / calibFrames;
      if (noiseSamples > 0) {
        var mean = noiseSum / noiseSamples;
        var varc = Math.max(0, noiseSumSq / noiseSamples - mean * mean);
        threshold = clamp(mean + o.noiseK * Math.sqrt(varc), o.minThreshold, o.maxThreshold);
      } else {
        threshold = o.minThreshold;
      }
      ready = true;
      return true;
    }

    function resetCalibration() {
      calibFrames = 0;
      calibSum.fill(0);
      prevFrame = null;
      noiseSum = noiseSumSq = noiseSamples = 0;
      threshold = o.minThreshold;
      ready = false;
      mask.fill(0);
      occupancy.fill(0);
    }

    function effectiveThreshold() {
      // More sensitivity -> lower threshold. Clamped so an extreme slider
      // cannot drop it into the noise floor and light the whole frame up.
      return clamp(threshold / Math.max(0.25, o.sensitivity), 6, o.maxThreshold);
    }

    /* ------------------------------------------------------------------ update
     * Classify every pixel as body / shadow / background, clean up the mask,
     * and summarise it into the coarse occupancy grid the game queries. */
    function update(gray) {
      requireFrame(gray);
      if (!ready) throw new Error('vision.update: call finishCalibration() first');

      var thr = effectiveThreshold();
      var shadowLo = o.shadowRatioLo, shadowHi = o.shadowRatioHi;
      var adapt = o.backgroundAdapt;
      var body = 0;

      for (var i = 0; i < nPixels; i++) {
        var b = background[i];
        var v = gray[i];
        var d = v - b;
        var isBody = false;

        if (d > thr) {
          isBody = true;                     // brighter than the room
        } else if (-d > thr) {
          if (o.shadowSuppression) {
            // Darker than the room: body, unless it looks like a shadow, i.e.
            // the pixel is a modest scaled-down copy of the background rather
            // than a different surface altogether.
            var ratio = v / (b > 1 ? b : 1);
            isBody = !(ratio >= shadowLo && ratio <= shadowHi);
          } else {
            isBody = true;
          }
        }

        maskTmp[i] = isBody ? 1 : 0;
        if (!isBody && adapt > 0) {
          background[i] = b + (v - b) * adapt;   // slow drift tracking only
        }
      }

      // Morphological opening removes isolated speckle without eating limbs.
      if (o.openingPasses > 0) {
        openMask(maskTmp, mask, W, H, o.openingPasses);
      } else {
        mask.set(maskTmp);
      }

      cellCount.fill(0);
      for (var q = 0; q < nPixels; q++) {
        if (mask[q]) { cellCount[cellOf[q]]++; body++; }
      }
      for (var c = 0; c < nCells; c++) occupancy[c] = cellCount[c] / cellPixels[c];

      var coverage = body / nPixels;
      return {
        ready: true,
        coverage: coverage,
        // Almost the whole frame changed: the lights went off, someone moved
        // the laptop, or the calibration is stale. The game shows a prompt
        // instead of pretending the player is everywhere at once.
        backgroundStale: coverage > 0.82,
        threshold: thr,
        mask: mask,
        occupancy: occupancy,
        width: W, height: H,
        gridCols: cols, gridRows: rows,
        lowestBodyY: lowestBodyY(),
        blobs: findBlobs()
      };
    }

    /* Lowest point of the body in normalised [0,1] coords (1 = frame bottom).
     * This is what "how close am I to the lava?" is measured from. */
    function lowestBodyY() {
      for (var gy = rows - 1; gy >= 0; gy--) {
        for (var gx = 0; gx < cols; gx++) {
          if (occupancy[gy * cols + gx] > 0.25) return (gy + 1) / rows;
        }
      }
      return null;
    }

    /* Fraction of a circle (normalised coords) covered by the body.
     * Used for both gem pickups and lava contact. */
    function occupancyInCircle(nx, ny, nr) {
      var cx = nx * cols, cy = ny * rows;
      var rx = Math.max(0.5, nr * cols), ry = Math.max(0.5, nr * rows);
      var x0 = Math.max(0, Math.floor(cx - rx)), x1 = Math.min(cols - 1, Math.ceil(cx + rx));
      var y0 = Math.max(0, Math.floor(cy - ry)), y1 = Math.min(rows - 1, Math.ceil(cy + ry));
      var sum = 0, peak = 0, n = 0;
      for (var gy = y0; gy <= y1; gy++) {
        for (var gx = x0; gx <= x1; gx++) {
          var ddx = (gx + 0.5 - cx) / rx, ddy = (gy + 0.5 - cy) / ry;
          if (ddx * ddx + ddy * ddy > 1) continue;
          var e = occupancy[gy * cols + gx];
          sum += e;
          if (e > peak) peak = e;
          n++;
        }
      }
      return { mean: n ? sum / n : 0, peak: peak, cells: n };
    }

    /* Fraction of a horizontal band (normalised y range) covered by the body.
     * The lava is a band, so this is the lava-contact test. */
    function occupancyInBand(ny0, ny1) {
      var y0 = Math.max(0, Math.floor(ny0 * rows));
      var y1 = Math.min(rows - 1, Math.ceil(ny1 * rows) - 1);
      if (y1 < y0) y1 = y0;
      var sum = 0, peak = 0, n = 0;
      for (var gy = y0; gy <= y1; gy++) {
        for (var gx = 0; gx < cols; gx++) {
          var e = occupancy[gy * cols + gx];
          sum += e;
          if (e > peak) peak = e;
          n++;
        }
      }
      return { mean: n ? sum / n : 0, peak: peak, cells: n };
    }

    function findBlobs() {
      visited.fill(0);
      var blobs = [];
      for (var start = 0; start < nCells; start++) {
        if (visited[start] || occupancy[start] < 0.2) continue;
        var sp = 0;
        stack[sp++] = start;
        visited[start] = 1;
        var mass = 0, sx = 0, sy = 0, cells = 0, topY = rows;
        while (sp > 0) {
          var idx = stack[--sp];
          var cy = (idx / cols) | 0, cx = idx - cy * cols;
          var w = occupancy[idx];
          mass += w; sx += (cx + 0.5) * w; sy += (cy + 0.5) * w; cells++;
          if (cy < topY) topY = cy;
          if (cx > 0) sp = maybePush(idx - 1, sp);
          if (cx < cols - 1) sp = maybePush(idx + 1, sp);
          if (cy > 0) sp = maybePush(idx - cols, sp);
          if (cy < rows - 1) sp = maybePush(idx + cols, sp);
        }
        if (cells < o.minBlobCells || mass <= 0) continue;
        blobs.push({
          x: (sx / mass) / cols, y: (sy / mass) / rows,
          topY: topY / rows, mass: mass, cells: cells
        });
      }
      blobs.sort(function (a, b) { return b.mass - a.mass; });
      return blobs;
    }

    function maybePush(idx, sp) {
      if (visited[idx] || occupancy[idx] < 0.2) return sp;
      visited[idx] = 1;
      stack[sp] = idx;
      return sp + 1;
    }

    function requireFrame(gray) {
      if (!gray || gray.length < nPixels) {
        throw new Error('vision: expected ' + nPixels + ' grayscale samples, got ' +
                        (gray ? gray.length : 'none'));
      }
    }

    return {
      learnBackground: learnBackground,
      finishCalibration: finishCalibration,
      resetCalibration: resetCalibration,
      update: update,
      occupancyInCircle: occupancyInCircle,
      occupancyInBand: occupancyInBand,
      setSensitivity: function (v) { o.sensitivity = v; },
      isReady: function () { return ready; },
      getThreshold: effectiveThreshold,
      calibrationFrames: function () { return calibFrames; },
      options: o
    };
  }

  /* Erode then dilate (a square structuring element), which deletes lone
   * speckles while leaving solid regions their original size. */
  function openMask(src, dst, W, H, passes) {
    var a = src, b = dst;
    for (var p = 0; p < passes; p++) {
      morph(a, b, W, H, true);
      morph(b, a, W, H, false);
    }
    if (a !== dst) dst.set(a);
  }

  function morph(src, dst, W, H, erode) {
    for (var y = 0; y < H; y++) {
      for (var x = 0; x < W; x++) {
        var i = y * W + x;
        var acc = erode ? 1 : 0;
        for (var dy = -1; dy <= 1 && (erode ? acc : !acc); dy++) {
          var yy = y + dy;
          if (yy < 0 || yy >= H) { if (erode) { acc = 0; } continue; }
          for (var dx = -1; dx <= 1; dx++) {
            var xx = x + dx;
            if (xx < 0 || xx >= W) { if (erode) { acc = 0; } break; }
            var v = src[yy * W + xx];
            if (erode) { if (!v) { acc = 0; break; } }
            else if (v) { acc = 1; break; }
          }
        }
        dst[i] = acc;
      }
    }
  }

  /* RGBA -> downscaled grayscale. Nearest-neighbour: we need a silhouette,
   * not detail, and box filtering every pixel would cost more than it buys. */
  function toGrayscale(rgba, srcW, srcH, dstW, dstH, out) {
    out = out || new Uint8Array(dstW * dstH);
    for (var y = 0; y < dstH; y++) {
      var sy = ((y * srcH / dstH) | 0) * srcW;
      var row = y * dstW;
      for (var x = 0; x < dstW; x++) {
        var si = (sy + ((x * srcW / dstW) | 0)) * 4;
        out[row + x] = (rgba[si] * 77 + rgba[si + 1] * 150 + rgba[si + 2] * 29) >> 8;
      }
    }
    return out;
  }

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  /* ------------------------------------------------------------------ tracker
   * Turns a raw silhouette into the two signals the game actually plays on:
   *
   *   WHICH TILE am I standing on   -> body mass summed per vertical lane
   *   DID I JUST JUMP               -> body centroid rising sharply
   *
   * Jump detection cannot use an absolute height, because every player stands
   * at a different distance and a different height in frame. So the tracker
   * keeps a slow-moving baseline of where the body normally sits and calls a
   * jump when the body rises far enough *above its own baseline*. The baseline
   * is frozen while airborne -- otherwise a long hang-time would drag the
   * baseline up with it and the landing would register as a second jump.
   */
  function createPlayerTracker(options) {
    var o = {
      lanes: 4,
      jumpRise: 0.055,        // fraction of frame height above baseline
      landFraction: 0.45,     // drop back below this fraction of the rise -> landed
      baselineRate: 0.05,     // how fast the resting position is learned
      refractoryMs: 420,      // ignore new jumps this soon after one fires
      minCoverage: 0.012,     // below this there is nobody in frame
      laneSmoothing: 0.35
    };
    if (options) for (var k in options) if (options[k] !== undefined) o[k] = options[k];

    var laneWeights = new Float32Array(o.lanes);
    var baselineY = null;
    var airborne = false;
    var lastJumpAt = -1e9;
    var peakRise = 0;
    var lastCentroid = null;
    var present = false;

    function reset() {
      laneWeights.fill(0);
      baselineY = null;
      airborne = false;
      lastJumpAt = -1e9;
      peakRise = 0;
      lastCentroid = null;
      present = false;
    }

    function update(st, nowMs) {
      var jumped = false;
      present = !!st && st.coverage >= o.minCoverage && !st.backgroundStale;

      if (!present) {
        // Decay the lanes so stale bars do not linger on screen, but keep the
        // baseline: the player has probably just stepped out for a moment.
        for (var d = 0; d < o.lanes; d++) laneWeights[d] *= 0.8;
        lastCentroid = null;
        airborne = false;
        return result(false);
      }

      // --- lane occupancy
      var cols = st.gridCols, rows = st.gridRows, occ = st.occupancy;
      var raw = new Float32Array(o.lanes);
      var total = 0;
      for (var gx = 0; gx < cols; gx++) {
        var lane = Math.min(o.lanes - 1, (gx * o.lanes / cols) | 0);
        for (var gy = 0; gy < rows; gy++) {
          var e = occ[gy * cols + gx];
          raw[lane] += e;
          total += e;
        }
      }
      for (var l = 0; l < o.lanes; l++) {
        var norm = total > 0 ? raw[l] / total : 0;
        laneWeights[l] += (norm - laneWeights[l]) * (1 - o.laneSmoothing);
      }

      // --- centroid (weighted by body mass)
      var sx = 0, sy = 0, m = 0;
      for (var gy2 = 0; gy2 < rows; gy2++) {
        for (var gx2 = 0; gx2 < cols; gx2++) {
          var w = occ[gy2 * cols + gx2];
          if (w <= 0) continue;
          sx += (gx2 + 0.5) / cols * w;
          sy += (gy2 + 0.5) / rows * w;
          m += w;
        }
      }
      var centroid = m > 0 ? { x: sx / m, y: sy / m } : null;
      lastCentroid = centroid;

      // --- jump detection against a self-learned baseline
      if (centroid) {
        if (baselineY === null) baselineY = centroid.y;
        var rise = baselineY - centroid.y;      // positive = moved upwards
        if (!airborne) {
          // Learn the resting height only while on the ground.
          baselineY += (centroid.y - baselineY) * o.baselineRate;
          if (rise >= o.jumpRise && (nowMs - lastJumpAt) > o.refractoryMs) {
            airborne = true;
            jumped = true;
            lastJumpAt = nowMs;
            peakRise = rise;
          }
        } else {
          if (rise > peakRise) peakRise = rise;
          if (rise < peakRise * o.landFraction) airborne = false;
        }
      }
      return result(jumped);
    }

    function result(jumped) {
      var best = -1, bestV = 0;
      for (var i = 0; i < o.lanes; i++) {
        if (laneWeights[i] > bestV) { bestV = laneWeights[i]; best = i; }
      }
      return {
        present: present,
        lane: bestV > 0.18 ? best : null,
        laneConfidence: bestV,
        laneWeights: laneWeights,
        centroid: lastCentroid,
        baselineY: baselineY,
        jumped: jumped,
        airborne: airborne,
        jumpHeight: airborne ? peakRise : 0
      };
    }

    return { update: update, reset: reset, options: o };
  }

  root.HoloVision = {
    createDetector: createDetector,
    createPlayerTracker: createPlayerTracker,
    toGrayscale: toGrayscale,
    openMask: openMask,
    DEFAULTS: DEFAULTS
  };
})(typeof globalThis !== 'undefined' ? globalThis : this);
