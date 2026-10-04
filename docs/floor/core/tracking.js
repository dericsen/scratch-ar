/* Player tracking: pose landmarks -> stable players standing on grid tiles.
 *
 * Three problems get solved here, and each one is a place where a naive
 * implementation produces something that looks broken on screen:
 *
 * 1. WHERE IS THE PLAYER STANDING?
 *    Not the bounding-box centre -- that sits around the chest and drifts
 *    wildly when someone leans or raises an arm. The contact point with the
 *    floor is the feet, so we use the heel and foot-index landmarks (falling
 *    back to the ankle, then to the lowest visible landmark) and take the
 *    midpoint of whichever feet are actually visible.
 *
 * 2. JITTER.
 *    Pose landmarks wobble by a few pixels every frame even when the subject
 *    is frozen, which makes a marker buzz and, worse, makes the detected tile
 *    flicker between neighbours when someone stands near a boundary. Fixed by
 *    a One Euro filter (adaptive low-pass: heavy smoothing when still, light
 *    when moving fast, so it does not feel laggy) plus hysteresis on the tile
 *    decision itself.
 *
 * 3. WHO IS WHO?
 *    MediaPipe returns a list of poses per frame with no identity attached --
 *    the order can change between frames. Identities are maintained by
 *    matching this frame's detections to existing tracks by nearest position
 *    on the floor, globally rather than greedily per track, with a timeout so
 *    a player who is briefly occluded keeps their ID and score.
 */
(function (root) {
  'use strict';

  var ns = root.CVFloor = root.CVFloor || {};

  // BlazePose 33-landmark topology (MediaPipe Pose / PoseLandmarker).
  var LM = {
    NOSE: 0,
    LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
    LEFT_HIP: 23, RIGHT_HIP: 24,
    LEFT_KNEE: 25, RIGHT_KNEE: 26,
    LEFT_ANKLE: 27, RIGHT_ANKLE: 28,
    LEFT_HEEL: 29, RIGHT_HEEL: 30,
    LEFT_FOOT_INDEX: 31, RIGHT_FOOT_INDEX: 32
  };

  var SKELETON_EDGES = [
    [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER],
    [LM.LEFT_SHOULDER, LM.LEFT_HIP], [LM.RIGHT_SHOULDER, LM.RIGHT_HIP],
    [LM.LEFT_HIP, LM.RIGHT_HIP],
    [LM.LEFT_HIP, LM.LEFT_KNEE], [LM.LEFT_KNEE, LM.LEFT_ANKLE],
    [LM.RIGHT_HIP, LM.RIGHT_KNEE], [LM.RIGHT_KNEE, LM.RIGHT_ANKLE],
    [LM.LEFT_ANKLE, LM.LEFT_HEEL], [LM.LEFT_HEEL, LM.LEFT_FOOT_INDEX],
    [LM.RIGHT_ANKLE, LM.RIGHT_HEEL], [LM.RIGHT_HEEL, LM.RIGHT_FOOT_INDEX]
  ];

  var MIN_VIS = 0.5;

  function visible(lm, idx) {
    var p = lm[idx];
    if (!p) return false;
    // PoseLandmarker reports visibility (and sometimes presence); treat a
    // missing field as visible so the code still works with sources that
    // do not provide it, such as the mock generator.
    var v = (p.visibility === undefined) ? 1 : p.visibility;
    return v >= MIN_VIS && isFinite(p.x) && isFinite(p.y);
  }

  /* Ground contact point for one foot, in normalised image coordinates.
   * Heel and toe straddle the actual contact patch, so their midpoint is a
   * better floor position than the ankle, which sits a little above it. */
  function footPoint(lm, heelIdx, toeIdx, ankleIdx) {
    var hasHeel = visible(lm, heelIdx), hasToe = visible(lm, toeIdx);
    if (hasHeel && hasToe) {
      return {
        x: (lm[heelIdx].x + lm[toeIdx].x) / 2,
        y: (lm[heelIdx].y + lm[toeIdx].y) / 2,
        quality: 1.0
      };
    }
    if (hasHeel) return { x: lm[heelIdx].x, y: lm[heelIdx].y, quality: 0.8 };
    if (hasToe) return { x: lm[toeIdx].x, y: lm[toeIdx].y, quality: 0.8 };
    if (visible(lm, ankleIdx)) {
      return { x: lm[ankleIdx].x, y: lm[ankleIdx].y, quality: 0.6 };
    }
    return null;
  }

  /* Turn one pose into a floor observation, in pixel coordinates.
   *
   * Returns the individual feet when visible plus the stance point the game
   * actually uses. When no foot landmark is usable at all (legs out of frame
   * or occluded by furniture) we fall back to the horizontal position of the
   * hips combined with the lowest visible landmark, and flag the low quality
   * so the caller can treat it as a guess.
   */
  function extractObservation(landmarks, frameW, frameH) {
    if (!landmarks || landmarks.length < 33) return null;
    var left = footPoint(landmarks, LM.LEFT_HEEL, LM.LEFT_FOOT_INDEX, LM.LEFT_ANKLE);
    var right = footPoint(landmarks, LM.RIGHT_HEEL, LM.RIGHT_FOOT_INDEX, LM.RIGHT_ANKLE);

    var stance = null, quality = 0;
    if (left && right) {
      stance = { x: (left.x + right.x) / 2, y: (left.y + right.y) / 2 };
      quality = Math.min(left.quality, right.quality);
    } else if (left || right) {
      var f = left || right;
      stance = { x: f.x, y: f.y };
      quality = f.quality * 0.85;      // one foot: usable, but less certain
    } else {
      var hipL = visible(landmarks, LM.LEFT_HIP), hipR = visible(landmarks, LM.RIGHT_HIP);
      if (!hipL && !hipR) return null;
      var hx = hipL && hipR
        ? (landmarks[LM.LEFT_HIP].x + landmarks[LM.RIGHT_HIP].x) / 2
        : (hipL ? landmarks[LM.LEFT_HIP].x : landmarks[LM.RIGHT_HIP].x);
      var lowest = 0;
      for (var i = 0; i < landmarks.length; i++) {
        if (visible(landmarks, i) && landmarks[i].y > lowest) lowest = landmarks[i].y;
      }
      stance = { x: hx, y: lowest };
      quality = 0.3;
    }

    function px(p) { return p ? { x: p.x * frameW, y: p.y * frameH } : null; }

    return {
      stancePixel: px(stance),
      leftPixel: px(left),
      rightPixel: px(right),
      quality: quality,
      landmarks: landmarks
    };
  }

  /* One Euro filter over a 2-D point.
   *
   * A plain exponential average forces a single trade-off: smooth but laggy,
   * or responsive but jittery. This adapts -- the cutoff frequency rises with
   * the measured speed, so a stationary player is filtered hard (no buzz) and
   * a running player is barely filtered (no rubber-banding). The adaptive term
   * uses the speed magnitude so x and y stay consistent with each other.
   */
  function createPointFilter(opts) {
    var o = opts || {};
    var minCutoff = o.minCutoff === undefined ? 0.8 : o.minCutoff;
    var beta = o.beta === undefined ? 0.45 : o.beta;
    var dCutoff = o.dCutoff === undefined ? 1.0 : o.dCutoff;
    var x = null, y = null, dxHat = 0, dyHat = 0;

    function alpha(cutoff, dt) {
      var tau = 1 / (2 * Math.PI * cutoff);
      return 1 / (1 + tau / dt);
    }

    return {
      reset: function () { x = y = null; dxHat = dyHat = 0; },
      get value() { return x === null ? null : { x: x, y: y }; },
      update: function (nx, ny, dt) {
        if (!(dt > 0)) dt = 1 / 30;
        if (x === null) { x = nx; y = ny; return { x: x, y: y }; }
        var ad = alpha(dCutoff, dt);
        var dx = (nx - x) / dt, dy = (ny - y) / dt;
        dxHat = dxHat + ad * (dx - dxHat);
        dyHat = dyHat + ad * (dy - dyHat);
        var speed = Math.hypot(dxHat, dyHat);
        var a = alpha(minCutoff + beta * speed, dt);
        x = x + a * (nx - x);
        y = y + a * (ny - y);
        return { x: x, y: y };
      }
    };
  }

  /* Commit to a new tile only on real evidence.
   *
   * Without this, a player standing on a boundary makes the game flip between
   * two tiles many times a second -- which would score and kill them at
   * random. A change is accepted once the smoothed point is either clearly
   * inside the new tile, or has stayed there for a few consecutive frames.
   */
  function createTileLatch(opts) {
    var o = opts || {};
    var minFrames = o.minFrames === undefined ? 3 : o.minFrames;
    var commitInset = o.commitInset === undefined ? 0.18 : o.commitInset;
    var current = null, candidate = null, count = 0;

    function same(a, b) {
      if (!a || !b) return a === b;
      return a.row === b.row && a.col === b.col;
    }

    return {
      get tile() { return current; },
      reset: function () { current = candidate = null; count = 0; },
      update: function (tile, inset) {
        if (same(tile, current)) { candidate = null; count = 0; return current; }
        if (current === null) { current = tile; candidate = null; count = 0; return current; }
        if (same(tile, candidate)) count++;
        else { candidate = tile; count = 1; }
        var deepInside = tile !== null && inset !== null && inset >= commitInset;
        if (count >= minFrames || deepInside) {
          current = tile;
          candidate = null;
          count = 0;
        }
        return current;
      }
    };
  }

  /* Global nearest-neighbour assignment.
   *
   * All (track, detection) pairs under the distance threshold are sorted by
   * distance and consumed in that order. Compared with looping over tracks and
   * grabbing each one's nearest detection, this cannot have an early track
   * steal a detection that belonged much more clearly to a later one -- the
   * failure that makes two players swap identities when they pass each other.
   */
  function assign(tracks, observations, maxDistance, distanceFn) {
    var pairs = [];
    for (var t = 0; t < tracks.length; t++) {
      for (var d = 0; d < observations.length; d++) {
        var dist = distanceFn(tracks[t], observations[d]);
        if (dist !== null && dist <= maxDistance) {
          pairs.push({ t: t, d: d, dist: dist });
        }
      }
    }
    pairs.sort(function (a, b) { return a.dist - b.dist; });
    var usedT = {}, usedD = {}, matches = [];
    for (var i = 0; i < pairs.length; i++) {
      var p = pairs[i];
      if (usedT[p.t] || usedD[p.d]) continue;
      usedT[p.t] = usedD[p.d] = true;
      matches.push(p);
    }
    var freeD = [], freeT = [];
    for (var j = 0; j < observations.length; j++) if (!usedD[j]) freeD.push(j);
    for (var k = 0; k < tracks.length; k++) if (!usedT[k]) freeT.push(k);
    return { matches: matches, unmatchedObservations: freeD, unmatchedTracks: freeT };
  }

  function createPlayerTracker(opts) {
    var o = {
      maxMatchDistance: 0.3,     // in floor units (1.0 = full floor width)
      lostTimeoutMs: 1500,       // keep an occluded player's ID this long
      minFramesToConfirm: 2,     // suppress one-frame false detections
      filter: { minCutoff: 0.8, beta: 0.45 },
      latch: { minFrames: 3, commitInset: 0.18 }
    };
    if (opts) for (var k in opts) if (opts[k] !== undefined) o[k] = opts[k];

    var tracks = [];
    var nextId = 1;

    function reset() { tracks.length = 0; nextId = 1; }

    function distanceFn(track, obs) {
      if (!track.floor || !obs.floor) return null;
      return Math.hypot(track.floor.x - obs.floor.x, track.floor.y - obs.floor.y);
    }

    /* `poses` is an array of landmark arrays (normalised image coords). */
    function update(poses, calib, nowMs, dtSeconds, frameW, frameH) {
      var observations = [];
      if (poses && calib) {
        for (var i = 0; i < poses.length; i++) {
          var obs = extractObservation(poses[i], frameW, frameH);
          if (!obs || !obs.stancePixel) continue;
          var f = calib.toFloor(obs.stancePixel.x, obs.stancePixel.y);
          if (!f) continue;
          obs.floor = { x: f[0], y: f[1] };
          observations.push(obs);
        }
      }

      var res = assign(tracks, observations, o.maxMatchDistance, distanceFn);

      for (var m = 0; m < res.matches.length; m++) {
        applyObservation(tracks[res.matches[m].t], observations[res.matches[m].d],
                         calib, nowMs, dtSeconds);
      }

      for (var n = 0; n < res.unmatchedObservations.length; n++) {
        var track = newTrack(nextId++);
        tracks.push(track);
        applyObservation(track, observations[res.unmatchedObservations[n]],
                         calib, nowMs, dtSeconds);
      }

      for (var u = 0; u < res.unmatchedTracks.length; u++) {
        var lost = tracks[res.unmatchedTracks[u]];
        lost.missingMs = nowMs - lost.lastSeenMs;
        lost.visible = false;
      }

      // Drop tracks that have been gone too long to still be the same person.
      for (var z = tracks.length - 1; z >= 0; z--) {
        if (!tracks[z].visible && (nowMs - tracks[z].lastSeenMs) > o.lostTimeoutMs) {
          tracks.splice(z, 1);
        }
      }

      return activePlayers();
    }

    function newTrack(id) {
      return {
        id: id,
        frames: 0,
        confirmed: false,
        visible: false,
        floor: null,
        pixel: null,
        leftPixel: null,
        rightPixel: null,
        quality: 0,
        tile: null,
        landmarks: null,
        lastSeenMs: 0,
        missingMs: 0,
        filter: createPointFilter(o.filter),
        latch: createTileLatch(o.latch)
      };
    }

    function applyObservation(track, obs, calib, nowMs, dt) {
      track.frames++;
      track.visible = true;
      track.lastSeenMs = nowMs;
      track.missingMs = 0;
      track.quality = obs.quality;
      track.landmarks = obs.landmarks;
      track.leftPixel = obs.leftPixel;
      track.rightPixel = obs.rightPixel;

      // Smooth in floor space, not pixel space: a fixed pixel wobble is a much
      // larger real-world error far from the camera than near it, and floor
      // space is where the game logic lives.
      var sm = track.filter.update(obs.floor.x, obs.floor.y, dt);
      track.floor = { x: sm.x, y: sm.y };
      var back = calib.toCanvas(sm.x, sm.y);
      track.pixel = back ? { x: back[0], y: back[1] } : obs.stancePixel;

      var tile = calib.tileAtFloor(sm.x, sm.y);
      var inset = tile ? calib.insetInTile(sm.x, sm.y) : null;
      track.tile = track.latch.update(tile, inset);
      if (track.frames >= o.minFramesToConfirm) track.confirmed = true;
    }

    function activePlayers() {
      var out = [];
      for (var i = 0; i < tracks.length; i++) {
        if (tracks[i].confirmed) out.push(tracks[i]);
      }
      out.sort(function (a, b) { return a.id - b.id; });
      return out;
    }

    return {
      update: update,
      reset: reset,
      get players() { return activePlayers(); },
      get allTracks() { return tracks.slice(); },
      options: o
    };
  }

  ns.tracking = {
    LM: LM,
    SKELETON_EDGES: SKELETON_EDGES,
    MIN_VIS: MIN_VIS,
    footPoint: footPoint,
    extractObservation: extractObservation,
    createPointFilter: createPointFilter,
    createTileLatch: createTileLatch,
    assign: assign,
    createPlayerTracker: createPlayerTracker
  };
})(typeof globalThis !== 'undefined' ? globalThis : this);
