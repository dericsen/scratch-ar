/* Pose sources.
 *
 * Deliberately thin. Everything interesting -- calibration, tile mapping,
 * tracking, game rules -- is in core/ and testable without a camera. This file
 * is the only place that knows MediaPipe exists, so if Google changes the API
 * this is the single file to fix.
 *
 * Two interchangeable implementations behind one interface:
 *
 *   createMediaPipeSource()  real pose estimation from the webcam
 *   createMockSource()       synthetic walkers, for trying the game with no
 *                            camera, for demoing on a machine that blocks
 *                            camera access, and for isolating a bug to either
 *                            the vision half or the game half
 *
 * Both expose: detect(video, timestampMs) -> { landmarks: Array<Array<{x,y,z,visibility}>> }
 * with landmark coordinates normalised to [0,1] over the frame, which is the
 * format MediaPipe already uses.
 */
(function (root) {
  'use strict';

  var ns = root.CVFloor = root.CVFloor || {};

  // Pinned first, then a floating fallback: a pinned version keeps the API
  // stable, but if that exact build ever disappears from the CDN the game
  // should still start rather than show a blank screen.
  var TASKS_VERSIONS = ['0.10.14', 'latest'];

  var MODEL_URLS = {
    lite: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
    full: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
  };

  /* Real MediaPipe PoseLandmarker in VIDEO mode.
   *
   * `numPoses` is what makes this multi-person -- the default is 1, which is
   * the trap behind most "why does it only track one person" questions. The
   * lite model is the default here because the floor game needs the ankle
   * positions at frame rate, not millimetre-accurate fingers. */
  async function createMediaPipeSource(opts) {
    var o = Object.assign({
      numPoses: 4,
      model: 'lite',
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      onStatus: null
    }, opts || {});

    function status(s) { if (o.onStatus) o.onStatus(s); }

    var lastError = null;
    for (var i = 0; i < TASKS_VERSIONS.length; i++) {
      var v = TASKS_VERSIONS[i];
      try {
        status('Loading MediaPipe (' + v + ')...');
        var mod = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@' + v);
        var FilesetResolver = mod.FilesetResolver, PoseLandmarker = mod.PoseLandmarker;
        if (!FilesetResolver || !PoseLandmarker) throw new Error('unexpected module shape');
        var fileset = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@' + v + '/wasm');

        // GPU first, CPU as a fallback: the WebGL delegate fails outright on
        // some integrated graphics and in some virtualised environments.
        var landmarker = null, delegateUsed = null;
        var delegates = ['GPU', 'CPU'];
        for (var d = 0; d < delegates.length; d++) {
          try {
            status('Creating landmarker (' + delegates[d] + ')...');
            landmarker = await PoseLandmarker.createFromOptions(fileset, {
              baseOptions: {
                modelAssetPath: MODEL_URLS[o.model] || MODEL_URLS.lite,
                delegate: delegates[d]
              },
              runningMode: 'VIDEO',
              numPoses: o.numPoses,
              minPoseDetectionConfidence: o.minPoseDetectionConfidence,
              minPosePresenceConfidence: o.minPosePresenceConfidence,
              minTrackingConfidence: o.minTrackingConfidence,
              outputSegmentationMasks: false
            });
            delegateUsed = delegates[d];
            break;
          } catch (errDelegate) {
            lastError = errDelegate;
          }
        }
        if (!landmarker) throw lastError || new Error('landmarker creation failed');

        var lastTs = -1;
        return {
          kind: 'mediapipe',
          version: v,
          delegate: delegateUsed,
          numPoses: o.numPoses,
          detect: function (video, timestampMs) {
            // MediaPipe requires strictly increasing timestamps in VIDEO mode
            // and throws if one repeats, which happens whenever the render
            // loop runs faster than the camera produces frames.
            if (timestampMs <= lastTs) return null;
            lastTs = timestampMs;
            var res = landmarker.detectForVideo(video, timestampMs);
            if (!res || !res.landmarks) return { landmarks: [] };
            return { landmarks: res.landmarks };
          },
          dispose: function () { try { landmarker.close(); } catch (e) { /* already gone */ } }
        };
      } catch (err) {
        lastError = err;
      }
    }
    throw lastError || new Error('could not load MediaPipe');
  }

  /* Synthetic walkers.
   *
   * Produces the same 33-landmark layout as MediaPipe for a configurable
   * number of people strolling between random points on the calibrated floor,
   * complete with a little landmark noise so the smoothing and the tile
   * hysteresis are actually exercised. Give it the calibration and the fake
   * people stand on the real grid. */
  function createMockSource(opts) {
    var o = Object.assign({ count: 2, seed: 5, noisePx: 3, speed: 0.22 }, opts || {});
    var calib = null;
    var frameW = 1280, frameH = 720;
    var seed = o.seed >>> 0;
    function rand() {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 4294967296;
    }

    var walkers = [];
    for (var i = 0; i < o.count; i++) {
      walkers.push({
        pos: { x: 0.2 + 0.6 * rand(), y: 0.3 + 0.6 * rand() },
        target: { x: rand(), y: rand() },
        phase: rand() * 6.28
      });
    }

    function retarget(w) {
      w.target = { x: 0.05 + 0.9 * rand(), y: 0.05 + 0.9 * rand() };
    }

    function landmarksFor(w, tMs) {
      // Place the body above the feet in *pixel* space, since that is how a
      // real camera sees a standing person.
      var px = calib ? calib.toCanvas(w.pos.x, w.pos.y) : [frameW / 2, frameH * 0.8];
      if (!px) px = [frameW / 2, frameH * 0.8];
      var n = o.noisePx;
      var cx = px[0] + (rand() - 0.5) * 2 * n;
      var cy = px[1] + (rand() - 0.5) * 2 * n;
      // Perspective: someone on a far tile appears smaller.
      var scale = 0.55 + 0.75 * w.pos.y;
      var bob = Math.sin(tMs / 220 + w.phase) * 4;
      var nx = cx / frameW, ny = cy / frameH;
      var dx = (17 * scale) / frameW;
      var up = function (pxUp) { return ny - (pxUp * scale + bob) / frameH; };

      var lm = [];
      for (var k = 0; k < 33; k++) lm.push({ x: nx, y: ny, z: 0, visibility: 0.2 });
      var L = ns.tracking.LM;
      function set(idx, x, y) { lm[idx] = { x: x, y: y, z: 0, visibility: 0.97 }; }
      var swing = Math.sin(tMs / 180 + w.phase) * (6 * scale) / frameW;
      set(L.LEFT_HEEL, nx - dx + swing, ny);
      set(L.LEFT_FOOT_INDEX, nx - dx + swing, up(7));
      set(L.RIGHT_HEEL, nx + dx - swing, ny);
      set(L.RIGHT_FOOT_INDEX, nx + dx - swing, up(7));
      set(L.LEFT_ANKLE, nx - dx + swing, up(16));
      set(L.RIGHT_ANKLE, nx + dx - swing, up(16));
      set(L.LEFT_KNEE, nx - dx, up(95));
      set(L.RIGHT_KNEE, nx + dx, up(95));
      set(L.LEFT_HIP, nx - dx * 0.9, up(175));
      set(L.RIGHT_HIP, nx + dx * 0.9, up(175));
      set(L.LEFT_SHOULDER, nx - dx * 1.5, up(265));
      set(L.RIGHT_SHOULDER, nx + dx * 1.5, up(265));
      set(L.NOSE, nx, up(305));
      return lm;
    }

    var lastMs = null;
    return {
      kind: 'mock',
      numPoses: o.count,
      setCalibration: function (c) { calib = c; },
      setFrameSize: function (w, h) { frameW = w; frameH = h; },
      detect: function (_video, timestampMs) {
        var dt = lastMs === null ? 1 / 30 : Math.min(0.1, (timestampMs - lastMs) / 1000);
        lastMs = timestampMs;
        var out = [];
        for (var i = 0; i < walkers.length; i++) {
          var w = walkers[i];
          var dxp = w.target.x - w.pos.x, dyp = w.target.y - w.pos.y;
          var dist = Math.hypot(dxp, dyp);
          if (dist < 0.03) retarget(w);
          else {
            w.pos.x += (dxp / dist) * o.speed * dt;
            w.pos.y += (dyp / dist) * o.speed * dt;
          }
          out.push(landmarksFor(w, timestampMs));
        }
        return { landmarks: out };
      },
      dispose: function () { walkers.length = 0; }
    };
  }

  ns.pose = {
    createMediaPipeSource: createMediaPipeSource,
    createMockSource: createMockSource,
    MODEL_URLS: MODEL_URLS,
    TASKS_VERSIONS: TASKS_VERSIONS
  };
})(typeof globalThis !== 'undefined' ? globalThis : this);
