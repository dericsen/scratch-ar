/* Node test suite for the browser vision core (docs/vision.js).
 *
 * There is no webcam in this environment, so the detector is fed synthetic
 * frames: a textured "room" plus rectangles standing in for a body, a shadow,
 * and a dark object. That cannot prove the game *feels* right, but it does
 * prove the parts that are easy to get silently wrong -- that a motionless
 * body stays detected, that camera noise alone does not light up the mask,
 * that shadows are rejected, that lanes map to the correct side of the frame
 * (including the mirror flip), and that a jump fires exactly once.
 *
 * Run: node tests/vision_test.js
 */
'use strict';

const fs = require('fs');
const path = require('path');

// vision.js assigns to a global, so loading it is just an eval in this scope.
const srcPath = path.join(__dirname, '..', 'docs', 'vision.js');
eval(fs.readFileSync(srcPath, 'utf8'));
const V = globalThis.HoloVision;

const W = 160, H = 120;
let failures = [];
let passed = 0;

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log('  ' + name.padEnd(58) + ' ok');
  } catch (err) {
    failures.push([name, err.message]);
    console.log('  ' + name.padEnd(58) + (err instanceof AssertErr ? ' FAIL' : ' ERROR'));
  }
}

class AssertErr extends Error {}
function ok(cond, msg) { if (!cond) throw new AssertErr(msg || 'assertion failed'); }
function near(a, b, tol, msg) {
  if (!(Math.abs(a - b) <= tol)) {
    throw new AssertErr((msg || 'near') + ': ' + a + ' vs ' + b + ' (tol ' + tol + ')');
  }
}

// Deterministic PRNG so a failure is always reproducible.
function rng(seed) {
  let s = seed >>> 0;
  return function () {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/* A plausible room: horizontal brightness ramp plus some texture, so the
 * background is not a flat colour that would make the test too easy. */
function makeRoom() {
  const f = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const ramp = 70 + 90 * (x / W);
      const texture = 14 * Math.sin(x * 0.4) * Math.cos(y * 0.3);
      f[y * W + x] = Math.max(0, Math.min(255, Math.round(ramp + texture)));
    }
  }
  return f;
}

function addNoise(frame, rand, sigma) {
  const out = new Uint8Array(frame.length);
  for (let i = 0; i < frame.length; i++) {
    // Box-Muller would be tidier; a sum of uniforms is close enough and cheap.
    const n = (rand() + rand() + rand() + rand() - 2) * sigma;
    out[i] = Math.max(0, Math.min(255, Math.round(frame[i] + n)));
  }
  return out;
}

/* Paint a rectangle. mode: 'solid' = fixed value, 'scale' = multiply the
 * background (which is what a shadow does). */
function paintRect(frame, x0, y0, x1, y1, mode, value) {
  const out = Uint8Array.from(frame);
  for (let y = Math.max(0, y0); y < Math.min(H, y1); y++) {
    for (let x = Math.max(0, x0); x < Math.min(W, x1); x++) {
      const i = y * W + x;
      out[i] = mode === 'scale'
        ? Math.max(0, Math.min(255, Math.round(frame[i] * value)))
        : value;
    }
  }
  return out;
}

function calibrate(det, room, rand, sigma, frames) {
  for (let i = 0; i < (frames || 25); i++) {
    det.learnBackground(addNoise(room, rand, sigma === undefined ? 2.5 : sigma));
  }
  ok(det.finishCalibration(), 'finishCalibration returned false');
}

// ---------------------------------------------------------------- grayscale

test('toGrayscale applies BT.601 luma and downscales', () => {
  const srcW = 4, srcH = 2;
  const rgba = new Uint8Array(srcW * srcH * 4);
  for (let i = 0; i < srcW * srcH; i++) {
    rgba[i * 4] = 255; rgba[i * 4 + 1] = 0; rgba[i * 4 + 2] = 0; rgba[i * 4 + 3] = 255;
  }
  const g = V.toGrayscale(rgba, srcW, srcH, 2, 1);
  ok(g.length === 2, 'wrong output length');
  near(g[0], 76, 1.5, 'pure red luma');
});

// -------------------------------------------------------------- calibration

test('calibration derives a threshold above the noise floor', () => {
  const room = makeRoom();
  const quiet = V.createDetector({ width: W, height: H });
  calibrate(quiet, room, rng(1), 1.0);
  const noisy = V.createDetector({ width: W, height: H });
  calibrate(noisy, room, rng(2), 9.0);
  ok(noisy.getThreshold() > quiet.getThreshold(),
     'a noisier camera must get a higher threshold (' +
     noisy.getThreshold().toFixed(1) + ' vs ' + quiet.getThreshold().toFixed(1) + ')');
  ok(quiet.getThreshold() >= 6, 'threshold must never collapse to zero');
});

test('update() refuses to run before calibration', () => {
  const det = V.createDetector({ width: W, height: H });
  let threw = false;
  try { det.update(new Uint8Array(W * H)); } catch (e) { threw = true; }
  ok(threw, 'expected an error when updating an uncalibrated detector');
});

test('an empty noisy room produces an almost empty mask', () => {
  const room = makeRoom();
  const det = V.createDetector({ width: W, height: H });
  const rand = rng(3);
  calibrate(det, room, rand, 4.0);
  let worst = 0;
  for (let i = 0; i < 12; i++) {
    const st = det.update(addNoise(room, rand, 4.0));
    worst = Math.max(worst, st.coverage);
  }
  ok(worst < 0.02, 'noise alone lit up ' + (worst * 100).toFixed(2) + '% of the frame');
});

// ----------------------------------------------------------------- silhouette

test('a motionless body stays detected frame after frame', () => {
  // This is the whole reason for background subtraction over frame
  // differencing: standing still must not make the player disappear.
  const room = makeRoom();
  const det = V.createDetector({ width: W, height: H });
  const rand = rng(4);
  calibrate(det, room, rand, 2.5);
  const body = paintRect(room, 60, 40, 100, 120, 'solid', 235);
  let first = null, last = null;
  for (let i = 0; i < 40; i++) {
    const st = det.update(addNoise(body, rand, 2.5));
    if (i === 0) first = st.coverage;
    last = st.coverage;
  }
  const expected = (40 * 80) / (W * H);
  near(first, expected, expected * 0.35, 'initial coverage');
  near(last, expected, expected * 0.35, 'coverage after 40 still frames');
  ok(last > expected * 0.6, 'the body faded out while holding still');
});

test('occupancy is high on the body and near zero away from it', () => {
  const room = makeRoom();
  const det = V.createDetector({ width: W, height: H });
  const rand = rng(5);
  calibrate(det, room, rand, 2.5);
  const body = paintRect(room, 60, 30, 100, 120, 'solid', 20);
  det.update(addNoise(body, rand, 2.5));
  const on = det.occupancyInCircle(80 / W, 75 / H, 10 / W);
  const off = det.occupancyInCircle(20 / W, 30 / H, 10 / W);
  ok(on.mean > 0.8, 'occupancy on the body was only ' + on.mean.toFixed(2));
  ok(off.mean < 0.05, 'occupancy off the body was ' + off.mean.toFixed(2));
});

test('lowestBodyY tracks the bottom of the silhouette', () => {
  const room = makeRoom();
  const det = V.createDetector({ width: W, height: H });
  const rand = rng(6);
  calibrate(det, room, rand, 2.0);
  const st = det.update(addNoise(paintRect(room, 60, 20, 100, 90, 'solid', 240), rand, 2.0));
  near(st.lowestBodyY, 90 / H, 0.06, 'lowest body point');
});

test('occupancyInBand measures the lava contact region only', () => {
  const room = makeRoom();
  const det = V.createDetector({ width: W, height: H });
  const rand = rng(7);
  calibrate(det, room, rand, 2.0);
  // Body occupies the lower half of the frame only.
  det.update(addNoise(paintRect(room, 40, 60, 120, 120, 'solid', 245), rand, 2.0));
  const low = det.occupancyInBand(0.6, 1.0);
  const high = det.occupancyInBand(0.0, 0.4);
  ok(low.peak > 0.8, 'body not seen in the lower band (peak ' + low.peak.toFixed(2) + ')');
  ok(high.mean < 0.05, 'phantom body in the upper band (' + high.mean.toFixed(3) + ')');
});

// --------------------------------------------------------------- shadows

test('shadows are rejected but dark objects are kept', () => {
  const room = makeRoom();
  const rand = rng(8);
  const det = V.createDetector({ width: W, height: H });
  calibrate(det, room, rand, 2.0);
  // A shadow dims the background to ~75% of its value.
  const shadow = paintRect(room, 20, 70, 70, 120, 'scale', 0.75);
  const shadowState = det.update(addNoise(shadow, rand, 2.0));
  const shadowOcc = det.occupancyInCircle(45 / W, 95 / H, 12 / W);

  const det2 = V.createDetector({ width: W, height: H });
  const rand2 = rng(9);
  calibrate(det2, room, rand2, 2.0);
  // A genuinely dark object is far darker than any shadow of this background.
  const dark = paintRect(room, 20, 70, 70, 120, 'solid', 12);
  det2.update(addNoise(dark, rand2, 2.0));
  const darkOcc = det2.occupancyInCircle(45 / W, 95 / H, 12 / W);

  ok(shadowOcc.mean < 0.25,
     'shadow leaked into the mask (' + shadowOcc.mean.toFixed(2) + ')');
  ok(darkOcc.mean > 0.8,
     'dark clothing was mistaken for shadow (' + darkOcc.mean.toFixed(2) + ')');
  ok(shadowState.coverage < 0.15, 'shadow dominated the frame');
});

test('shadow suppression can be switched off', () => {
  const room = makeRoom();
  const rand = rng(10);
  const det = V.createDetector({ width: W, height: H, shadowSuppression: false });
  calibrate(det, room, rand, 2.0);
  det.update(addNoise(paintRect(room, 20, 70, 70, 120, 'scale', 0.75), rand, 2.0));
  const occ = det.occupancyInCircle(45 / W, 95 / H, 12 / W);
  ok(occ.mean > 0.5, 'with suppression off the shadow should register as body');
});

// ------------------------------------------------------- stale background

test('a wholesale lighting change is reported as a stale background', () => {
  const room = makeRoom();
  const det = V.createDetector({ width: W, height: H });
  const rand = rng(11);
  calibrate(det, room, rand, 2.0);
  const lightsOff = new Uint8Array(W * H).fill(4);
  const st = det.update(lightsOff);
  ok(st.backgroundStale,
     'expected backgroundStale when the whole frame changes (coverage ' +
     st.coverage.toFixed(2) + ')');
});

test('recalibration recovers after the scene changes', () => {
  const room = makeRoom();
  const det = V.createDetector({ width: W, height: H });
  const rand = rng(12);
  calibrate(det, room, rand, 2.0);
  const newRoom = paintRect(room, 0, 0, W, H, 'scale', 0.45);
  ok(det.update(newRoom).backgroundStale, 'should be stale before recalibrating');
  det.resetCalibration();
  ok(!det.isReady(), 'detector should be unready after reset');
  calibrate(det, newRoom, rand, 2.0);
  const st = det.update(addNoise(newRoom, rand, 2.0));
  ok(!st.backgroundStale && st.coverage < 0.05,
     'after recalibration the new room should read as empty (' +
     st.coverage.toFixed(3) + ')');
});

// ------------------------------------------------------------------ opening

test('morphological opening deletes speckles but keeps solid shapes', () => {
  const w = 20, h = 20;
  const src = new Uint8Array(w * h);
  for (let y = 6; y < 14; y++) for (let x = 6; x < 14; x++) src[y * w + x] = 1;
  src[2 * w + 2] = 1;          // lone speckle
  src[17 * w + 3] = 1;         // another
  const dst = new Uint8Array(w * h);
  V.openMask(src, dst, w, h, 1);
  ok(dst[10 * w + 10] === 1, 'solid interior was eroded away');
  ok(dst[2 * w + 2] === 0, 'speckle survived opening');
  ok(dst[17 * w + 3] === 0, 'speckle survived opening');
});

// ------------------------------------------------------------------ tracker

function stateWithBody(det, room, rand, x0, y0, x1, y1) {
  return det.update(addNoise(paintRect(room, x0, y0, x1, y1, 'solid', 240), rand, 2.0));
}

test('lane detection follows the body across the frame', () => {
  const room = makeRoom();
  const det = V.createDetector({ width: W, height: H });
  const rand = rng(13);
  calibrate(det, room, rand, 2.0);
  const tr = V.createPlayerTracker({ lanes: 4 });

  // Far left of the frame -> lane 0. Settle over several frames because lane
  // weights are smoothed.
  let r;
  for (let i = 0; i < 12; i++) r = tr.update(stateWithBody(det, room, rand, 2, 30, 36, 120), i * 33);
  ok(r.lane === 0, 'expected lane 0, got ' + r.lane);

  for (let i = 0; i < 12; i++) r = tr.update(stateWithBody(det, room, rand, 124, 30, 158, 120), 1000 + i * 33);
  ok(r.lane === 3, 'expected lane 3, got ' + r.lane);

  for (let i = 0; i < 12; i++) r = tr.update(stateWithBody(det, room, rand, 62, 30, 98, 120), 2000 + i * 33);
  ok(r.lane === 1 || r.lane === 2, 'expected a middle lane, got ' + r.lane);
});

test('tracker reports nobody when the frame is empty', () => {
  const room = makeRoom();
  const det = V.createDetector({ width: W, height: H });
  const rand = rng(14);
  calibrate(det, room, rand, 2.0);
  const tr = V.createPlayerTracker({ lanes: 4 });
  const r = tr.update(det.update(addNoise(room, rand, 2.0)), 0);
  ok(!r.present, 'an empty room should not report a player');
  ok(r.lane === null, 'an empty room should not report a lane');
});

test('a jump fires exactly once and resets on landing', () => {
  const room = makeRoom();
  const det = V.createDetector({ width: W, height: H });
  const rand = rng(15);
  calibrate(det, room, rand, 2.0);
  const tr = V.createPlayerTracker({ lanes: 4, jumpRise: 0.05 });

  let t = 0;
  // Stand still so the tracker learns the resting height.
  for (let i = 0; i < 25; i++, t += 33) {
    tr.update(stateWithBody(det, room, rand, 60, 40, 100, 120), t);
  }
  // Rise by 18 px (0.15 of frame height) and hold briefly: one jump event.
  let jumps = 0;
  for (let i = 0; i < 6; i++, t += 33) {
    const r = tr.update(stateWithBody(det, room, rand, 60, 22, 100, 102), t);
    if (r.jumped) jumps++;
    ok(r.airborne, 'should be airborne while raised');
  }
  ok(jumps === 1, 'expected exactly 1 jump event, got ' + jumps);

  // Come back down: airborne must clear.
  let r2;
  for (let i = 0; i < 8; i++, t += 33) {
    r2 = tr.update(stateWithBody(det, room, rand, 60, 40, 100, 120), t);
  }
  ok(!r2.airborne, 'still airborne after landing');

  // A second, later jump is allowed.
  let jumps2 = 0;
  for (let i = 0; i < 6; i++, t += 33) {
    const r = tr.update(stateWithBody(det, room, rand, 60, 22, 100, 102), t);
    if (r.jumped) jumps2++;
  }
  ok(jumps2 === 1, 'expected a second jump to register, got ' + jumps2);
});

test('standing still never registers a jump', () => {
  const room = makeRoom();
  const det = V.createDetector({ width: W, height: H });
  const rand = rng(16);
  calibrate(det, room, rand, 3.0);
  const tr = V.createPlayerTracker({ lanes: 4 });
  let jumps = 0;
  for (let i = 0; i < 90; i++) {
    const r = tr.update(stateWithBody(det, room, rand, 60, 40, 100, 120), i * 33);
    if (r.jumped) jumps++;
  }
  ok(jumps === 0, 'noise produced ' + jumps + ' phantom jumps');
});

test('walking sideways does not register as a jump', () => {
  // Horizontal movement changes the centroid's x, not its y. If the baseline
  // logic were keyed on overall movement this would misfire.
  const room = makeRoom();
  const det = V.createDetector({ width: W, height: H });
  const rand = rng(17);
  calibrate(det, room, rand, 2.0);
  const tr = V.createPlayerTracker({ lanes: 4 });
  let jumps = 0;
  for (let i = 0; i < 40; i++) {
    const x = 10 + i * 2;
    const r = tr.update(stateWithBody(det, room, rand, x, 40, x + 36, 120), i * 33);
    if (r.jumped) jumps++;
  }
  ok(jumps === 0, 'sideways walking produced ' + jumps + ' jumps');
});

test('jump threshold is relative, so it works at any body size', () => {
  // A player far from the camera is small and high in frame; a close player is
  // large. The same relative rise must trigger in both cases.
  const room = makeRoom();
  for (const cfg of [{ y0: 50, y1: 110, rise: 10 }, { y0: 20, y1: 120, rise: 14 }]) {
    const det = V.createDetector({ width: W, height: H });
    const rand = rng(18);
    calibrate(det, room, rand, 2.0);
    const tr = V.createPlayerTracker({ lanes: 4, jumpRise: 0.05 });
    let t = 0;
    for (let i = 0; i < 25; i++, t += 33) {
      tr.update(stateWithBody(det, room, rand, 60, cfg.y0, 100, cfg.y1), t);
    }
    let fired = false;
    for (let i = 0; i < 5; i++, t += 33) {
      const r = tr.update(stateWithBody(det, room, rand, 60, cfg.y0 - cfg.rise, 100, cfg.y1 - cfg.rise), t);
      if (r.jumped) fired = true;
    }
    ok(fired, 'no jump detected for body spanning ' + cfg.y0 + '-' + cfg.y1);
  }
});

// ---------------------------------------------------------------- sensitivity

test('sensitivity moves the threshold in the expected direction', () => {
  const room = makeRoom();
  const det = V.createDetector({ width: W, height: H });
  calibrate(det, room, rng(19), 3.0);
  const base = det.getThreshold();
  det.setSensitivity(2.0);
  const high = det.getThreshold();
  det.setSensitivity(0.5);
  const low = det.getThreshold();
  ok(high < base, 'higher sensitivity should lower the threshold');
  ok(low > base, 'lower sensitivity should raise the threshold');
  det.setSensitivity(1000);
  ok(det.getThreshold() >= 6, 'threshold must stay clamped above the noise floor');
});

// --------------------------------------------------------------------- report

console.log('\nvision core: ' + passed + ' passed, ' + failures.length + ' failed');
if (failures.length) {
  for (const [name, msg] of failures) console.log('  - ' + name + '\n      ' + msg);
  process.exit(1);
}
