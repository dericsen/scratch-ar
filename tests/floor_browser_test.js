/* Smoke tests for the browser-facing halves: render.js and pose.js.
 *
 * These files never execute in the sandbox (no browser, no camera), so without
 * a harness their first run would be on the user's machine -- and a single
 * typo in 400 lines of drawing code produces a blank canvas with a console
 * error. A stubbed 2-D context is enough to prove every drawing path is
 * reachable and calls only real API methods, and the mock pose source is
 * exercised through the real tracker so the no-camera fallback is known to
 * work end to end.
 *
 * Run: node tests/floor_browser_test.js
 */
'use strict';

const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..', 'docs', 'floor');
for (const f of ['core/geometry.js', 'core/tracking.js', 'core/questions.js', 'core/game.js']) {
  eval(fs.readFileSync(path.join(root, f), 'utf8'));
}

let failures = [];
let passed = 0;
class AssertErr extends Error {}
function ok(c, m) { if (!c) throw new AssertErr(m || 'assertion failed'); }
function test(name, fn) {
  try {
    fn(); passed++;
    console.log('  ' + name.padEnd(62) + ' ok');
  } catch (e) {
    failures.push([name, e.message, e.stack]);
    console.log('  ' + name.padEnd(62) + (e instanceof AssertErr ? ' FAIL' : ' ERROR'));
  }
}

// --------------------------------------------------------------- canvas stub
/* Records every call so a test can assert that drawing actually happened, and
 * throws on any unknown property so a typo like ctx.fillStyel is caught rather
 * than silently ignored (which is what a plain object would do). */
function makeCtxStub() {
  const calls = { fill: 0, stroke: 0, fillText: 0, drawImage: 0, fillRect: 0, arc: 0 };
  const target = {
    canvas: null,
    fillStyle: '', strokeStyle: '', lineWidth: 1, font: '', textAlign: '',
    textBaseline: '', globalAlpha: 1, globalCompositeOperation: '',
    imageSmoothingEnabled: true, lineCap: '', lineJoin: '',
    save() {}, restore() {}, translate() {}, scale() {}, rotate() {},
    setTransform() {}, clearRect() {},
    fillRect() { calls.fillRect++; }, strokeRect() {},
    beginPath() {}, closePath() {}, moveTo() {}, lineTo() {},
    arc() { calls.arc++; }, ellipse() {},
    fill() { calls.fill++; }, stroke() { calls.stroke++; },
    setLineDash() {},
    fillText() { calls.fillText++; }, strokeText() {},
    measureText(s) { return { width: String(s).length * 7 }; },
    drawImage() { calls.drawImage++; },
    createLinearGradient() { return { addColorStop() {} }; },
    createRadialGradient() { return { addColorStop() {} }; },
    createImageData(w, h) { return { data: new Uint8ClampedArray(w * h * 4), width: w, height: h }; },
    putImageData() {},
    getImageData(x, y, w, h) { return { data: new Uint8ClampedArray(w * h * 4), width: w, height: h }; },
    _calls: calls
  };
  return new Proxy(target, {
    get(t, prop) {
      if (prop in t) return t[prop];
      throw new Error('render.js used an unsupported canvas property: ' + String(prop));
    },
    set(t, prop, value) {
      if (!(prop in t)) throw new Error('render.js set an unknown canvas property: ' + String(prop));
      t[prop] = value;
      return true;
    }
  });
}

function makeCanvasStub(w, h) {
  const ctx = makeCtxStub();
  const c = {
    width: w, height: h,
    getContext() { return ctx; },
    getBoundingClientRect() { return { left: 0, top: 0, width: w, height: h }; },
    addEventListener() {},
    style: {}
  };
  ctx.canvas = c;
  return c;
}

globalThis.window = globalThis;
globalThis.document = { getElementById: () => null, createElement: () => makeCanvasStub(16, 16) };
if (!globalThis.performance) globalThis.performance = { now: () => Date.now() };

eval(fs.readFileSync(path.join(root, 'render.js'), 'utf8'));
eval(fs.readFileSync(path.join(root, 'pose.js'), 'utf8'));

const CV = globalThis.CVFloor;
const FRAME_W = 1280, FRAME_H = 720;
const QUAD = [[430, 300], [870, 300], [1180, 670], [110, 670]];
const CALIB = CV.geometry.createCalibration(QUAD, 8, 8);

console.log('floor browser layer (node ' + process.version + ')\n');

// ------------------------------------------------------------------- render
test('renderer constructs against a canvas', () => {
  const r = CV.render.createRenderer(makeCanvasStub(FRAME_W, FRAME_H));
  ok(r && typeof r.drawTiles === 'function', 'renderer missing drawTiles');
});

test('every tile state draws without touching an unknown canvas API', () => {
  const canvas = makeCanvasStub(FRAME_W, FRAME_H);
  const r = CV.render.createRenderer(canvas);
  const states = ['empty', 'safe', 'danger', 'cracked', 'broken', 'answer', 'correct', 'wrong'];
  const tiles = [];
  for (let i = 0; i < states.length; i++) {
    tiles.push({
      row: i, col: i, state: states[i],
      symbol: (states[i] === 'answer' ? 'O(log n)' : null),
      optionIndex: 0, anim: 0.5, flash: 0.5
    });
  }
  r.clear();
  r.drawVideo(null);
  r.drawTiles(CALIB, tiles, true, 'active');
  const calls = canvas.getContext()._calls;
  ok(calls.fill > 0 && calls.stroke > 0, 'nothing was actually drawn');
  ok(calls.fillText > 0, 'no tile text or labels drawn');
});

test('long answer text is shrunk to fit its tile', () => {
  const canvas = makeCanvasStub(FRAME_W, FRAME_H);
  const ctx = canvas.getContext();
  const r = CV.render.createRenderer(canvas);
  const fonts = [];
  // Capture the font sizes chosen for a far (small) tile vs a near (big) one.
  const original = Object.getOwnPropertyDescriptor(ctx, 'font');
  let last = '';
  Object.defineProperty(ctx, 'font', {
    get() { return last; },
    set(v) { last = v; fonts.push(v); },
    configurable: true
  });
  r.drawTiles(CALIB, [
    { row: 0, col: 0, state: 'answer', symbol: 'O(n log n)', optionIndex: 0, anim: 0, flash: 0 },
    { row: 7, col: 0, state: 'answer', symbol: 'O(n log n)', optionIndex: 1, anim: 0, flash: 0 }
  ], false, 'active');
  if (original) Object.defineProperty(ctx, 'font', original);
  const sizes = fonts.map(f => parseFloat(f.replace(/[^0-9.]/g, ' ').trim())).filter(n => n > 0);
  ok(sizes.length >= 2, 'no fonts were set');
  ok(Math.min.apply(null, sizes) >= 9, 'font shrank below the legible floor');
  ok(Math.max.apply(null, sizes) <= 34, 'font exceeded the cap');
});

test('player markers, skeleton and nameplate all render', () => {
  const canvas = makeCanvasStub(FRAME_W, FRAME_H);
  const r = CV.render.createRenderer(canvas);
  const tr = CV.tracking.createPlayerTracker();
  const src = CV.pose.createMockSource({ count: 2, seed: 3 });
  src.setCalibration(CALIB);
  src.setFrameSize(FRAME_W, FRAME_H);
  let players = [];
  for (let i = 0, t = 0; i < 20; i++, t += 33) {
    players = tr.update(src.detect(null, t).landmarks, CALIB, t, 1 / 30, FRAME_W, FRAME_H);
  }
  ok(players.length === 2, 'mock source did not produce 2 trackable players');
  const g = CV.game.createGame({ seed: 5 });
  g.start();
  const view = g.playersView(players);
  const byId = {};
  view.forEach(v => { byId[v.id] = v; });
  const before = canvas.getContext()._calls.fill;
  for (const p of players) {
    r.drawPlayerMarker(CALIB, p, byId[p.id], true, FRAME_W, FRAME_H);
  }
  ok(canvas.getContext()._calls.fill > before, 'markers drew nothing');
});

test('a marker for a player with no floor position is skipped safely', () => {
  const r = CV.render.createRenderer(makeCanvasStub(FRAME_W, FRAME_H));
  r.drawPlayerMarker(CALIB, { id: 1, floor: null, pixel: null, landmarks: null },
                     { tileLabel: '--', score: 0, lives: 3, status: 'ALIVE' },
                     true, FRAME_W, FRAME_H);
});

test('calibration overlay renders at every stage of corner placement', () => {
  const canvas = makeCanvasStub(FRAME_W, FRAME_H);
  const r = CV.render.createRenderer(canvas);
  for (let n = 0; n <= 4; n++) {
    r.drawCalibrationOverlay(QUAD.slice(0, n), n, n === 4 ? CALIB : null);
  }
  ok(canvas.getContext()._calls.arc > 0, 'corner dots were not drawn');
});

test('particles spawn, decay and disappear', () => {
  const r = CV.render.createRenderer(makeCanvasStub(FRAME_W, FRAME_H));
  r.burst(100, 100, [255, 0, 0], 20);
  ok(r.particleCount === 20, 'expected 20 particles, got ' + r.particleCount);
  r.drawParticles();
  for (let i = 0; i < 100; i++) r.stepParticles(1 / 30);
  ok(r.particleCount === 0, 'particles never expired (' + r.particleCount + ' left)');
});

test('banners and the round timer render in every phase', () => {
  const canvas = makeCanvasStub(FRAME_W, FRAME_H);
  const r = CV.render.createRenderer(canvas);
  const g = CV.game.createGame({ seed: 5 });
  g.start();
  r.drawRoundTimer(g.state);
  g.update([{ id: 1, tile: { row: 0, col: 0 } }], 99);   // -> resolve
  r.drawRoundTimer(g.state);
  r.drawBanner(['GAME OVER', 'Press RESTART GAME'], '#ff4842');
  ok(canvas.getContext()._calls.fillText > 0, 'banner text not drawn');
});

test('player colours are distinct and wrap around', () => {
  const seen = new Set();
  for (let i = 1; i <= 6; i++) seen.add(CV.render.playerColor(i).join(','));
  ok(seen.size === 6, 'the first six players must be visually distinct');
  ok(CV.render.playerColor(7).join(',') === CV.render.playerColor(1).join(','),
     'colours should wrap for player 7');
});

// --------------------------------------------------------------------- pose
test('the mock source emits MediaPipe-shaped landmarks', () => {
  const src = CV.pose.createMockSource({ count: 3, seed: 11 });
  src.setCalibration(CALIB);
  src.setFrameSize(FRAME_W, FRAME_H);
  const res = src.detect(null, 100);
  ok(res && Array.isArray(res.landmarks), 'no landmarks array');
  ok(res.landmarks.length === 3, 'expected 3 poses, got ' + res.landmarks.length);
  for (const lm of res.landmarks) {
    ok(lm.length === 33, 'a pose must have 33 landmarks, got ' + lm.length);
    for (const p of lm) {
      ok(typeof p.x === 'number' && typeof p.y === 'number', 'landmark missing coordinates');
      ok(p.visibility !== undefined, 'landmark missing visibility');
      ok(isFinite(p.x) && isFinite(p.y), 'non-finite landmark');
    }
  }
});

test('mock walkers stay on the calibrated floor and get real tiles', () => {
  const src = CV.pose.createMockSource({ count: 3, seed: 21 });
  src.setCalibration(CALIB);
  src.setFrameSize(FRAME_W, FRAME_H);
  const tr = CV.tracking.createPlayerTracker();
  let players = [], withTile = 0, frames = 0;
  for (let i = 0, t = 0; i < 180; i++, t += 33) {
    players = tr.update(src.detect(null, t).landmarks, CALIB, t, 1 / 30, FRAME_W, FRAME_H);
    if (i > 20) {
      frames++;
      for (const p of players) if (p.tile) withTile++;
    }
  }
  ok(players.length === 3, 'expected 3 tracked walkers, got ' + players.length);
  // Walkers aim at points across the whole floor, so they should essentially
  // always be on some tile.
  ok(withTile > frames * 2.4, 'walkers spent too long off the grid (' + withTile + '/' + (frames * 3) + ')');
});

test('mock walkers actually move between tiles over time', () => {
  const src = CV.pose.createMockSource({ count: 1, seed: 31 });
  src.setCalibration(CALIB);
  src.setFrameSize(FRAME_W, FRAME_H);
  const tr = CV.tracking.createPlayerTracker();
  const tiles = new Set();
  for (let i = 0, t = 0; i < 400; i++, t += 33) {
    const players = tr.update(src.detect(null, t).landmarks, CALIB, t, 1 / 30, FRAME_W, FRAME_H);
    if (players[0] && players[0].tile) tiles.add(players[0].tile.row + ',' + players[0].tile.col);
  }
  ok(tiles.size >= 5, 'a walker should visit several tiles, saw ' + tiles.size);
});

test('the mock source survives having no calibration', () => {
  const src = CV.pose.createMockSource({ count: 2 });
  const res = src.detect(null, 10);
  ok(res.landmarks.length === 2, 'should still emit poses without a calibration');
  ok(isFinite(res.landmarks[0][0].x), 'landmarks must stay finite');
});

test('the MediaPipe source is exposed with pinned versions and model URLs', () => {
  ok(typeof CV.pose.createMediaPipeSource === 'function', 'missing createMediaPipeSource');
  ok(CV.pose.TASKS_VERSIONS.length >= 2, 'expected a pinned version plus a fallback');
  ok(/pose_landmarker_lite\.task$/.test(CV.pose.MODEL_URLS.lite), 'lite model URL looks wrong');
  ok(/pose_landmarker_full\.task$/.test(CV.pose.MODEL_URLS.full), 'full model URL looks wrong');
  ok(/^https:\/\//.test(CV.pose.MODEL_URLS.lite), 'model URL must be https');
});

async function atest(name, fn) {
  try {
    await fn(); passed++;
    console.log('  ' + name.padEnd(62) + ' ok');
  } catch (e) {
    failures.push([name, e.message, e.stack]);
    console.log('  ' + name.padEnd(62) + (e instanceof AssertErr ? ' FAIL' : ' ERROR'));
  }
}

(async function () {
  await atest('MediaPipe loading failure rejects rather than hanging', async () => {
    // There is no network in the sandbox, so the dynamic import must fail.
    // The point of the test is that it surfaces as a *rejected promise* --
    // which main.js catches and uses to fall back to mock players -- rather
    // than leaving the app stuck on "Loading pose model..." forever.
    let rejected = false;
    await CV.pose.createMediaPipeSource({ numPoses: 2 })
      .then(function () {}, function () { rejected = true; });
    ok(rejected, 'expected the model load to reject when the CDN is unreachable');
  });

  console.log('\nfloor browser layer: ' + passed + ' passed, ' + failures.length + ' failed');
  if (failures.length) {
    for (const [n, m, s] of failures) {
      console.log('  - ' + n + '\n      ' + m);
      if (/unsupported canvas|unknown canvas|is not a function|Cannot read/.test(m)) {
        console.log(String(s).split('\n').slice(0, 5).join('\n'));
      }
    }
    process.exit(1);
  }
})();
