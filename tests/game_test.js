/* Headless integration test for the HOLO LAVA game loop.
 *
 * There is no browser and no webcam in the development sandbox, so this
 * harness stubs the handful of browser APIs game.js touches (DOM lookups, a
 * canvas 2D context, getUserMedia, localStorage, requestAnimationFrame) and
 * then feeds the game synthetic camera frames: an empty "room" during
 * calibration, and a body-shaped rectangle placed in whichever lane the test
 * wants to stand in.
 *
 * What this proves: the state machine really advances (boot -> calibrate ->
 * ready -> play), landing on the safe tile scores, landing in lava costs a
 * heart, a jump saves you from the wave, and three mistakes end the game.
 *
 * What it cannot prove: that the game *feels* right with a real camera, real
 * lighting and a real person. It also bypasses the mirror flip, since frames
 * are injected after the stage where game.js mirrors the capture.
 *
 * Run: node tests/game_test.js
 */
'use strict';

const fs = require('fs');
const path = require('path');

const PROC_W = 160, PROC_H = 120, LANES = 4;

let failures = [];
let passed = 0;
class AssertErr extends Error {}
function ok(cond, msg) { if (!cond) throw new AssertErr(msg || 'assertion failed'); }

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log('  ' + name.padEnd(58) + ' ok');
  } catch (err) {
    failures.push([name, err.message, err.stack]);
    console.log('  ' + name.padEnd(58) +
                (err instanceof AssertErr ? ' FAIL' : ' ERROR'));
  }
}

// ----------------------------------------------------------------- frame data
function makeRoomGray() {
  const f = new Uint8Array(PROC_W * PROC_H);
  for (let y = 0; y < PROC_H; y++) {
    for (let x = 0; x < PROC_W; x++) {
      f[y * PROC_W + x] = Math.max(0, Math.min(255, Math.round(
        72 + 86 * (x / PROC_W) + 12 * Math.sin(x * 0.35) * Math.cos(y * 0.28))));
    }
  }
  return f;
}

const ROOM = makeRoomGray();
let rngState = 12345;
function rand() {
  rngState = (rngState * 1664525 + 1013904223) >>> 0;
  return rngState / 4294967296;
}

/* Body as a bright rectangle in a given lane. `rise` lifts it to fake a jump. */
function frameWithBody(lane, rise) {
  const g = Uint8Array.from(ROOM);
  if (lane !== null && lane !== undefined) {
    const laneW = PROC_W / LANES;
    const x0 = Math.round(lane * laneW + 5);
    const x1 = Math.round(lane * laneW + laneW - 5);
    const y0 = 34 - (rise || 0);
    const y1 = PROC_H - (rise || 0);
    for (let y = Math.max(0, y0); y < Math.min(PROC_H, y1); y++) {
      for (let x = x0; x < x1; x++) g[y * PROC_W + x] = 240;
    }
  }
  // A little sensor noise so nothing passes only because the input is perfect.
  for (let i = 0; i < g.length; i++) {
    g[i] = Math.max(0, Math.min(255, g[i] + Math.round((rand() - 0.5) * 5)));
  }
  return g;
}

let currentRGBA = new Uint8ClampedArray(PROC_W * PROC_H * 4);
function setFrame(gray) {
  for (let i = 0; i < gray.length; i++) {
    const o = i * 4;
    currentRGBA[o] = currentRGBA[o + 1] = currentRGBA[o + 2] = gray[i];
    currentRGBA[o + 3] = 255;
  }
}
setFrame(ROOM);

// ------------------------------------------------------------- browser stubs
let now = 1000;
let rafCb = null;

function makeCtx() {
  const noop = () => {};
  return {
    canvas: null,
    fillStyle: '', strokeStyle: '', lineWidth: 1, font: '', textAlign: '',
    globalAlpha: 1, globalCompositeOperation: '', imageSmoothingEnabled: true,
    save: noop, restore: noop, translate: noop, scale: noop, rotate: noop,
    fillRect: noop, strokeRect: noop, clearRect: noop, fillText: noop,
    strokeText: noop, beginPath: noop, closePath: noop, moveTo: noop,
    lineTo: noop, arc: noop, fill: noop, stroke: noop, setLineDash: noop,
    drawImage: noop,
    createLinearGradient: () => ({ addColorStop: noop }),
    createRadialGradient: () => ({ addColorStop: noop }),
    measureText: () => ({ width: 10 }),
    createImageData: (w, h) => ({
      data: new Uint8ClampedArray(w * h * 4), width: w, height: h
    }),
    putImageData: noop,
    // The game reads its camera frame through here; this is the injection point.
    getImageData: () => ({ data: currentRGBA, width: PROC_W, height: PROC_H })
  };
}

function makeEl(id) {
  return {
    id,
    textContent: '',
    value: '1',
    _handlers: {},
    addEventListener(type, fn) { (this._handlers[type] = this._handlers[type] || []).push(fn); },
    click() { (this._handlers.click || []).forEach(fn => fn({})); },
    width: 0, height: 0,
    getContext: () => makeCtx(),
    // video bits
    srcObject: null,
    readyState: 4,
    videoWidth: 640,
    videoHeight: 480,
    play: () => Promise.resolve()
  };
}

const elements = {};
['stage', 'cam', 'status', 'btn-start', 'btn-recal', 'btn-debug', 'btn-mute', 'sens']
  .forEach(id => { elements[id] = makeEl(id); });

const store = new Map();

globalThis.window = globalThis;
globalThis.document = {
  getElementById: id => elements[id] || null,
  createElement: tag => makeEl('created-' + tag)
};
// Node >= 21 defines `navigator` as a getter-only global, so plain assignment
// throws; redefine the property instead.
Object.defineProperty(globalThis, 'navigator', {
  value: { mediaDevices: { getUserMedia: () => Promise.resolve({ getTracks: () => [] }) } },
  configurable: true,
  writable: true
});
globalThis.performance = { now: () => now };
globalThis.localStorage = {
  getItem: k => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, v),
  removeItem: k => store.delete(k)
};
globalThis.requestAnimationFrame = cb => { rafCb = cb; return 1; };
globalThis.addEventListener = () => {};

// ------------------------------------------------------------------ load game
eval(fs.readFileSync(path.join(__dirname, '..', 'docs', 'vision.js'), 'utf8'));
eval(fs.readFileSync(path.join(__dirname, '..', 'docs', 'game.js'), 'utf8'));

const D = globalThis.window.HoloLavaDebug;
ok(D && D.G, 'debug handle missing');
const G = D.G;

function pump(frames, gray) {
  for (let i = 0; i < frames; i++) {
    if (gray) setFrame(gray);
    now += 33;
    rafCb(now);
  }
}

function pumpUntil(pred, maxFrames, grayFn) {
  for (let i = 0; i < maxFrames; i++) {
    if (pred()) return true;
    if (grayFn) setFrame(grayFn(i));
    now += 33;
    rafCb(now);
  }
  return pred();
}

/* Frame that keeps the test player alive: stand on whatever tile is currently
 * safe. Without this, waiting around for a wave round burns through all three
 * hearts and the game ends before the wave ever arrives. */
function standSafeFrame() {
  const r = G.round;
  if (r && !r.isWave && r.safeLane >= 0) return frameWithBody(r.safeLane, 0);
  return frameWithBody(1, 0);
}

/* Play correctly until a wave round begins. Restarts the game if it ended. */
function seekWave(maxFrames) {
  if (G.phase === 'over') {
    pumpUntil(() => G.phase === 'ready', 120, () => ROOM);
    pumpUntil(() => G.phase === 'play', 120, () => frameWithBody(1, 0));
  }
  return pumpUntil(
    () => G.phase === 'play' && G.round && G.round.isWave && G.round.phase === 'warn',
    maxFrames, () => standSafeFrame());
}

/* Drive boot -> calibrate -> ready -> play, standing in `lane`. */
function bootToPlay(lane) {
  G.muted = true;
  elements['btn-start'].click();          // triggers the stubbed getUserMedia
  // Let the promise chain settle so beginCalibration() runs.
  return new Promise(resolve => setImmediate(() => {
    ok(G.phase === 'calibrate', 'expected calibrate phase, got ' + G.phase);
    // Countdown then 30 background frames, all of the empty room.
    pumpUntil(() => G.phase === 'ready', 400, () => ROOM);
    ok(G.phase === 'ready', 'calibration never completed (phase ' + G.phase + ')');
    // Step into frame: the game starts itself.
    pumpUntil(() => G.phase === 'play', 120, () => frameWithBody(lane, 0));
    ok(G.phase === 'play', 'game did not start (phase ' + G.phase + ')');
    resolve();
  }));
}

// ---------------------------------------------------------------------- tests
async function main() {
  console.log('HOLO LAVA headless game tests (node ' + process.version + ')\n');

  await bootToPlay(0);

  test('calibration and startup reach the play state', () => {
    ok(G.phase === 'play', 'phase is ' + G.phase);
    ok(D.detector.isReady(), 'detector not calibrated');
    ok(G.hearts === 3, 'hearts should start at 3, got ' + G.hearts);
    ok(G.score === 0, 'score should start at 0');
    ok(G.round, 'no round was created');
  });

  test('the silhouette puts the player in the expected lane', () => {
    pump(12, frameWithBody(2, 0));
    ok(G.player && G.player.present, 'player not detected');
    ok(G.player.lane === 2, 'expected lane 2, got ' + G.player.lane);
    pump(12, frameWithBody(0, 0));
    ok(G.player.lane === 0, 'expected lane 0, got ' + G.player.lane);
  });

  test('standing on the safe tile scores and keeps all hearts', () => {
    let scored = 0;
    const heartsBefore = G.hearts;
    for (let round = 0; round < 3; round++) {
      // Wait for a normal (non-wave) warning phase, then move to the safe tile.
      pumpUntil(() => G.round && G.round.phase === 'warn' && !G.round.isWave, 200,
                () => frameWithBody(G.player && G.player.lane !== null ? G.player.lane : 0, 0));
      const safe = G.round.safeLane;
      const scoreBefore = G.score;
      // Stand on it until the round resolves.
      pumpUntil(() => G.round.phase === 'resolve', 200, () => frameWithBody(safe, 0));
      pump(4, frameWithBody(safe, 0));
      if (G.score > scoreBefore) scored++;
    }
    ok(scored === 3, 'only ' + scored + ' of 3 safe rounds scored');
    ok(G.hearts === heartsBefore, 'lost a heart while standing on the safe tile');
  });

  test('combo multiplier grows with a streak', () => {
    ok(G.combo > 1, 'combo did not grow after a winning streak (combo ' + G.combo + ')');
  });

  test('standing in lava costs exactly one heart', () => {
    pumpUntil(() => G.round && G.round.phase === 'warn' && !G.round.isWave, 200,
              () => frameWithBody(0, 0));
    const wrong = (G.round.safeLane + 2) % LANES;   // deliberately not the safe tile
    const heartsBefore = G.hearts;
    pumpUntil(() => G.round.phase === 'resolve', 200, () => frameWithBody(wrong, 0));
    pump(3, frameWithBody(wrong, 0));
    ok(G.hearts === heartsBefore - 1,
       'expected to lose 1 heart, went from ' + heartsBefore + ' to ' + G.hearts);
    ok(G.combo === 1, 'combo should reset after a hit');
  });

  test('a jump survives the lava wave', () => {
    const found = seekWave(900);
    ok(found, 'no wave round appeared');
    const heartsBefore = G.hearts;
    // Hold still long enough for the tracker to learn the resting height, then
    // rise and stay up through the eruption.
    pumpUntil(() => G.round.t > G.round.duration * 0.45, 200, () => frameWithBody(1, 0));
    pumpUntil(() => G.player && G.player.airborne, 40, () => frameWithBody(1, 16));
    ok(G.player.airborne, 'jump was not detected');
    pumpUntil(() => G.round.phase === 'resolve', 200, () => frameWithBody(1, 16));
    pump(3, frameWithBody(1, 16));
    ok(G.hearts === heartsBefore, 'lost a heart despite jumping over the wave');
  });

  test('ignoring the lava wave costs a heart', () => {
    const found = seekWave(900);
    ok(found, 'no second wave round appeared');
    const heartsBefore = G.hearts;
    pumpUntil(() => G.round.phase === 'resolve', 300, () => frameWithBody(1, 0));
    pump(3, frameWithBody(1, 0));
    ok(G.hearts === heartsBefore - 1, 'standing through a wave should cost a heart');
  });

  test('three mistakes end the game and record a best score', () => {
    let guard = 0;
    while (G.hearts > 0 && guard++ < 40) {
      pumpUntil(() => G.round && G.round.phase === 'warn', 300, () => frameWithBody(0, 0));
      const wrong = G.round.isWave ? 0 : (G.round.safeLane + 2) % LANES;
      pumpUntil(() => G.round.phase === 'resolve' || G.phase === 'over', 300,
                () => frameWithBody(wrong, 0));
      pump(3, frameWithBody(wrong, 0));
    }
    ok(G.phase === 'over', 'game did not end (phase ' + G.phase + ', hearts ' + G.hearts + ')');
    ok(Number(store.get('holo-lava-best')) === G.best, 'best score was not persisted');
    ok(G.best > 0, 'best score should be positive after scoring earlier');
  });

  test('leaving the frame returns to the ready state for a rematch', () => {
    pumpUntil(() => G.phase === 'ready', 120, () => ROOM);
    ok(G.phase === 'ready', 'expected ready after stepping out, got ' + G.phase);
    pumpUntil(() => G.phase === 'play', 120, () => frameWithBody(3, 0));
    ok(G.phase === 'play', 'game did not restart on re-entry');
    ok(G.hearts === 3, 'hearts not restored on restart');
    ok(G.score === 0, 'score not reset on restart');
  });

  test('a lighting blowout pauses the game instead of killing the player', () => {
    const blown = new Uint8Array(PROC_W * PROC_H).fill(250);
    pumpUntil(() => G.phase === 'ready', 30, () => blown);
    ok(G.phase === 'ready',
       'a stale background should pause to ready, got ' + G.phase);
    ok(G.vision.backgroundStale, 'backgroundStale flag was not set');
  });

  test('recalibration works after the scene changes', () => {
    D.beginCalibration();
    ok(G.phase === 'calibrate', 'recalibration did not enter calibrate phase');
    pumpUntil(() => G.phase === 'ready', 400, () => ROOM);
    ok(G.phase === 'ready', 'recalibration never finished');
    pumpUntil(() => G.phase === 'play', 120, () => frameWithBody(1, 0));
    ok(G.phase === 'play', 'could not start a game after recalibration');
  });

  console.log('\ngame logic: ' + passed + ' passed, ' + failures.length + ' failed');
  if (failures.length) {
    for (const [name, msg, stack] of failures) {
      console.log('  - ' + name + '\n      ' + msg);
      if (!msg || /is not|undefined|null/.test(msg)) console.log(String(stack).split('\n').slice(0, 4).join('\n'));
    }
    process.exit(1);
  }
}

main().catch(err => { console.error(err); process.exit(1); });
