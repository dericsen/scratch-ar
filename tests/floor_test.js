/* Tests for the Computer Vision Floor core: geometry, tracking, game logic.
 *
 * The development sandbox has no camera, no browser and no network, so the
 * pipeline is exercised with synthetic pose landmarks placed at known floor
 * positions. That is enough to verify everything between "landmarks arrive"
 * and "tile state changes" -- which is where the project's real logic lives:
 *
 *   homography from four clicked corners, camera px <-> floor <-> grid,
 *   foot-landmark extraction, jitter filtering, tile hysteresis,
 *   multi-player identity assignment, scoring and round resolution.
 *
 * It cannot verify MediaPipe's own output, lighting, or how it feels to play.
 *
 * Run: node tests/floor_test.js
 */
'use strict';

const fs = require('fs');
const path = require('path');

const base = path.join(__dirname, '..', 'docs', 'floor', 'core');
for (const f of ['geometry.js', 'tracking.js', 'questions.js', 'game.js']) {
  eval(fs.readFileSync(path.join(base, f), 'utf8'));
}
const CV = globalThis.CVFloor;
const { geometry: GEO, tracking: TRK, questions: Q, game: GAME } = CV;

let failures = [];
let passed = 0;
class AssertErr extends Error {}
function ok(c, m) { if (!c) throw new AssertErr(m || 'assertion failed'); }
function near(a, b, tol, m) {
  if (!(Math.abs(a - b) <= tol)) throw new AssertErr((m || 'near') + ': ' + a + ' vs ' + b);
}
function section(name) { console.log('\n' + name); }
function test(name, fn) {
  try {
    fn(); passed++;
    console.log('  ' + name.padEnd(62) + ' ok');
  } catch (e) {
    failures.push([name, e.message, e.stack]);
    console.log('  ' + name.padEnd(62) + (e instanceof AssertErr ? ' FAIL' : ' ERROR'));
  }
}

// A realistic angled view of a floor in a 1280x720 frame: the far edge is
// narrower than the near edge, which is exactly the perspective the
// homography has to account for.
const FRAME_W = 1280, FRAME_H = 720;
const QUAD = [[430, 300], [870, 300], [1180, 670], [110, 670]];
const CALIB = GEO.createCalibration(QUAD, 8, 8);

function polyArea(pts) {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i], q = pts[(i + 1) % pts.length];
    a += p[0] * q[1] - q[0] * p[1];
  }
  return Math.abs(a / 2);
}

// --------------------------------------------------------------- geometry
section('geometry / floor calibration');

test('calibration is created from a sane quad', () => {
  ok(CALIB !== null, 'createCalibration returned null for a valid quad');
  ok(CALIB.cols === 8 && CALIB.rows === 8, 'grid size wrong');
});

test('the four clicked corners map exactly to the unit square', () => {
  const expect = [[0, 0], [1, 0], [1, 1], [0, 1]];
  for (let i = 0; i < 4; i++) {
    const f = CALIB.toFloor(QUAD[i][0], QUAD[i][1]);
    near(f[0], expect[i][0], 1e-9, 'corner ' + i + ' x');
    near(f[1], expect[i][1], 1e-9, 'corner ' + i + ' y');
  }
});

test('camera -> floor -> camera round trips to sub-pixel accuracy', () => {
  let worst = 0;
  for (let fy = 0.02; fy < 1; fy += 0.07) {
    for (let fx = 0.02; fx < 1; fx += 0.07) {
      const px = CALIB.toCanvas(fx, fy);
      const back = CALIB.toFloor(px[0], px[1]);
      worst = Math.max(worst, Math.hypot(back[0] - fx, back[1] - fy));
    }
  }
  ok(worst < 1e-9, 'round-trip error ' + worst);
});

test('homography handles raw pixel magnitudes without conditioning blowup', () => {
  // Same floor expressed in a 4K frame: coordinates in the thousands are
  // where an unnormalised DLT starts bending the grid off the floor.
  const big = GEO.createCalibration([[1720, 1200], [3480, 1200], [4720, 2680], [440, 2680]], 8, 8);
  ok(big !== null, 'calibration failed at 4K scale');
  const px = big.toCanvas(0.37, 0.81);
  const back = big.toFloor(px[0], px[1]);
  near(back[0], 0.37, 1e-8, 'x');
  near(back[1], 0.81, 1e-8, 'y');
});

test('perspective is real: far tiles are smaller on screen than near tiles', () => {
  const far = polyArea(CALIB.tileCanvasQuad(0, 4));
  const near_ = polyArea(CALIB.tileCanvasQuad(7, 4));
  ok(near_ > far * 1.5,
     'near tile area ' + near_.toFixed(0) + ' should clearly exceed far ' + far.toFixed(0));
});

test('floor coordinates map to the expected tiles', () => {
  ok(CALIB.tileAtFloor(0.01, 0.01).row === 0, 'top-left row');
  ok(CALIB.tileAtFloor(0.01, 0.01).col === 0, 'top-left col');
  ok(CALIB.tileAtFloor(0.99, 0.99).row === 7, 'bottom-right row');
  ok(CALIB.tileAtFloor(0.99, 0.99).col === 7, 'bottom-right col');
  // Tile C4 is row index 2, col index 3 -> floor centre (3.5/8, 2.5/8)
  const t = CALIB.tileAtFloor(3.5 / 8, 2.5 / 8);
  ok(t.row === 2 && t.col === 3, 'expected row 2 col 3, got ' + JSON.stringify(t));
  ok(CALIB.label(t.row, t.col) === 'C4', 'label should be C4, got ' + CALIB.label(t.row, t.col));
});

test('a point outside the calibrated floor has no tile', () => {
  ok(CALIB.tileAtFloor(-0.01, 0.5) === null, 'left of floor');
  ok(CALIB.tileAtFloor(0.5, 1.01) === null, 'past the near edge');
  ok(CALIB.tileAtPixel(5, 5) === null, 'a pixel far outside the quad');
});

test('row letters keep working past Z', () => {
  ok(GEO.rowLetter(0) === 'A', 'A');
  ok(GEO.rowLetter(25) === 'Z', 'Z');
  ok(GEO.rowLetter(26) === 'AA', 'AA, got ' + GEO.rowLetter(26));
});

test('degenerate calibration quads are rejected, not rendered', () => {
  ok(GEO.createCalibration([[0, 0], [10, 0], [10, 10], [0, 10]], 8, 8) === null,
     'a 10x10 px quad is too small to be a floor');
  ok(GEO.createCalibration([[100, 100], [500, 100], [100, 500], [500, 500]], 8, 8) === null,
     'a bow-tie (crossed) quad must be rejected');
  ok(GEO.createCalibration([[0, 0], [100, 100], [200, 200], [300, 300]], 8, 8) === null,
     'collinear points must be rejected');
  ok(GEO.createCalibration(null, 8, 8) === null, 'null quad');
});

test('insetInTile is 0.5 at a tile centre and ~0 on an edge', () => {
  near(CALIB.insetInTile(3.5 / 8, 2.5 / 8), 0.5, 1e-9, 'centre');
  ok(CALIB.insetInTile(3.001 / 8, 2.5 / 8) < 0.01, 'near a vertical edge');
});

test('a floor circle projects to a perspective-correct polygon', () => {
  const nearPoly = CALIB.floorCirclePolygon(0.5, 0.9, 0.05, 24);
  const farPoly = CALIB.floorCirclePolygon(0.5, 0.1, 0.05, 24);
  ok(nearPoly && farPoly, 'polygon generation failed');
  ok(polyArea(nearPoly) > polyArea(farPoly) * 1.5,
     'the same floor circle must look bigger near the camera');
});

// --------------------------------------------------------------- tracking
section('tracking / pose -> player');

function blankLandmarks() {
  const a = [];
  for (let i = 0; i < 33; i++) a.push({ x: 0.5, y: 0.5, z: 0, visibility: 0 });
  return a;
}

/* Build a pose standing at floor (fx, fy): feet on that point, body above it. */
function poseAtFloor(fx, fy, jitterPx, rand) {
  const px = CALIB.toCanvas(fx, fy);
  const j = jitterPx || 0;
  const r = rand || (() => 0.5);
  const cx = px[0] + (r() - 0.5) * 2 * j;
  const cy = px[1] + (r() - 0.5) * 2 * j;
  const nx = cx / FRAME_W, ny = cy / FRAME_H;
  const lm = blankLandmarks();
  const set = (i, x, y) => { lm[i] = { x: x, y: y, z: 0, visibility: 0.99 }; };
  const L = TRK.LM;
  const dx = 18 / FRAME_W;                 // feet straddle the stance point
  set(L.LEFT_HEEL, nx - dx, ny);
  set(L.LEFT_FOOT_INDEX, nx - dx, ny - 6 / FRAME_H);
  set(L.RIGHT_HEEL, nx + dx, ny);
  set(L.RIGHT_FOOT_INDEX, nx + dx, ny - 6 / FRAME_H);
  set(L.LEFT_ANKLE, nx - dx, ny - 14 / FRAME_H);
  set(L.RIGHT_ANKLE, nx + dx, ny - 14 / FRAME_H);
  set(L.LEFT_KNEE, nx - dx, ny - 90 / FRAME_H);
  set(L.RIGHT_KNEE, nx + dx, ny - 90 / FRAME_H);
  set(L.LEFT_HIP, nx - dx, ny - 170 / FRAME_H);
  set(L.RIGHT_HIP, nx + dx, ny - 170 / FRAME_H);
  set(L.LEFT_SHOULDER, nx - dx * 1.6, ny - 260 / FRAME_H);
  set(L.RIGHT_SHOULDER, nx + dx * 1.6, ny - 260 / FRAME_H);
  set(L.NOSE, nx, ny - 300 / FRAME_H);
  return lm;
}

test('the stance point comes from the feet, not the body centre', () => {
  const lm = poseAtFloor(0.5, 0.8);
  const obs = TRK.extractObservation(lm, FRAME_W, FRAME_H);
  ok(obs && obs.stancePixel, 'no observation');
  const feetY = ((lm[TRK.LM.LEFT_HEEL].y + lm[TRK.LM.LEFT_FOOT_INDEX].y) / 2) * FRAME_H;
  const hipY = lm[TRK.LM.LEFT_HIP].y * FRAME_H;
  near(obs.stancePixel.y, feetY, 6, 'stance should sit at the feet');
  ok(Math.abs(obs.stancePixel.y - hipY) > 100,
     'stance must not be anywhere near the hips/bbox centre');
  ok(obs.quality >= 0.9, 'two clean feet should be high quality');
});

test('one visible foot still gives a usable, lower-confidence stance', () => {
  const lm = poseAtFloor(0.5, 0.8);
  for (const i of [TRK.LM.RIGHT_HEEL, TRK.LM.RIGHT_FOOT_INDEX, TRK.LM.RIGHT_ANKLE]) {
    lm[i].visibility = 0.1;
  }
  const obs = TRK.extractObservation(lm, FRAME_W, FRAME_H);
  ok(obs && obs.stancePixel, 'should still produce a stance');
  ok(obs.quality > 0.5 && obs.quality < 0.95, 'quality should drop, got ' + obs.quality);
  near(obs.stancePixel.x, lm[TRK.LM.LEFT_HEEL].x * FRAME_W, 2, 'should follow the visible foot');
});

test('with no feet at all it falls back and flags low quality', () => {
  const lm = poseAtFloor(0.5, 0.8);
  for (const i of [TRK.LM.LEFT_HEEL, TRK.LM.LEFT_FOOT_INDEX, TRK.LM.LEFT_ANKLE,
                   TRK.LM.RIGHT_HEEL, TRK.LM.RIGHT_FOOT_INDEX, TRK.LM.RIGHT_ANKLE]) {
    lm[i].visibility = 0.0;
  }
  const obs = TRK.extractObservation(lm, FRAME_W, FRAME_H);
  ok(obs && obs.stancePixel, 'fallback should still return something');
  ok(obs.quality <= 0.35, 'fallback must be marked low quality, got ' + obs.quality);
});

test('a pose with too few landmarks is rejected', () => {
  ok(TRK.extractObservation([{ x: 0.5, y: 0.5 }], FRAME_W, FRAME_H) === null, 'short array');
  ok(TRK.extractObservation(null, FRAME_W, FRAME_H) === null, 'null');
});

test('the filter removes jitter while standing still', () => {
  let seed = 7;
  const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  const f = TRK.createPointFilter({ minCutoff: 0.8, beta: 0.45 });
  const rawXs = [], outXs = [];
  for (let i = 0; i < 120; i++) {
    const rx = 0.5 + (rand() - 0.5) * 0.02;    // +/- 1% of the floor
    const ry = 0.5 + (rand() - 0.5) * 0.02;
    const o = f.update(rx, ry, 1 / 30);
    if (i > 40) { rawXs.push(rx); outXs.push(o.x); }
  }
  const sd = a => {
    const m = a.reduce((s, v) => s + v, 0) / a.length;
    return Math.sqrt(a.reduce((s, v) => s + (v - m) * (v - m), 0) / a.length);
  };
  ok(sd(outXs) < sd(rawXs) * 0.4,
     'jitter not suppressed: raw sd ' + sd(rawXs).toFixed(5) + ' -> ' + sd(outXs).toFixed(5));
});

test('the filter still tracks a decisive move quickly', () => {
  const f = TRK.createPointFilter({ minCutoff: 0.8, beta: 0.45 });
  for (let i = 0; i < 30; i++) f.update(0.2, 0.5, 1 / 30);
  let frames = 0, v = null;
  for (let i = 0; i < 60; i++) {
    v = f.update(0.8, 0.5, 1 / 30);
    frames++;
    if (Math.abs(v.x - 0.8) < 0.02) break;
  }
  ok(Math.abs(v.x - 0.8) < 0.02, 'never converged, ended at ' + v.x);
  ok(frames <= 20, 'took ' + frames + ' frames to catch up (too laggy)');
});

test('the tile latch does not flicker on a boundary', () => {
  const latch = TRK.createTileLatch({ minFrames: 3, commitInset: 0.18 });
  const boundary = 4 / 8;                     // exact edge between col 3 and 4
  let changes = 0, last = null;
  for (let i = 0; i < 60; i++) {
    const fx = boundary + (i % 2 ? 0.0008 : -0.0008);   // jitter across the edge
    const tile = CALIB.tileAtFloor(fx, 0.5);
    const inset = CALIB.insetInTile(fx, 0.5);
    const cur = latch.update(tile, inset);
    const key = cur ? cur.row + ',' + cur.col : 'null';
    if (last !== null && key !== last) changes++;
    last = key;
  }
  ok(changes === 0, 'tile flipped ' + changes + ' times while standing on a seam');
});

test('the tile latch commits promptly on a real step', () => {
  const latch = TRK.createTileLatch({ minFrames: 3, commitInset: 0.18 });
  latch.update(CALIB.tileAtFloor(0.1, 0.5), CALIB.insetInTile(0.1, 0.5));
  const before = latch.tile;
  let frames = 0;
  for (let i = 0; i < 10; i++) {
    const t = latch.update(CALIB.tileAtFloor(0.3, 0.5), CALIB.insetInTile(0.3, 0.5));
    frames++;
    if (t.col !== before.col) break;
  }
  ok(latch.tile.col !== before.col, 'never committed to the new tile');
  ok(frames <= 3, 'took ' + frames + ' frames to accept a clear step');
});

test('assignment is global, so close players do not steal identities', () => {
  // Track A is slightly nearer to detection 0, but detection 0 is a far better
  // match for track B. A per-track greedy loop gets this wrong.
  const tracks = [{ floor: { x: 0.50, y: 0.5 } }, { floor: { x: 0.62, y: 0.5 } }];
  const obs = [{ floor: { x: 0.60, y: 0.5 } }, { floor: { x: 0.40, y: 0.5 } }];
  const res = TRK.assign(tracks, obs, 0.3, (t, o) =>
    Math.hypot(t.floor.x - o.floor.x, t.floor.y - o.floor.y));
  const map = {};
  res.matches.forEach(m => { map[m.t] = m.d; });
  ok(map[1] === 0, 'track 1 should take detection 0 (distance 0.02)');
  ok(map[0] === 1, 'track 0 should take detection 1, got ' + map[0]);
});

test('two players keep stable, distinct IDs across frames', () => {
  const tr = TRK.createPlayerTracker();
  let t = 0, ids = null;
  for (let i = 0; i < 30; i++, t += 33) {
    const players = tr.update(
      [poseAtFloor(0.25, 0.7), poseAtFloor(0.75, 0.4)],
      CALIB, t, 1 / 30, FRAME_W, FRAME_H);
    if (i > 5) {
      ok(players.length === 2, 'expected 2 players, got ' + players.length);
      const now = players.map(p => p.id).join(',');
      if (ids) ok(now === ids, 'IDs changed from ' + ids + ' to ' + now);
      ids = now;
    }
  }
  ok(ids === '1,2', 'expected ids 1,2 got ' + ids);
});

test('players are placed on the tiles they are standing on', () => {
  const tr = TRK.createPlayerTracker();
  let t = 0;
  let players = [];
  for (let i = 0; i < 20; i++, t += 33) {
    players = tr.update([poseAtFloor(3.5 / 8, 2.5 / 8)], CALIB, t, 1 / 30, FRAME_W, FRAME_H);
  }
  ok(players.length === 1, 'expected one player');
  ok(players[0].tile, 'player has no tile');
  ok(players[0].tile.row === 2 && players[0].tile.col === 3,
     'expected C4 (row 2 col 3), got ' + JSON.stringify(players[0].tile));
});

test('a briefly occluded player keeps their ID', () => {
  const tr = TRK.createPlayerTracker({ lostTimeoutMs: 1500 });
  let t = 0;
  for (let i = 0; i < 20; i++, t += 33) {
    tr.update([poseAtFloor(0.4, 0.6)], CALIB, t, 1 / 30, FRAME_W, FRAME_H);
  }
  const idBefore = tr.players[0].id;
  for (let i = 0; i < 20; i++, t += 33) {       // ~0.66 s with nobody detected
    tr.update([], CALIB, t, 1 / 30, FRAME_W, FRAME_H);
  }
  let players = [];
  for (let i = 0; i < 10; i++, t += 33) {
    players = tr.update([poseAtFloor(0.42, 0.62)], CALIB, t, 1 / 30, FRAME_W, FRAME_H);
  }
  ok(players.length === 1, 'player not re-acquired');
  ok(players[0].id === idBefore,
     'ID changed across a short occlusion: ' + idBefore + ' -> ' + players[0].id);
});

test('a player gone past the timeout becomes a new player', () => {
  const tr = TRK.createPlayerTracker({ lostTimeoutMs: 400 });
  let t = 0;
  for (let i = 0; i < 20; i++, t += 33) {
    tr.update([poseAtFloor(0.4, 0.6)], CALIB, t, 1 / 30, FRAME_W, FRAME_H);
  }
  const idBefore = tr.players[0].id;
  for (let i = 0; i < 40; i++, t += 33) {
    tr.update([], CALIB, t, 1 / 30, FRAME_W, FRAME_H);
  }
  let players = [];
  for (let i = 0; i < 10; i++, t += 33) {
    players = tr.update([poseAtFloor(0.4, 0.6)], CALIB, t, 1 / 30, FRAME_W, FRAME_H);
  }
  ok(players.length === 1 && players[0].id !== idBefore,
     'a long-gone player should not inherit the old ID');
});

test('jitter in pixel space does not change the reported tile', () => {
  let seed = 99;
  const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  const tr = TRK.createPlayerTracker();
  let t = 0, changes = 0, last = null;
  for (let i = 0; i < 90; i++, t += 33) {
    // Stand mid-tile but with 4 px of landmark noise every frame.
    const players = tr.update([poseAtFloor(2.5 / 8, 5.5 / 8, 4, rand)],
                              CALIB, t, 1 / 30, FRAME_W, FRAME_H);
    if (i > 10 && players.length) {
      const key = players[0].tile ? players[0].tile.row + ',' + players[0].tile.col : 'null';
      if (last !== null && key !== last) changes++;
      last = key;
    }
  }
  ok(changes === 0, 'tile changed ' + changes + ' times from pure jitter');
  ok(last === '5,2', 'expected tile row 5 col 2, got ' + last);
});

// --------------------------------------------------------------- game logic
section('game logic / rounds and scoring');

test('the question bank is well formed', () => {
  ok(Q.validate(), 'a question has the wrong option count, bad answer index, or duplicates');
  ok(Q.count >= 12, 'need at least 12 questions for 12 levels, have ' + Q.count);
});

function fakePlayer(id, row, col) {
  return { id: id, tile: (row === null ? null : { row: row, col: col }), visible: true, quality: 1 };
}

function findAnswerTile(g, optionIndex) {
  return g.state.answerTiles.find(t => t.optionIndex === optionIndex);
}

test('level 1 is a question round with four spread answer tiles', () => {
  const g = GAME.createGame({ seed: 42 });
  g.start();
  ok(g.state.phase === 'active', 'phase ' + g.state.phase);
  ok(g.state.levelType === 'question', 'level 1 should be a question');
  ok(g.state.question && g.state.question.options.length === 4, 'no question loaded');
  ok(g.state.answerTiles.length === 4, 'expected 4 answer tiles');
  const seen = new Set(g.state.answerTiles.map(t => t.optionIndex));
  ok(seen.size === 4, 'answer tiles must cover all four options');
  // One per quadrant => no two answers share a quadrant.
  const quad = t => (t.row < 4 ? 0 : 2) + (t.col < 4 ? 0 : 1);
  ok(new Set(g.state.answerTiles.map(quad)).size === 4, 'answers are not spread across quadrants');
});

test('standing on the correct answer scores and costs no life', () => {
  const g = GAME.createGame({ seed: 42 });
  g.start();
  const correct = findAnswerTile(g, g.state.question.answer);
  const players = [fakePlayer(1, correct.row, correct.col)];
  g.update(players, 99);                       // run the clock past the round
  const st = g.statsFor(1);
  ok(st.score > 0, 'no points awarded');
  ok(st.lives === 3, 'lost a life while answering correctly');
  ok(g.state.lastResults[0].ok === true, 'result not marked correct');
  ok(g.tileStateAt(correct.row, correct.col) === 'correct', 'correct tile not highlighted');
});

test('standing on a wrong answer costs exactly one life', () => {
  const g = GAME.createGame({ seed: 42 });
  g.start();
  const wrongIdx = (g.state.question.answer + 1) % 4;
  const wrong = findAnswerTile(g, wrongIdx);
  g.update([fakePlayer(1, wrong.row, wrong.col)], 99);
  const st = g.statsFor(1);
  ok(st.lives === 2, 'expected 2 lives, got ' + st.lives);
  ok(st.score === 0, 'should not score for a wrong answer');
  ok(g.state.lastResults[0].reason === 'wrong-answer', 'reason ' + g.state.lastResults[0].reason);
});

test('standing on no answer tile at all is also a miss', () => {
  const g = GAME.createGame({ seed: 42 });
  g.start();
  const taken = new Set(g.state.answerTiles.map(t => t.row + ',' + t.col));
  let spot = null;
  for (let r = 0; r < 8 && !spot; r++) {
    for (let c = 0; c < 8; c++) if (!taken.has(r + ',' + c)) { spot = { r, c }; break; }
  }
  g.update([fakePlayer(1, spot.r, spot.c)], 99);
  ok(g.statsFor(1).lives === 2, 'a non-answer should cost a life in strict mode');
  ok(g.state.lastResults[0].reason === 'no-answer', g.state.lastResults[0].reason);
});

test('lenient mode does not punish a player who did not answer', () => {
  const g = GAME.createGame({ seed: 42, strict: false });
  g.start();
  g.update([fakePlayer(1, null)], 99);
  ok(g.statsFor(1).lives === 3, 'lenient mode should not take a life');
});

test('survival round: safe tile scores, lava costs a life, danger breaks', () => {
  const g = GAME.createGame({ seed: 7 });
  g.start();
  g.startLevel(2);
  ok(g.state.levelType === 'survival', 'level 2 should be survival');
  ok(g.state.dangerCount > 0 && g.state.dangerCount < 64, 'danger count ' + g.state.dangerCount);
  let safe = null, danger = null;
  for (const t of g.tiles) {
    if (!safe && t.state === 'safe') safe = t;
    if (!danger && t.state === 'danger') danger = t;
  }
  ok(safe && danger, 'need both a safe and a dangerous tile');
  g.update([fakePlayer(1, safe.row, safe.col), fakePlayer(2, danger.row, danger.col)], 99);
  ok(g.statsFor(1).score > 0, 'safe player did not score');
  ok(g.statsFor(1).lives === 3, 'safe player lost a life');
  ok(g.statsFor(2).lives === 2, 'lava player should lose exactly one life');
  ok(g.tileStateAt(danger.row, danger.col) === 'wrong' ||
     g.tileStateAt(danger.row, danger.col) === 'broken',
     'the dangerous tile should break');
});

test('a level layout is reproducible from its seed', () => {
  const a = GAME.createGame({ seed: 2024 }); a.start(); a.startLevel(4);
  const b = GAME.createGame({ seed: 2024 }); b.start(); b.startLevel(4);
  const sig = g => g.tiles.map(t => t.state).join('');
  ok(sig(a) === sig(b), 'same seed produced different layouts');
  const c = GAME.createGame({ seed: 999 }); c.start(); c.startLevel(4);
  ok(sig(a) !== sig(c), 'different seeds produced identical layouts');
});

test('combo grows on a streak, caps, and resets on a miss', () => {
  const g = GAME.createGame({ seed: 42 });
  g.start();
  for (let i = 0; i < 8; i++) {
    const lvl = g.state.level;
    if (g.state.levelType === 'question') {
      const t = findAnswerTile(g, g.state.question.answer);
      g.update([fakePlayer(1, t.row, t.col)], 99);
    } else {
      const safe = g.tiles.find(t => t.state === 'safe');
      g.update([fakePlayer(1, safe.row, safe.col)], 99);
    }
    g.update([fakePlayer(1, 0, 0)], 99);       // let the resolve phase elapse
    ok(g.state.level >= lvl, 'level went backwards');
  }
  ok(g.statsFor(1).combo === 5, 'combo should cap at 5, got ' + g.statsFor(1).combo);
  ok(g.statsFor(1).score > 500, 'score too low after a long streak');
});

test('the round advances to the next level after the resolve phase', () => {
  const g = GAME.createGame({ seed: 42 });
  g.start();
  ok(g.state.level === 1, 'should start at level 1');
  const t = findAnswerTile(g, g.state.question.answer);
  g.update([fakePlayer(1, t.row, t.col)], 99);
  ok(g.state.phase === 'resolve', 'phase ' + g.state.phase);
  g.update([fakePlayer(1, t.row, t.col)], 99);
  ok(g.state.level === 2, 'did not advance, level ' + g.state.level);
  ok(g.state.phase === 'active', 'next level should be active');
});

test('three misses puts a player out and ends a solo game', () => {
  const g = GAME.createGame({ seed: 42 });
  g.start();
  for (let i = 0; i < 3; i++) {
    g.update([fakePlayer(1, null)], 99);       // miss
    if (g.state.phase === 'gameover') break;
    g.update([fakePlayer(1, null)], 99);       // elapse resolve
  }
  ok(g.statsFor(1).lives === 0, 'lives ' + g.statsFor(1).lives);
  ok(g.statsFor(1).status === 'OUT', 'status ' + g.statsFor(1).status);
  ok(g.state.phase === 'gameover', 'solo game should be over, phase ' + g.state.phase);
});

test('one player being out does not end a game with someone still alive', () => {
  const g = GAME.createGame({ seed: 42 });
  g.start();
  for (let i = 0; i < 3; i++) {
    const t = findAnswerTile(g, g.state.question ? g.state.question.answer : 0);
    const good = t ? fakePlayer(2, t.row, t.col) : fakePlayer(2, 0, 0);
    const safe = g.state.levelType === 'survival'
      ? g.tiles.find(x => x.state === 'safe') : null;
    g.update([fakePlayer(1, null), safe ? fakePlayer(2, safe.row, safe.col) : good], 99);
    g.update([fakePlayer(1, null)], 99);
  }
  ok(g.statsFor(1).status === 'OUT', 'player 1 should be out');
  ok(g.statsFor(2).status === 'ALIVE', 'player 2 should still be alive');
  ok(g.state.phase !== 'gameover', 'game should continue while someone is alive');
});

test('a player joining mid-game starts with a full set of lives', () => {
  const g = GAME.createGame({ seed: 42 });
  g.start();
  g.update([fakePlayer(1, 0, 0)], 99);
  const late = g.statsFor(9);
  ok(late.lives === 3 && late.score === 0, 'late joiner should start fresh');
  ok(late.joinedLevel >= 1, 'joinedLevel not recorded');
});

test('playersView exposes the HUD fields with tile labels', () => {
  const g = GAME.createGame({ seed: 42 });
  g.start();
  const view = g.playersView([fakePlayer(1, 2, 3), fakePlayer(2, null)]);
  ok(view[0].name === 'PLAYER 1', 'name ' + view[0].name);
  ok(view[0].tileLabel === 'C4', 'tile label should be C4, got ' + view[0].tileLabel);
  ok(view[0].status === 'ALIVE', 'status');
  ok(view[1].tileLabel === '--', 'a player off the floor should show --');
});

test('finishing the last level completes the game', () => {
  const g = GAME.createGame({ seed: 42, totalLevels: 3 });
  g.start();
  let guard = 0;
  while (g.state.phase !== 'gameover' && guard++ < 20) {
    let spot = { r: 0, c: 0 };
    if (g.state.levelType === 'question') {
      const t = findAnswerTile(g, g.state.question.answer);
      spot = { r: t.row, c: t.col };
    } else {
      const safe = g.tiles.find(t => t.state === 'safe');
      spot = { r: safe.row, c: safe.col };
    }
    g.update([fakePlayer(1, spot.r, spot.c)], 99);
    g.update([fakePlayer(1, spot.r, spot.c)], 99);
  }
  ok(g.state.phase === 'gameover', 'phase ' + g.state.phase);
  ok(g.statsFor(1).lives === 3, 'should have survived unscathed');
  ok(g.state.level === 3, 'should have reached level 3, got ' + g.state.level);
});

// --------------------------------------------------- end-to-end integration
section('integration / camera pixels to score');

test('a synthetic player walks onto the correct answer tile and scores', () => {
  // The full chain with no shortcuts: landmarks -> homography -> grid ->
  // tracker -> game, including the filter and the tile latch.
  const tr = TRK.createPlayerTracker();
  const g = GAME.createGame({ seed: 2025 });
  g.start();
  const target = findAnswerTile(g, g.state.question.answer);
  const dest = CALIB.tileCenterFloor(target.row, target.col);

  let t = 0, players = [];
  const start = [0.5, 0.95];
  for (let i = 0; i < 45; i++, t += 33) {      // walk there over ~1.5 s
    const k = Math.min(1, i / 30);
    const fx = start[0] + (dest[0] - start[0]) * k;
    const fy = start[1] + (dest[1] - start[1]) * k;
    players = tr.update([poseAtFloor(fx, fy, 2, Math.random)], CALIB, t, 1 / 30, FRAME_W, FRAME_H);
    g.update(g.playersView(players), 1 / 30);
  }
  ok(players.length === 1, 'lost the player while walking');
  ok(players[0].tile && players[0].tile.row === target.row && players[0].tile.col === target.col,
     'player should be standing on ' + CALIB.label(target.row, target.col) +
     ', detected ' + (players[0].tile ? CALIB.label(players[0].tile.row, players[0].tile.col) : 'none'));

  g.update(g.playersView(players), 99);        // expire the round timer
  ok(g.statsFor(players[0].id).score > 0, 'walking onto the right tile did not score');
  ok(g.statsFor(players[0].id).lives === 3, 'should not have lost a life');
});

test('two players are scored independently on their own tiles', () => {
  const tr = TRK.createPlayerTracker();
  const g = GAME.createGame({ seed: 2025 });
  g.start();
  const correct = findAnswerTile(g, g.state.question.answer);
  const wrong = findAnswerTile(g, (g.state.question.answer + 2) % 4);
  const a = CALIB.tileCenterFloor(correct.row, correct.col);
  const b = CALIB.tileCenterFloor(wrong.row, wrong.col);

  let t = 0, players = [];
  for (let i = 0; i < 25; i++, t += 33) {
    players = tr.update([poseAtFloor(a[0], a[1]), poseAtFloor(b[0], b[1])],
                        CALIB, t, 1 / 30, FRAME_W, FRAME_H);
    g.update(g.playersView(players), 1 / 30);
  }
  ok(players.length === 2, 'expected 2 players, got ' + players.length);
  g.update(g.playersView(players), 99);
  const byTile = {};
  g.playersView(players).forEach(p => { byTile[p.id] = p.tileLabel; });
  const correctLabel = CALIB.label(correct.row, correct.col);
  const winner = Object.keys(byTile).find(id => byTile[id] === correctLabel);
  const loser = Object.keys(byTile).find(id => byTile[id] !== correctLabel);
  ok(winner && loser, 'could not tell the two players apart');
  ok(g.statsFor(Number(winner)).score > 0, 'the player on the right tile did not score');
  ok(g.statsFor(Number(loser)).lives === 2, 'the player on the wrong tile kept all lives');
});

// -------------------------------------------------------------------- report
console.log('\nfloor core: ' + passed + ' passed, ' + failures.length + ' failed');
if (failures.length) {
  for (const [n, m, s] of failures) {
    console.log('  - ' + n + '\n      ' + m);
    if (!(m || '').length || /is not a function|undefined|Cannot read/.test(m)) {
      console.log(String(s).split('\n').slice(0, 5).join('\n'));
    }
  }
  process.exit(1);
}
