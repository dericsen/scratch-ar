/* Floor Challenge game logic.
 *
 * Deliberately knows nothing about cameras, canvases or MediaPipe. Its input
 * is a list of tracked players, each carrying an id and the grid tile they are
 * currently standing on; its output is tile states, scores and events. That
 * separation is what lets the whole game be tested headlessly by feeding it
 * synthetic players standing on chosen tiles.
 *
 * Twelve levels alternate between two round types:
 *
 *   SURVIVAL  some tiles are marked dangerous during the round and erupt when
 *             the timer ends. Be on a safe tile.
 *   QUESTION  a CS question with four answer tiles spread across the floor.
 *             Walk onto the right one before time runs out.
 *
 * Both resolve the same way: whatever tile a player occupies at the instant
 * the timer expires is their answer. Resolving on a timer rather than on
 * contact is what makes it fair to play with a camera -- a player crossing a
 * dangerous tile on the way to a safe one should not be punished for passing
 * through.
 */
(function (root) {
  'use strict';

  var ns = root.CVFloor = root.CVFloor || {};

  var TILE = {
    EMPTY: 'empty',
    SAFE: 'safe',
    DANGER: 'danger',
    CRACKED: 'cracked',
    BROKEN: 'broken',
    ANSWER: 'answer',
    CORRECT: 'correct',
    WRONG: 'wrong'
  };

  var PHASE = { IDLE: 'idle', ACTIVE: 'active', RESOLVE: 'resolve', GAMEOVER: 'gameover' };

  /* Small seeded PRNG so a level layout is reproducible: the tests need the
   * same danger pattern every run, and a player retrying a level should face
   * the same puzzle. */
  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function createGame(opts) {
    var o = {
      cols: 8,
      rows: 8,
      totalLevels: 12,
      lives: 3,
      seed: 1337,
      questionSeconds: 11,
      survivalSeconds: 6,
      minSeconds: 5,
      resolveSeconds: 2.6,
      strict: true,          // standing nowhere useful costs a life
      onEvent: null
    };
    if (opts) for (var k in opts) if (opts[k] !== undefined) o[k] = opts[k];

    var tiles = [];
    var stats = {};            // player id -> { score, lives, status, combo }
    var state = {
      phase: PHASE.IDLE,
      level: 1,
      levelType: null,
      question: null,
      answerTiles: [],
      dangerCount: 0,
      timeLeft: 0,
      roundDuration: 0,
      lastResults: [],
      totalLevels: o.totalLevels,
      cols: o.cols,
      rows: o.rows
    };

    function emit(type, data) {
      if (typeof o.onEvent === 'function') {
        o.onEvent(Object.assign({ type: type }, data || {}));
      }
    }

    function idx(row, col) { return row * o.cols + col; }

    function buildTiles() {
      tiles = [];
      for (var r = 0; r < o.rows; r++) {
        for (var c = 0; c < o.cols; c++) {
          tiles.push({
            row: r, col: c,
            state: TILE.EMPTY,
            symbol: null,
            optionIndex: -1,
            anim: 0,
            flash: 0
          });
        }
      }
    }

    function tileAt(row, col) {
      if (row < 0 || col < 0 || row >= o.rows || col >= o.cols) return null;
      return tiles[idx(row, col)];
    }

    function reset() {
      buildTiles();
      stats = {};
      state.phase = PHASE.IDLE;
      state.level = 1;
      state.levelType = null;
      state.question = null;
      state.answerTiles = [];
      state.timeLeft = 0;
      state.lastResults = [];
      emit('reset', {});
    }

    function statsFor(id) {
      if (!stats[id]) {
        stats[id] = { id: id, score: 0, lives: o.lives, status: 'ALIVE', combo: 1, joinedLevel: state.level };
        emit('playerJoined', { id: id });
      }
      return stats[id];
    }

    function levelTypeFor(level) {
      // Alternate, starting with a question so the CS angle is visible at once.
      return (level % 2 === 1) ? 'question' : 'survival';
    }

    function roundSeconds(level, type) {
      var base = type === 'question' ? o.questionSeconds : o.survivalSeconds;
      return Math.max(o.minSeconds, base - (level - 1) * 0.4);
    }

    /* One answer tile per quadrant, so the four options are always far apart.
     * Clustered options would be unreadable in perspective and would make the
     * tile latch arbitrate between two answers a step apart. */
    function pickAnswerTiles(rng) {
      var halfR = Math.floor(o.rows / 2), halfC = Math.floor(o.cols / 2);
      var quadrants = [
        [0, halfR - 1, 0, halfC - 1],
        [0, halfR - 1, halfC, o.cols - 1],
        [halfR, o.rows - 1, 0, halfC - 1],
        [halfR, o.rows - 1, halfC, o.cols - 1]
      ];
      var picks = [];
      for (var i = 0; i < 4; i++) {
        var q = quadrants[i];
        var r = q[0] + Math.floor(rng() * (q[1] - q[0] + 1));
        var c = q[2] + Math.floor(rng() * (q[3] - q[2] + 1));
        picks.push({ row: r, col: c });
      }
      return picks;
    }

    function startLevel(level) {
      state.level = level;
      state.levelType = levelTypeFor(level);
      state.phase = PHASE.ACTIVE;
      state.lastResults = [];
      buildTiles();

      var rng = mulberry32(o.seed + level * 7919);
      state.roundDuration = roundSeconds(level, state.levelType);
      state.timeLeft = state.roundDuration;

      if (state.levelType === 'question') {
        var q = ns.questions.get(Math.floor((level - 1) / 2));
        state.question = { q: q.q, options: q.options.slice(), answer: q.answer, note: q.note };
        // Shuffle which quadrant holds which option so the answer position is
        // not predictable from the layout.
        var order = [0, 1, 2, 3];
        for (var s = order.length - 1; s > 0; s--) {
          var j = Math.floor(rng() * (s + 1));
          var t = order[s]; order[s] = order[j]; order[j] = t;
        }
        var spots = pickAnswerTiles(rng);
        state.answerTiles = [];
        for (var i = 0; i < 4; i++) {
          var optionIndex = order[i];
          var spot = spots[i];
          var tile = tileAt(spot.row, spot.col);
          if (!tile) continue;
          tile.state = TILE.ANSWER;
          tile.symbol = q.options[optionIndex];
          tile.optionIndex = optionIndex;
          state.answerTiles.push({ row: spot.row, col: spot.col, optionIndex: optionIndex, text: q.options[optionIndex] });
        }
        state.dangerCount = 0;
      } else {
        state.question = null;
        state.answerTiles = [];
        var frac = Math.min(0.55, 0.22 + (level - 1) * 0.03);
        var danger = 0;
        for (var ti = 0; ti < tiles.length; ti++) {
          if (rng() < frac) { tiles[ti].state = TILE.DANGER; danger++; }
          else tiles[ti].state = TILE.SAFE;
        }
        // Guarantee at least one safe tile exists, otherwise the round is
        // unwinnable by construction.
        if (danger === tiles.length) {
          tiles[Math.floor(rng() * tiles.length)].state = TILE.SAFE;
          danger--;
        }
        state.dangerCount = danger;
      }
      emit('levelStart', { level: level, type: state.levelType, question: state.question });
    }

    function start() {
      reset();
      startLevel(1);
    }

    /* Evaluate every alive player against the tile they occupy right now. */
    function resolve(players) {
      var results = [];
      for (var i = 0; i < players.length; i++) {
        var p = players[i];
        var st = statsFor(p.id);
        if (st.status !== 'ALIVE') continue;

        var tile = p.tile ? tileAt(p.tile.row, p.tile.col) : null;
        var ok = false, reason = '';

        if (state.levelType === 'question') {
          if (tile && tile.state === TILE.ANSWER && tile.optionIndex === state.question.answer) {
            ok = true; reason = 'correct';
          } else if (tile && tile.state === TILE.ANSWER) {
            ok = false; reason = 'wrong-answer';
          } else {
            ok = false; reason = 'no-answer';
            if (!o.strict) { results.push({ id: p.id, ok: null, reason: reason, tile: p.tile }); continue; }
          }
        } else {
          if (tile && tile.state === TILE.SAFE) { ok = true; reason = 'safe'; }
          else if (tile && tile.state === TILE.DANGER) { ok = false; reason = 'lava'; }
          else {
            ok = false; reason = 'off-grid';
            if (!o.strict) { results.push({ id: p.id, ok: null, reason: reason, tile: p.tile }); continue; }
          }
        }

        if (ok) {
          st.combo = Math.min(5, st.combo + 1);
          var points = (state.levelType === 'question' ? 100 : 60) * st.combo;
          st.score += points;
          if (tile) { tile.state = TILE.CORRECT; tile.flash = 1; }
          results.push({ id: p.id, ok: true, reason: reason, points: points, tile: p.tile });
          emit('playerCorrect', { id: p.id, points: points, tile: p.tile });
        } else {
          st.combo = 1;
          st.lives--;
          if (tile) { tile.state = TILE.WRONG; tile.flash = 1; }
          if (st.lives <= 0) { st.lives = 0; st.status = 'OUT'; emit('playerOut', { id: p.id }); }
          results.push({ id: p.id, ok: false, reason: reason, tile: p.tile });
          emit('playerWrong', { id: p.id, reason: reason, tile: p.tile });
        }
      }

      // Every dangerous tile breaks, whether or not anyone was on it.
      if (state.levelType === 'survival') {
        for (var t = 0; t < tiles.length; t++) {
          if (tiles[t].state === TILE.DANGER) { tiles[t].state = TILE.BROKEN; tiles[t].anim = 1; }
        }
      } else {
        for (var a = 0; a < tiles.length; a++) {
          if (tiles[a].state === TILE.ANSWER) {
            tiles[a].state = (tiles[a].optionIndex === state.question.answer) ? TILE.CORRECT : TILE.CRACKED;
          }
        }
      }

      state.lastResults = results;
      state.phase = PHASE.RESOLVE;
      state.timeLeft = o.resolveSeconds;
      emit('roundResolved', { level: state.level, results: results });

      var known = Object.keys(stats);
      if (known.length > 0) {
        var anyAlive = false;
        for (var s = 0; s < known.length; s++) if (stats[known[s]].status === 'ALIVE') anyAlive = true;
        if (!anyAlive) {
          state.phase = PHASE.GAMEOVER;
          emit('gameOver', { level: state.level });
        }
      }
      return results;
    }

    function advance() {
      if (state.level >= o.totalLevels) {
        state.phase = PHASE.GAMEOVER;
        emit('completed', { level: state.level });
        return;
      }
      startLevel(state.level + 1);
      emit('levelUp', { level: state.level });
    }

    /* Drive the clock. `players` is the tracker's output. */
    function update(players, dt) {
      players = players || [];
      for (var i = 0; i < players.length; i++) statsFor(players[i].id);

      for (var t = 0; t < tiles.length; t++) {
        if (tiles[t].flash > 0) tiles[t].flash = Math.max(0, tiles[t].flash - dt * 1.6);
        if (tiles[t].anim > 0) tiles[t].anim = Math.max(0, tiles[t].anim - dt * 0.8);
      }

      if (state.phase === PHASE.ACTIVE) {
        state.timeLeft -= dt;
        if (state.timeLeft <= 0) { state.timeLeft = 0; resolve(players); }
      } else if (state.phase === PHASE.RESOLVE) {
        state.timeLeft -= dt;
        if (state.timeLeft <= 0) { state.timeLeft = 0; advance(); }
      }
      return state;
    }

    /* Players merged with their game stats -- the shape the HUD renders. */
    function playersView(players) {
      var out = [];
      players = players || [];
      for (var i = 0; i < players.length; i++) {
        var p = players[i];
        var st = statsFor(p.id);
        out.push({
          id: p.id,
          name: 'PLAYER ' + p.id,
          tile: p.tile,
          tileLabel: p.tile ? (ns.geometry.rowLetter(p.tile.row) + (p.tile.col + 1)) : '--',
          score: st.score,
          lives: st.lives,
          status: st.status,
          combo: st.combo,
          floor: p.floor,
          pixel: p.pixel,
          quality: p.quality,
          visible: p.visible
        });
      }
      return out;
    }

    function tileStateAt(row, col) {
      var t = tileAt(row, col);
      return t ? t.state : null;
    }

    buildTiles();

    return {
      TILE: TILE,
      PHASE: PHASE,
      get tiles() { return tiles; },
      get state() { return state; },
      get stats() { return stats; },
      start: start,
      reset: reset,
      startLevel: startLevel,
      update: update,
      resolve: resolve,
      advance: advance,
      tileAt: tileAt,
      tileStateAt: tileStateAt,
      statsFor: statsFor,
      playersView: playersView,
      options: o
    };
  }

  ns.game = { createGame: createGame, TILE: TILE, PHASE: PHASE, mulberry32: mulberry32 };
})(typeof globalThis !== 'undefined' ? globalThis : this);
