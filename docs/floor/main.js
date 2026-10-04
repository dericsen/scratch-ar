/* Application wiring: camera -> pose -> calibration -> tracking -> game -> AR.
 *
 * App states: boot -> camera -> calibrate -> ready -> playing (-> error)
 *
 * Calibration is persisted per camera resolution, so a returning player does
 * not have to re-click the floor corners every reload.
 */
(function () {
  'use strict';

  var CV = window.CVFloor;
  var $ = function (id) { return document.getElementById(id); };

  var video = $('cam');
  var canvas = $('stage');
  var renderer = CV.render.createRenderer(canvas);

  var el = {
    status: $('status'),
    level: $('hud-level'),
    phase: $('hud-phase'),
    players: $('hud-players'),
    question: $('question-text'),
    questionSub: $('question-sub'),
    questionBar: $('question-bar'),
    btnStart: $('btn-start'),
    btnRecal: $('btn-recal'),
    btnRestart: $('btn-restart'),
    btnDefault: $('btn-default'),
    btnMock: $('btn-mock'),
    btnSkeleton: $('btn-skeleton'),
    btnLabels: $('btn-labels'),
    grid: $('grid-size'),
    hint: $('calib-hint')
  };

  var S = {
    phase: 'boot',
    error: '',
    corners: [],
    activeCorner: -1,
    dragging: -1,
    calib: null,
    cols: 8,
    rows: 8,
    useMock: false,
    showSkeleton: true,
    showLabels: true,
    source: null,
    tracker: CV.tracking.createPlayerTracker(),
    game: null,
    players: [],
    view: [],
    fps: 0,
    lastHud: 0
  };

  var STORE_KEY = 'cv-floor-calibration';

  // ------------------------------------------------------------------ audio
  var audioCtx = null;
  function beep(freq, ms, type, gain) {
    try {
      if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      var o = audioCtx.createOscillator(), g = audioCtx.createGain();
      o.type = type || 'square';
      o.frequency.value = freq;
      g.gain.value = gain === undefined ? 0.05 : gain;
      g.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + ms / 1000);
      o.connect(g).connect(audioCtx.destination);
      o.start(); o.stop(audioCtx.currentTime + ms / 1000);
    } catch (e) { /* audio is optional */ }
  }

  function setStatus(s) { if (el.status) el.status.textContent = s; }

  // ------------------------------------------------------------------ camera
  function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return failHard('This browser cannot access a camera.',
                      'Try a recent Chrome, Edge, Firefox or Safari.');
    }
    S.phase = 'camera';
    setStatus('Requesting camera permission...');
    navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    }).then(function (stream) {
      video.srcObject = stream;
      return video.play();
    }).then(function () {
      return waitForVideo();
    }).then(function () {
      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;
      return initPoseSource();
    }).then(function () {
      restoreOrCalibrate();
    }).catch(function (err) {
      var name = (err && err.name) || 'Error';
      if (name === 'NotAllowedError' || name === 'SecurityError') {
        failHard('Camera permission denied.',
                 'Allow the camera in the address bar, then reload. ' +
                 'Note: the page must be served over https:// or localhost.');
      } else if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
        failHard('No camera found.', 'You can still try the game with "Mock players".');
      } else if (name === 'NotReadableError') {
        failHard('The camera is in use by another app.', 'Close Zoom/Meet/OBS and reload.');
      } else {
        failHard('Camera failed to start (' + name + ').', String((err && err.message) || ''));
      }
    });
  }

  function waitForVideo() {
    return new Promise(function (resolve) {
      if (video.videoWidth) return resolve();
      var tries = 0;
      var iv = setInterval(function () {
        if (video.videoWidth || ++tries > 60) { clearInterval(iv); resolve(); }
      }, 50);
    });
  }

  function failHard(msg, sub) {
    S.phase = 'error';
    S.error = msg + (sub ? ' ' + sub : '');
    setStatus(msg + ' ' + (sub || ''));
    if (!canvas.width) { canvas.width = 1280; canvas.height = 720; }
  }

  function initPoseSource() {
    if (S.source && S.source.dispose) S.source.dispose();
    if (S.useMock) {
      S.source = CV.pose.createMockSource({ count: 3 });
      S.source.setFrameSize(canvas.width, canvas.height);
      if (S.calib) S.source.setCalibration(S.calib);
      setStatus('Mock players active (no pose model).');
      return Promise.resolve();
    }
    setStatus('Loading pose model...');
    return CV.pose.createMediaPipeSource({
      numPoses: 4,
      model: 'lite',
      onStatus: setStatus
    }).then(function (src) {
      S.source = src;
      setStatus('Pose model ready (' + src.kind + ' ' + src.version + ', ' + src.delegate + ').');
    }).catch(function (err) {
      // Falling back keeps the app usable and makes it obvious that the
      // failure was the model download, not the game.
      S.useMock = true;
      syncButtons();
      S.source = CV.pose.createMockSource({ count: 3 });
      S.source.setFrameSize(canvas.width, canvas.height);
      if (S.calib) S.source.setCalibration(S.calib);
      setStatus('Pose model failed to load (' + ((err && err.message) || err) +
                '). Falling back to mock players.');
    });
  }

  // ------------------------------------------------------------ calibration
  function restoreOrCalibrate() {
    var saved = loadCalibration();
    if (saved) {
      S.corners = saved;
      if (buildCalibration()) {
        S.phase = 'ready';
        setStatus('Loaded saved calibration. Check the grid, then press START GAME.');
        return;
      }
    }
    beginCalibration();
  }

  function beginCalibration() {
    S.phase = 'calibrate';
    S.corners = [];
    S.activeCorner = 0;
    S.calib = null;
    setStatus('Click the four corners of the play area on the floor.');
  }

  function defaultQuad() {
    var w = canvas.width, h = canvas.height;
    // A trapezoid roughly matching a camera looking down at a floor: the far
    // edge is narrower. Only a starting point -- the corners are draggable.
    return [
      [w * 0.30, h * 0.42],
      [w * 0.70, h * 0.42],
      [w * 0.95, h * 0.95],
      [w * 0.05, h * 0.95]
    ];
  }

  function buildCalibration() {
    if (S.corners.length !== 4) return false;
    var c = CV.geometry.createCalibration(S.corners, S.cols, S.rows);
    if (!c) {
      setStatus('Those corners do not form a usable quad. Click them in order: ' +
                'top-left, top-right, bottom-right, bottom-left.');
      return false;
    }
    S.calib = c;
    if (S.source && S.source.setCalibration) S.source.setCalibration(c);
    S.tracker.reset();
    saveCalibration(S.corners);
    return true;
  }

  function saveCalibration(corners) {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({
        w: canvas.width, h: canvas.height, cols: S.cols, rows: S.rows, corners: corners
      }));
    } catch (e) { /* private mode */ }
  }

  function loadCalibration() {
    try {
      var raw = localStorage.getItem(STORE_KEY);
      if (!raw) return null;
      var d = JSON.parse(raw);
      // Corners are in pixels, so they are only valid at the resolution they
      // were clicked at.
      if (d.w !== canvas.width || d.h !== canvas.height) return null;
      if (!d.corners || d.corners.length !== 4) return null;
      if (d.cols) { S.cols = d.cols; S.rows = d.rows; if (el.grid) el.grid.value = String(d.cols); }
      return d.corners;
    } catch (e) { return null; }
  }

  function canvasPoint(ev) {
    var r = canvas.getBoundingClientRect();
    var cx = (ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left;
    var cy = (ev.touches ? ev.touches[0].clientY : ev.clientY) - r.top;
    return [cx * (canvas.width / r.width), cy * (canvas.height / r.height)];
  }

  function nearestCorner(p, maxDist) {
    var best = -1, bd = maxDist;
    for (var i = 0; i < S.corners.length; i++) {
      var d = Math.hypot(S.corners[i][0] - p[0], S.corners[i][1] - p[1]);
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }

  function onPointerDown(ev) {
    if (S.phase !== 'calibrate' && S.phase !== 'ready') return;
    ev.preventDefault();
    var p = canvasPoint(ev);
    var hit = nearestCorner(p, 28 * (canvas.width / canvas.getBoundingClientRect().width));
    if (hit >= 0) {
      S.dragging = hit;
      S.activeCorner = hit;
      S.phase = 'calibrate';
      return;
    }
    if (S.corners.length < 4) {
      S.corners.push(p);
      S.activeCorner = S.corners.length;
      if (S.corners.length === 4) {
        if (buildCalibration()) {
          S.phase = 'ready';
          setStatus('Calibrated. Drag a corner to adjust, then press START GAME.');
        }
      } else {
        setStatus('Corner ' + S.corners.length + ' of 4 placed.');
      }
    }
  }

  function onPointerMove(ev) {
    if (S.dragging < 0) return;
    ev.preventDefault();
    S.corners[S.dragging] = canvasPoint(ev);
    if (S.corners.length === 4) buildCalibration();
  }

  function onPointerUp() {
    if (S.dragging >= 0 && S.corners.length === 4 && S.calib) {
      S.phase = 'ready';
      setStatus('Calibration updated. Press START GAME.');
    }
    S.dragging = -1;
  }

  canvas.addEventListener('mousedown', onPointerDown);
  canvas.addEventListener('mousemove', onPointerMove);
  window.addEventListener('mouseup', onPointerUp);
  canvas.addEventListener('touchstart', onPointerDown, { passive: false });
  canvas.addEventListener('touchmove', onPointerMove, { passive: false });
  window.addEventListener('touchend', onPointerUp);

  // ------------------------------------------------------------------- game
  function newGame() {
    S.game = CV.game.createGame({
      cols: S.cols, rows: S.rows,
      seed: 20261004,
      onEvent: onGameEvent
    });
    S.game.start();
    S.phase = 'playing';
    setStatus('Game running. Move onto the right tile!');
  }

  function onGameEvent(e) {
    if (e.type === 'playerCorrect') {
      beep(900, 110, 'square'); setTimeout(function () { beep(1340, 90, 'square'); }, 70);
      flashTile(e.tile, CV.render.playerColor(e.id), 22);
    } else if (e.type === 'playerWrong') {
      beep(140, 240, 'sawtooth', 0.08);
      flashTile(e.tile, [255, 90, 60], 26);
    } else if (e.type === 'levelStart') {
      beep(600, 80, 'triangle', 0.04);
    } else if (e.type === 'gameOver' || e.type === 'completed') {
      beep(220, 420, 'sawtooth', 0.07);
    }
  }

  function flashTile(tile, color, count) {
    if (!tile || !S.calib) return;
    var c = S.calib.tileCenterCanvas(tile.row, tile.col);
    if (c) renderer.burst(c[0], c[1], color, count);
  }

  // ------------------------------------------------------------------- loop
  var lastT = 0, fpsAcc = 0, fpsN = 0;

  function loop(now) {
    requestAnimationFrame(loop);
    var dt = lastT ? Math.min(0.1, (now - lastT) / 1000) : 0;
    lastT = now;
    if (dt > 0) { fpsAcc += 1 / dt; fpsN++; if (fpsN >= 15) { S.fps = fpsAcc / fpsN; fpsAcc = fpsN = 0; } }

    // --- vision
    if (S.source && S.calib && (S.phase === 'ready' || S.phase === 'playing')) {
      var ready = S.source.kind === 'mock' || (video.readyState >= 2 && video.videoWidth);
      if (ready) {
        var res = null;
        try {
          res = S.source.detect(video, now);
        } catch (err) {
          setStatus('Pose detection error: ' + ((err && err.message) || err));
        }
        if (res && res.landmarks) {
          S.players = S.tracker.update(res.landmarks, S.calib, now, dt,
                                       canvas.width, canvas.height);
        }
      }
    }

    // --- game
    if (S.phase === 'playing' && S.game) {
      S.view = S.game.playersView(S.players);
      S.game.update(S.view, dt);
    } else if (S.game) {
      S.view = S.game.playersView(S.players);
    } else {
      S.view = [];
    }

    renderer.stepParticles(dt);
    draw();
    if (now - S.lastHud > 140) { S.lastHud = now; updateHud(); }
  }

  function draw() {
    renderer.clear();
    renderer.drawVideo(video);

    if (S.phase === 'error') {
      renderer.drawBanner(['CAMERA UNAVAILABLE', S.error], '#ff4842');
      return;
    }
    if (S.phase === 'boot') {
      renderer.drawBanner(['COMPUTER VISION FLOOR',
                           'Press START CAMERA to begin'], '#00e5ff');
      return;
    }

    if (S.phase === 'calibrate' || (S.phase === 'ready' && S.dragging >= 0)) {
      renderer.drawCalibrationOverlay(S.corners, S.activeCorner, S.calib);
      if (S.corners.length < 4) {
        var names = ['TOP LEFT', 'TOP RIGHT', 'BOTTOM RIGHT', 'BOTTOM LEFT'];
        renderer.drawBanner(['CLICK: ' + names[S.corners.length],
                             'Mark the four corners of the floor area'], '#ffc428');
      }
      return;
    }

    if (!S.calib) return;

    if (S.game) {
      renderer.drawTiles(S.calib, S.game.tiles, S.showLabels, S.game.state.phase);
    } else {
      renderer.drawGridPreview(S.calib);
    }

    var byId = {};
    for (var i = 0; i < S.view.length; i++) byId[S.view[i].id] = S.view[i];
    for (var p = 0; p < S.players.length; p++) {
      renderer.drawPlayerMarker(S.calib, S.players[p], byId[S.players[p].id],
                                S.showSkeleton, canvas.width, canvas.height);
    }
    renderer.drawParticles();

    if (S.game) {
      renderer.drawRoundTimer(S.game.state);
      var st = S.game.state;
      if (st.phase === 'gameover') {
        var done = st.level >= S.game.options.totalLevels;
        renderer.drawBanner([done ? 'FLOOR COMPLETE' : 'GAME OVER',
                             'Press RESTART GAME'], done ? '#a8ff46' : '#ff4842');
      } else if (st.phase === 'resolve') {
        var okCount = 0;
        for (var r = 0; r < st.lastResults.length; r++) if (st.lastResults[r].ok) okCount++;
        renderer.drawBanner([okCount ? 'CORRECT' : 'MISSED',
                             st.question ? st.question.note : 'The marked tiles collapse'],
                            okCount ? '#a8ff46' : '#ff4842');
      }
    } else if (S.phase === 'ready') {
      renderer.drawBanner(['CALIBRATED', 'Press START GAME'], '#a8ff46');
    }

    if (S.players.length === 0 && S.phase === 'playing') {
      renderer.drawBanner(['NO PLAYER DETECTED',
                           'Step into the camera view, feet inside the grid'], '#ffc428');
    }
  }

  function updateHud() {
    if (el.level) {
      el.level.textContent = S.game
        ? ('LEVEL ' + S.game.state.level + ' / ' + S.game.state.totalLevels)
        : 'LEVEL - / -';
    }
    if (el.phase) {
      var bits = [];
      if (S.game) bits.push(S.game.state.levelType ? S.game.state.levelType.toUpperCase() : '');
      bits.push(S.fps.toFixed(0) + ' FPS');
      bits.push((S.source ? S.source.kind : 'no source'));
      bits.push(S.players.length + ' TRACKED');
      el.phase.textContent = bits.filter(Boolean).join('  |  ');
    }
    if (el.question) {
      if (S.game && S.game.state.question && S.game.state.phase === 'active') {
        el.question.textContent = S.game.state.question.q;
        el.questionSub.textContent = 'Stand on your answer before the timer ends';
        el.questionBar.style.display = '';
      } else if (S.game && S.game.state.levelType === 'survival' && S.game.state.phase === 'active') {
        el.question.textContent = 'SURVIVAL: the glowing tiles are about to collapse';
        el.questionSub.textContent = 'Get onto a safe tile';
        el.questionBar.style.display = '';
      } else if (S.game && S.game.state.phase === 'resolve' && S.game.state.question) {
        el.question.textContent = S.game.state.question.q;
        el.questionSub.textContent = 'Answer: ' +
          S.game.state.question.options[S.game.state.question.answer] +
          ' - ' + S.game.state.question.note;
        el.questionBar.style.display = '';
      } else {
        el.questionBar.style.display = 'none';
      }
    }
    if (el.players) {
      if (!S.view.length) {
        el.players.innerHTML = '<div class="pl empty">no players detected</div>';
      } else {
        var html = '';
        for (var i = 0; i < S.view.length; i++) {
          var v = S.view[i];
          var c = CV.render.playerColor(v.id);
          var hearts = '';
          for (var k = 0; k < 3; k++) hearts += k < v.lives ? '\u2665' : '\u2661';
          html += '<div class="pl" style="border-left-color: rgb(' + c.join(',') + ')">' +
            '<span class="pn">' + v.name + '</span>' +
            '<span class="pt">' + v.tileLabel + '</span>' +
            '<span class="ps">' + v.score + '</span>' +
            '<span class="ph">' + hearts + '</span>' +
            '<span class="pst ' + (v.status === 'OUT' ? 'out' : '') + '">' + v.status + '</span>' +
            '</div>';
        }
        el.players.innerHTML = html;
      }
    }
    if (el.hint) {
      el.hint.textContent = S.phase === 'calibrate'
        ? 'Click the 4 floor corners (or press USE DEFAULT and drag them).'
        : '';
    }
  }

  // --------------------------------------------------------------- controls
  function syncButtons() {
    if (el.btnMock) el.btnMock.textContent = S.useMock ? 'Mock players: ON' : 'Mock players: OFF';
    if (el.btnSkeleton) el.btnSkeleton.textContent = S.showSkeleton ? 'Skeleton: ON' : 'Skeleton: OFF';
    if (el.btnLabels) el.btnLabels.textContent = S.showLabels ? 'Tile labels: ON' : 'Tile labels: OFF';
  }

  el.btnStart.addEventListener('click', function () {
    if (S.phase === 'boot' || S.phase === 'error') return startCamera();
    if (S.phase === 'ready') return newGame();
    if (S.phase === 'playing') return;
    setStatus('Finish the calibration first: click the four floor corners.');
  });

  el.btnRecal.addEventListener('click', function () {
    if (!canvas.width) { canvas.width = 1280; canvas.height = 720; }
    S.game = null;
    beginCalibration();
  });

  el.btnRestart.addEventListener('click', function () {
    if (!S.calib) return setStatus('Calibrate the floor first.');
    newGame();
  });

  el.btnDefault.addEventListener('click', function () {
    if (!canvas.width) { canvas.width = 1280; canvas.height = 720; }
    S.corners = defaultQuad();
    S.activeCorner = -1;
    if (buildCalibration()) {
      S.phase = 'ready';
      setStatus('Default floor placed. Drag the corners to match your real floor.');
    }
  });

  el.btnMock.addEventListener('click', function () {
    S.useMock = !S.useMock;
    syncButtons();
    S.tracker.reset();
    initPoseSource();
  });

  el.btnSkeleton.addEventListener('click', function () {
    S.showSkeleton = !S.showSkeleton; syncButtons();
  });

  el.btnLabels.addEventListener('click', function () {
    S.showLabels = !S.showLabels; syncButtons();
  });

  if (el.grid) {
    el.grid.addEventListener('change', function () {
      S.cols = S.rows = parseInt(el.grid.value, 10) || 8;
      if (S.corners.length === 4) buildCalibration();
      if (S.game) S.game = null;
      setStatus('Grid set to ' + S.cols + ' x ' + S.rows + '. Press START GAME.');
    });
  }

  window.addEventListener('keydown', function (e) {
    if (e.key === 'r' || e.key === 'R') el.btnRecal.click();
    if (e.key === 's' || e.key === 'S') el.btnSkeleton.click();
    if (e.key === 'l' || e.key === 'L') el.btnLabels.click();
    if (e.key === 'm' || e.key === 'M') el.btnMock.click();
  });

  // Debug handle: inspect CVFloorApp.S from the console when something looks
  // wrong on a real camera.
  window.CVFloorApp = S;

  canvas.width = 1280;
  canvas.height = 720;
  syncButtons();
  setStatus('Press START CAMERA.');
  requestAnimationFrame(loop);
})();
