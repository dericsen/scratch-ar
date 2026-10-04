/* HOLO LAVA -- the floor is lava, played with your body.
 *
 * Rules: the bottom of the screen is a row of tiles. Every round one tile is
 * safe and the rest turn to lava. Physically step left or right so that your
 * body is standing on the safe tile before it erupts. Sometimes a wave sweeps
 * every tile at once and the only way out is to JUMP.
 *
 * Everything the game knows about the player comes from vision.js: a
 * silhouette extracted by background subtraction, summarised into which lane
 * holds the most body mass, plus a jump detector keyed on the body rising
 * above its own learned resting height.
 *
 * No engine, no framework, no assets. Canvas 2D for drawing and WebAudio for
 * the beeps, both built in to the browser.
 */
(function () {
  'use strict';

  var PROC_W = 160, PROC_H = 120;   // vision runs on a small frame; it only
  var LANES = 4;                    // needs a silhouette, not detail
  var CANVAS_W = 960, CANVAS_H = 720;

  var COL = {
    bg: '#07080d',
    cyan: '#00e5ff',
    magenta: '#ff40b4',
    lime: '#a8ff46',
    amber: '#ffc428',
    red: '#ff4842',
    white: '#f5f8ff',
    stone: '#5a6072',
    stoneLit: '#8e97ad'
  };

  // ---------------------------------------------------------------- elements
  var canvas = document.getElementById('stage');
  var ctx = canvas.getContext('2d');
  canvas.width = CANVAS_W;
  canvas.height = CANVAS_H;

  var video = document.getElementById('cam');
  var proc = document.createElement('canvas');
  proc.width = PROC_W;
  proc.height = PROC_H;
  var pctx = proc.getContext('2d', { willReadFrequently: true });

  // Offscreen canvas used to blow the small silhouette mask up to screen size.
  var maskCanvas = document.createElement('canvas');
  maskCanvas.width = PROC_W;
  maskCanvas.height = PROC_H;
  var mctx = maskCanvas.getContext('2d');
  var maskImage = mctx.createImageData(PROC_W, PROC_H);

  var elStatus = document.getElementById('status');
  var elStart = document.getElementById('btn-start');
  var elRecal = document.getElementById('btn-recal');
  var elDebug = document.getElementById('btn-debug');
  var elMute = document.getElementById('btn-mute');
  var elSens = document.getElementById('sens');

  // ------------------------------------------------------------------ vision
  var detector = HoloVision.createDetector({
    width: PROC_W, height: PROC_H, gridCols: 40, gridRows: 30
  });
  var tracker = HoloVision.createPlayerTracker({ lanes: LANES, jumpRise: 0.05 });
  var gray = new Uint8Array(PROC_W * PROC_H);

  // -------------------------------------------------------------------- state
  var G = {
    phase: 'boot',      // boot | calibrate | ready | play | over | error
    message: '',
    sub: '',
    debug: false,
    muted: false,

    calibCountdown: 3.0,
    calibFrames: 0,

    score: 0,
    best: Number(localStorage.getItem('holo-lava-best') || 0),
    combo: 1,
    streak: 0,
    hearts: 3,
    level: 1,
    rounds: 0,

    round: null,
    shake: 0,
    toasts: [],
    particles: [],
    vision: null,
    player: null,
    fps: 0
  };

  // -------------------------------------------------------------------- audio
  var audioCtx = null;
  function beep(freq, durMs, type, gainVal) {
    if (G.muted) return;
    try {
      if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      var osc = audioCtx.createOscillator();
      var gain = audioCtx.createGain();
      osc.type = type || 'square';
      osc.frequency.value = freq;
      gain.gain.value = gainVal === undefined ? 0.06 : gainVal;
      gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + durMs / 1000);
      osc.connect(gain).connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + durMs / 1000);
    } catch (e) { /* audio is a nicety, never a hard failure */ }
  }
  var SFX = {
    safe: function () { beep(880, 120, 'square'); setTimeout(function () { beep(1320, 100, 'square'); }, 70); },
    hit: function () { beep(130, 260, 'sawtooth', 0.09); },
    jump: function () { beep(520, 90, 'triangle', 0.05); },
    warn: function () { beep(300, 60, 'square', 0.03); },
    bonus: function () { beep(1760, 90, 'square', 0.05); },
    over: function () { beep(220, 400, 'sawtooth', 0.08); }
  };

  // ------------------------------------------------------------------- camera
  function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return fail('Browser ini tidak mendukung akses kamera.',
                  'Coba Chrome, Edge, atau Firefox versi terbaru.');
    }
    setStatus('Meminta izin kamera...');
    navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    }).then(function (stream) {
      video.srcObject = stream;
      return video.play();
    }).then(function () {
      beginCalibration();
    }).catch(function (err) {
      var name = err && err.name ? err.name : 'Error';
      if (name === 'NotAllowedError' || name === 'SecurityError') {
        fail('Izin kamera ditolak.',
             'Klik ikon kamera di address bar, pilih Allow, lalu muat ulang halaman.');
      } else if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
        fail('Tidak ada kamera yang terdeteksi.', 'Pastikan webcam terpasang dan tidak dipakai aplikasi lain.');
      } else if (name === 'NotReadableError') {
        fail('Kamera sedang dipakai aplikasi lain.', 'Tutup Zoom/Meet/OBS lalu coba lagi.');
      } else {
        fail('Kamera gagal dibuka (' + name + ').', 'Coba muat ulang halaman.');
      }
    });
  }

  function fail(msg, sub) {
    G.phase = 'error';
    G.message = msg;
    G.sub = sub || '';
    setStatus(msg);
  }

  function setStatus(s) { if (elStatus) elStatus.textContent = s; }

  // -------------------------------------------------------------- calibration
  function beginCalibration() {
    G.phase = 'calibrate';
    G.calibCountdown = 3.0;
    G.calibFrames = 0;
    detector.resetCalibration();
    tracker.reset();
    setStatus('Kalibrasi: keluar dulu dari frame.');
  }

  function stepCalibration(dt) {
    if (G.calibCountdown > 0) {
      var before = Math.ceil(G.calibCountdown);
      G.calibCountdown -= dt;
      if (Math.ceil(G.calibCountdown) !== before) SFX.warn();
      return;
    }
    // Countdown finished: average ~30 frames of the empty room.
    detector.learnBackground(gray);
    G.calibFrames++;
    if (G.calibFrames >= 30) {
      detector.finishCalibration();
      G.phase = 'ready';
      setStatus('Siap. Masuk kembali ke frame.');
    }
  }

  // --------------------------------------------------------------- game rounds
  function resetGame() {
    G.score = 0;
    G.combo = 1;
    G.streak = 0;
    G.hearts = 3;
    G.level = 1;
    G.rounds = 0;
    G.particles.length = 0;
    G.toasts.length = 0;
    G.round = null;
    G.phase = 'play';
    nextRound();
  }

  function warnDuration() {
    // Shrinks with level but never becomes physically impossible to react to.
    return Math.max(0.95, 2.1 - (G.level - 1) * 0.13);
  }

  function nextRound() {
    G.rounds++;
    // Every fourth round is a wave: no safe tile exists, the only escape is up.
    var isWave = G.rounds > 2 && G.rounds % 4 === 0;
    G.round = {
      isWave: isWave,
      safeLane: isWave ? -1 : Math.floor(Math.random() * LANES),
      phase: 'warn',
      t: 0,
      duration: warnDuration(),
      outcome: null,
      jumpedAtErupt: false
    };
    if (isWave) toast('GELOMBANG LAVA - LOMPAT!', COL.magenta, 1.1);
  }

  function resolveRound() {
    var r = G.round;
    var p = G.player || {};
    var survived, bonus = 0, label, color;

    if (r.isWave) {
      survived = !!p.airborne;
      label = survived ? 'NICE JUMP!' : 'KENA GELOMBANG!';
      color = survived ? COL.lime : COL.red;
    } else {
      survived = p.lane === r.safeLane;
      if (survived && p.airborne) {
        bonus = 50;
        label = 'PERFECT HOP! +50';
        color = COL.amber;
      } else {
        label = survived ? 'AMAN!' : 'KECEBUR LAVA!';
        color = survived ? COL.lime : COL.red;
      }
    }

    if (survived) {
      G.streak++;
      G.combo = Math.min(8, 1 + Math.floor(G.streak / 3));
      var gained = (r.isWave ? 150 : 100) * G.combo + bonus;
      G.score += gained;
      if (bonus) SFX.bonus(); else SFX.safe();
      if (G.streak > 0 && G.streak % 5 === 0) {
        G.level++;
        toast('LEVEL ' + G.level, COL.cyan, 1.0);
      }
    } else {
      G.hearts--;
      G.streak = 0;
      G.combo = 1;
      G.shake = 1.0;
      SFX.hit();
    }
    toast(label, color, 1.0);
    eruptParticles(r);

    r.outcome = { survived: survived, label: label, color: color };
    r.phase = 'resolve';
    r.t = 0;
    r.duration = 0.85;

    if (G.hearts <= 0) {
      G.phase = 'over';
      if (G.score > G.best) {
        G.best = G.score;
        localStorage.setItem('holo-lava-best', String(G.best));
      }
      SFX.over();
      setStatus('Game over. Skor ' + G.score);
    }
  }

  function stepRound(dt) {
    var r = G.round;
    if (!r) return nextRound();
    r.t += dt;
    if (r.phase === 'warn') {
      if (r.t >= r.duration) {
        r.jumpedAtErupt = !!(G.player && G.player.airborne);
        resolveRound();
      }
    } else if (r.phase === 'resolve') {
      if (r.t >= r.duration && G.phase === 'play') nextRound();
    }
  }

  // ------------------------------------------------------------------ effects
  function toast(text, color, life) {
    G.toasts.push({ text: text, color: color, life: life, max: life });
  }

  function eruptParticles(r) {
    for (var lane = 0; lane < LANES; lane++) {
      if (!r.isWave && lane === r.safeLane) continue;
      var rect = laneRect(lane);
      for (var i = 0; i < 26; i++) {
        G.particles.push({
          x: rect.x + Math.random() * rect.w,
          y: rect.y + Math.random() * 20,
          vx: (Math.random() - 0.5) * 120,
          vy: -120 - Math.random() * 260,
          life: 0.5 + Math.random() * 0.6,
          max: 1.1,
          size: 2 + Math.random() * 4
        });
      }
    }
  }

  function stepEffects(dt) {
    for (var i = G.toasts.length - 1; i >= 0; i--) {
      G.toasts[i].life -= dt;
      if (G.toasts[i].life <= 0) G.toasts.splice(i, 1);
    }
    for (var j = G.particles.length - 1; j >= 0; j--) {
      var p = G.particles[j];
      p.life -= dt;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.vy += 620 * dt;
      if (p.life <= 0) G.particles.splice(j, 1);
    }
    G.shake = Math.max(0, G.shake - dt * 2.2);
  }

  // ------------------------------------------------------------------ layout
  var FLOOR_TOP = 0.66;   // tiles occupy the bottom third of the screen

  function laneRect(lane) {
    var w = CANVAS_W / LANES;
    return { x: lane * w, y: CANVAS_H * FLOOR_TOP, w: w, h: CANVAS_H * (1 - FLOOR_TOP) };
  }

  // ------------------------------------------------------------------- frame
  function grabFrame() {
    if (video.readyState < 2 || !video.videoWidth) return false;
    // Mirror at capture time. Everything downstream -- vision, lanes, drawing
    // -- then lives in the same "as seen in a mirror" space, so moving to your
    // physical left moves you left on screen and no axis has to be flipped
    // twice (which is exactly the bug that makes these games feel inverted).
    pctx.save();
    pctx.scale(-1, 1);
    pctx.drawImage(video, -PROC_W, 0, PROC_W, PROC_H);
    pctx.restore();
    var data = pctx.getImageData(0, 0, PROC_W, PROC_H).data;
    HoloVision.toGrayscale(data, PROC_W, PROC_H, PROC_W, PROC_H, gray);
    return true;
  }

  // -------------------------------------------------------------------- loop
  var lastT = 0, fpsAcc = 0, fpsN = 0;

  function loop(now) {
    requestAnimationFrame(loop);
    var dt = lastT ? Math.min(0.05, (now - lastT) / 1000) : 0;
    lastT = now;
    if (dt > 0) { fpsAcc += 1 / dt; fpsN++; if (fpsN >= 20) { G.fps = fpsAcc / fpsN; fpsAcc = 0; fpsN = 0; } }

    var haveFrame = grabFrame();

    if (haveFrame) {
      if (G.phase === 'calibrate') {
        stepCalibration(dt);
      } else if (detector.isReady()) {
        G.vision = detector.update(gray);
        G.player = tracker.update(G.vision, now);
        if (G.player && G.player.jumped) SFX.jump();

        if (G.vision.backgroundStale && (G.phase === 'play' || G.phase === 'ready')) {
          // The room no longer matches the model: lights changed or the camera
          // moved. Guessing would punish the player for something they cannot
          // see, so stop and ask for a recalibration instead.
          G.phase = 'ready';
          setStatus('Latar berubah - tekan Kalibrasi Ulang.');
        }

        if (G.phase === 'ready' && G.player.present && !G.vision.backgroundStale) {
          resetGame();
        } else if (G.phase === 'play') {
          stepRound(dt);
        }
      }
    }
    stepEffects(dt);
    draw();
  }

  // -------------------------------------------------------------------- draw
  function draw() {
    ctx.save();
    if (G.shake > 0) {
      var m = G.shake * 14;
      ctx.translate((Math.random() - 0.5) * m, (Math.random() - 0.5) * m);
    }

    ctx.fillStyle = COL.bg;
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    // --- camera feed, mirrored and dimmed so the overlay stays readable
    if (video.readyState >= 2 && video.videoWidth) {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.globalAlpha = 0.62;
      ctx.drawImage(video, -CANVAS_W, 0, CANVAS_W, CANVAS_H);
      ctx.restore();
      ctx.globalAlpha = 1;
    }

    if (G.vision && (G.phase === 'play' || G.phase === 'ready' || G.phase === 'over')) {
      drawSilhouette();
    }

    if (G.phase === 'play' || G.phase === 'over') {
      drawFloor();
      drawParticles();
      drawHud();
    }

    if (G.debug) drawDebug();
    drawToasts();
    drawOverlay();
    ctx.restore();
  }

  /* Paint the detected body as a neon wash. This is the game's honesty
   * window: if the silhouette looks wrong, the player can see why they are
   * dying instead of blaming the game. */
  function drawSilhouette() {
    var mask = G.vision.mask;
    var d = maskImage.data;
    var cyan = [0, 229, 255];
    for (var i = 0; i < mask.length; i++) {
      var on = mask[i];
      var o = i * 4;
      d[o] = cyan[0]; d[o + 1] = cyan[1]; d[o + 2] = cyan[2];
      d[o + 3] = on ? 70 : 0;
    }
    mctx.putImageData(maskImage, 0, 0);
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.globalCompositeOperation = 'lighter';
    ctx.drawImage(maskCanvas, 0, 0, CANVAS_W, CANVAS_H);
    ctx.restore();
  }

  function drawFloor() {
    var r = G.round;
    var pulse = r && r.phase === 'warn' ? (0.55 + 0.45 * Math.sin(r.t * 14)) : 1;
    var erupting = r && r.phase === 'resolve';

    for (var lane = 0; lane < LANES; lane++) {
      var rect = laneRect(lane);
      var safe = r && !r.isWave && lane === r.safeLane;
      var top = rect.y;

      if (safe) {
        // Safe tile: solid stone with a lime rim.
        ctx.fillStyle = COL.stone;
        ctx.fillRect(rect.x + 3, top, rect.w - 6, rect.h);
        ctx.fillStyle = COL.stoneLit;
        ctx.fillRect(rect.x + 3, top, rect.w - 6, 10);
        ctx.strokeStyle = COL.lime;
        ctx.lineWidth = 4;
        ctx.strokeRect(rect.x + 3, top, rect.w - 6, rect.h - 2);
        ctx.fillStyle = COL.lime;
        ctx.font = 'bold 22px ui-monospace, Menlo, monospace';
        ctx.textAlign = 'center';
        ctx.fillText('AMAN', rect.x + rect.w / 2, top + rect.h / 2 + 8);
      } else {
        // Lava tile: brighter and more violent once it erupts.
        var heat = erupting ? 1 : 0.45 + 0.35 * pulse;
        var grad = ctx.createLinearGradient(0, top, 0, top + rect.h);
        grad.addColorStop(0, 'rgba(255,' + Math.round(90 + 90 * heat) + ',30,' + (0.55 + 0.45 * heat) + ')');
        grad.addColorStop(1, 'rgba(150,20,10,' + (0.6 + 0.4 * heat) + ')');
        ctx.fillStyle = grad;
        ctx.fillRect(rect.x + 3, top, rect.w - 6, rect.h);

        // Wobbling lava surface.
        ctx.beginPath();
        ctx.moveTo(rect.x + 3, top + 6);
        for (var x = 0; x <= rect.w - 6; x += 8) {
          var yy = top + 6 + Math.sin((x + performance.now() / 90 + lane * 30) / 16) * 5 * heat;
          ctx.lineTo(rect.x + 3 + x, yy);
        }
        ctx.strokeStyle = 'rgba(255,220,120,' + (0.5 + 0.5 * heat) + ')';
        ctx.lineWidth = 3;
        ctx.stroke();
      }

      // Lane occupancy bar: shows the game's read of where your body is.
      if (G.player && G.player.laneWeights) {
        var wgt = G.player.laneWeights[lane] || 0;
        var bw = (rect.w - 24) * Math.min(1, wgt * 1.6);
        ctx.fillStyle = 'rgba(0,229,255,0.75)';
        ctx.fillRect(rect.x + 12, CANVAS_H - 16, bw, 7);
        ctx.strokeStyle = 'rgba(255,255,255,0.25)';
        ctx.lineWidth = 1;
        ctx.strokeRect(rect.x + 12, CANVAS_H - 16, rect.w - 24, 7);
      }
    }

    // Where the game thinks you are standing.
    if (G.player && G.player.lane !== null && G.round) {
      var pr = laneRect(G.player.lane);
      ctx.strokeStyle = G.player.airborne ? COL.amber : COL.cyan;
      ctx.lineWidth = 3;
      ctx.setLineDash([10, 7]);
      ctx.strokeRect(pr.x + 7, pr.y - 4, pr.w - 14, pr.h - 8);
      ctx.setLineDash([]);
      if (G.player.airborne) {
        ctx.fillStyle = COL.amber;
        ctx.font = 'bold 18px ui-monospace, Menlo, monospace';
        ctx.textAlign = 'center';
        ctx.fillText('DI UDARA', pr.x + pr.w / 2, pr.y - 14);
      }
    }

    // Countdown bar for the current round.
    if (G.round && G.round.phase === 'warn') {
      var frac = 1 - G.round.t / G.round.duration;
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.fillRect(0, CANVAS_H * FLOOR_TOP - 14, CANVAS_W, 8);
      ctx.fillStyle = G.round.isWave ? COL.magenta : COL.amber;
      ctx.fillRect(0, CANVAS_H * FLOOR_TOP - 14, CANVAS_W * frac, 8);
    }
  }

  function drawParticles() {
    for (var i = 0; i < G.particles.length; i++) {
      var p = G.particles[i];
      var a = Math.max(0, p.life / p.max);
      ctx.fillStyle = 'rgba(255,' + Math.round(120 + 110 * a) + ',40,' + a + ')';
      ctx.fillRect(p.x, p.y, p.size, p.size);
    }
  }

  function drawHud() {
    panel(14, 14, 250, 108);
    ctx.textAlign = 'left';
    ctx.fillStyle = COL.cyan;
    ctx.font = 'bold 15px ui-monospace, Menlo, monospace';
    ctx.fillText('HOLO LAVA', 28, 38);
    ctx.fillStyle = COL.white;
    ctx.font = 'bold 30px ui-monospace, Menlo, monospace';
    ctx.fillText(String(G.score), 28, 72);
    ctx.font = '13px ui-monospace, Menlo, monospace';
    ctx.fillStyle = 'rgba(245,248,255,0.75)';
    ctx.fillText('BEST ' + G.best + '   LEVEL ' + G.level, 28, 94);
    if (G.combo > 1) {
      ctx.fillStyle = COL.amber;
      ctx.font = 'bold 15px ui-monospace, Menlo, monospace';
      ctx.fillText('COMBO x' + G.combo, 28, 114);
    }

    // Hearts
    ctx.textAlign = 'right';
    for (var i = 0; i < 3; i++) {
      ctx.fillStyle = i < G.hearts ? COL.red : 'rgba(255,255,255,0.16)';
      ctx.font = '28px ui-monospace, Menlo, monospace';
      ctx.fillText('\u2665', CANVAS_W - 22 - i * 32, 44);
    }
  }

  function panel(x, y, w, h) {
    ctx.fillStyle = 'rgba(7,8,13,0.72)';
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = 'rgba(0,229,255,0.28)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, w, h);
  }

  function drawToasts() {
    ctx.textAlign = 'center';
    for (var i = 0; i < G.toasts.length; i++) {
      var t = G.toasts[i];
      var a = Math.min(1, t.life / (t.max * 0.5));
      var rise = (1 - t.life / t.max) * 40;
      ctx.globalAlpha = a;
      ctx.fillStyle = t.color;
      ctx.font = 'bold 44px ui-monospace, Menlo, monospace';
      ctx.fillText(t.text, CANVAS_W / 2, CANVAS_H * 0.42 - rise + i * 46);
      ctx.globalAlpha = 1;
    }
  }

  function drawDebug() {
    if (!G.vision) return;
    var cols = G.vision.gridCols, rows = G.vision.gridRows, occ = G.vision.occupancy;
    var cw = CANVAS_W / cols, ch = CANVAS_H / rows;
    for (var gy = 0; gy < rows; gy++) {
      for (var gx = 0; gx < cols; gx++) {
        var e = occ[gy * cols + gx];
        if (e <= 0.02) continue;
        ctx.fillStyle = 'rgba(255,64,180,' + (0.18 + 0.5 * e) + ')';
        ctx.fillRect(gx * cw, gy * ch, cw - 1, ch - 1);
      }
    }
    panel(CANVAS_W - 254, CANVAS_H - 132, 240, 118);
    ctx.textAlign = 'left';
    ctx.fillStyle = COL.magenta;
    ctx.font = '12px ui-monospace, Menlo, monospace';
    var p = G.player || {};
    var lines = [
      'DEBUG VIEW',
      'fps          ' + G.fps.toFixed(0),
      'threshold    ' + G.vision.threshold.toFixed(1),
      'coverage     ' + (G.vision.coverage * 100).toFixed(1) + '%',
      'lane         ' + (p.lane === null || p.lane === undefined ? '-' : p.lane),
      'airborne     ' + (p.airborne ? 'yes' : 'no'),
      'baselineY    ' + (p.baselineY ? p.baselineY.toFixed(3) : '-')
    ];
    for (var i = 0; i < lines.length; i++) {
      ctx.fillText(lines[i], CANVAS_W - 240, CANVAS_H - 112 + i * 15);
    }
  }

  /* Full-screen prompts for every non-playing state. */
  function drawOverlay() {
    if (G.phase === 'play') return;
    ctx.fillStyle = 'rgba(7,8,13,0.78)';
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
    ctx.textAlign = 'center';

    var title = '', lines = [], color = COL.cyan;

    if (G.phase === 'boot') {
      title = 'HOLO LAVA';
      lines = [
        'Lantainya lava. Pindah ke kotak AMAN pakai badanmu.',
        '',
        'Tekan MULAI, lalu izinkan akses kamera.',
        'Berdiri sekitar 2 meter dari kamera supaya kamu',
        'punya ruang untuk geser kiri-kanan.'
      ];
    } else if (G.phase === 'error') {
      title = 'KAMERA BERMASALAH';
      color = COL.red;
      lines = [G.message, '', G.sub];
    } else if (G.phase === 'calibrate') {
      color = COL.amber;
      if (G.calibCountdown > 0) {
        title = 'KELUAR DARI FRAME: ' + Math.ceil(G.calibCountdown);
        lines = [
          'Program sedang menghafal bentuk ruanganmu yang kosong.',
          'Menyingkir dulu dari pandangan kamera.'
        ];
      } else {
        title = 'MEREKAM LATAR...';
        lines = ['Jangan bergerak di depan kamera (' + G.calibFrames + '/30)'];
      }
    } else if (G.phase === 'ready') {
      color = COL.lime;
      title = 'MASUK KE FRAME';
      lines = [
        'Berdiri di depan kamera, permainan mulai otomatis.',
        '',
        'Geser badan kiri/kanan untuk pindah kotak.',
        'LOMPAT kalau muncul GELOMBANG LAVA.',
        '',
        'Kalau siluetmu terlihat aneh, tekan Kalibrasi Ulang.'
      ];
    } else if (G.phase === 'over') {
      color = COL.red;
      title = 'GAME OVER';
      lines = [
        'Skor ' + G.score + '   Best ' + G.best,
        'Bertahan ' + G.rounds + ' putaran, level ' + G.level,
        '',
        'Keluar dari frame lalu masuk lagi untuk main ulang,',
        'atau tekan Kalibrasi Ulang.'
      ];
      // Leaving and re-entering restarts: a natural gesture, no button needed.
      if (G.player && !G.player.present) {
        G.phase = 'ready';
        G.hearts = 3;
      }
    }

    ctx.fillStyle = color;
    ctx.font = 'bold 52px ui-monospace, Menlo, monospace';
    ctx.fillText(title, CANVAS_W / 2, CANVAS_H * 0.34);
    ctx.fillStyle = COL.white;
    ctx.font = '19px ui-monospace, Menlo, monospace';
    for (var i = 0; i < lines.length; i++) {
      ctx.fillText(lines[i], CANVAS_W / 2, CANVAS_H * 0.44 + i * 28);
    }

    if (G.phase === 'calibrate' && G.calibCountdown <= 0) {
      var w = 320, x = (CANVAS_W - w) / 2, y = CANVAS_H * 0.62;
      ctx.fillStyle = 'rgba(255,255,255,0.15)';
      ctx.fillRect(x, y, w, 12);
      ctx.fillStyle = COL.amber;
      ctx.fillRect(x, y, w * Math.min(1, G.calibFrames / 30), 12);
    }
  }

  // ------------------------------------------------------------------ controls
  elStart.addEventListener('click', function () {
    if (G.phase === 'boot' || G.phase === 'error') startCamera();
    else beginCalibration();
  });
  elRecal.addEventListener('click', beginCalibration);
  elDebug.addEventListener('click', function () {
    G.debug = !G.debug;
    elDebug.textContent = G.debug ? 'Debug: ON' : 'Debug: OFF';
  });
  elMute.addEventListener('click', function () {
    G.muted = !G.muted;
    elMute.textContent = G.muted ? 'Suara: OFF' : 'Suara: ON';
  });
  elSens.addEventListener('input', function () {
    detector.setSensitivity(Number(elSens.value));
  });

  window.addEventListener('keydown', function (e) {
    if (e.key === 'd' || e.key === 'D') elDebug.click();
    if (e.key === 'r' || e.key === 'R') beginCalibration();
    if (e.key === 'm' || e.key === 'M') elMute.click();
  });

  /* Debug handle. Useful from the browser console when the camera is doing
   * something unexpected (inspect HoloLavaDebug.G.vision.coverage, etc.), and
   * it is what the headless harness in tests/game_test.js drives the game
   * through -- there is no browser in the dev sandbox, so the round logic is
   * exercised with synthetic frames instead. */
  window.HoloLavaDebug = {
    G: G,
    detector: detector,
    tracker: tracker,
    laneRect: laneRect,
    startCamera: startCamera,
    beginCalibration: beginCalibration
  };

  setStatus('Tekan MULAI untuk menyalakan kamera.');
  requestAnimationFrame(loop);
})();
