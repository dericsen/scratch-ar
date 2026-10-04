/* AR overlay renderer.
 *
 * The camera image stays visible underneath everything: real people on a real
 * floor, with the virtual game drawn on top. Nothing here is a 3-D scene --
 * every tile, marker and ring is a 2-D polygon whose vertices were projected
 * through the calibration homography, which is why they sit flat on the floor
 * and foreshorten correctly as they recede.
 */
(function (root) {
  'use strict';

  var ns = root.CVFloor = root.CVFloor || {};

  var PLAYER_COLORS = [
    [0, 229, 255],    // cyan
    [255, 64, 180],   // magenta
    [168, 255, 70],   // lime
    [255, 196, 40],   // amber
    [150, 110, 255],  // violet
    [255, 110, 70]    // coral
  ];

  function playerColor(id) { return PLAYER_COLORS[(id - 1) % PLAYER_COLORS.length]; }
  function rgba(c, a) { return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')'; }

  var TILE_STYLE = {
    empty:   { fill: [0, 229, 255, 0.06], stroke: [0, 229, 255, 0.22], line: 1 },
    safe:    { fill: [90, 230, 140, 0.16], stroke: [120, 255, 170, 0.5], line: 1.5 },
    danger:  { fill: [255, 110, 40, 0.3],  stroke: [255, 170, 80, 0.8], line: 2 },
    cracked: { fill: [40, 30, 36, 0.55],   stroke: [120, 90, 100, 0.6], line: 1 },
    broken:  { fill: [12, 10, 14, 0.78],   stroke: [90, 60, 70, 0.55], line: 1 },
    answer:  { fill: [40, 120, 255, 0.26], stroke: [120, 190, 255, 0.85], line: 2 },
    correct: { fill: [60, 240, 130, 0.42], stroke: [160, 255, 200, 0.95], line: 3 },
    wrong:   { fill: [255, 60, 60, 0.42],  stroke: [255, 150, 150, 0.95], line: 3 }
  };

  function createRenderer(canvas) {
    var ctx = canvas.getContext('2d');
    var particles = [];

    function clear() {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    function drawVideo(video) {
      // Not mirrored. This is a third-person camera watching a floor, so
      // flipping it would put players on the wrong side of the room and make
      // the grid labels disagree with reality. (A selfie-style game would want
      // the opposite.)
      if (video && video.readyState >= 2 && video.videoWidth) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      } else {
        ctx.fillStyle = '#07080d';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }
      // Darken slightly so the overlay reads clearly against a bright room.
      ctx.fillStyle = 'rgba(7,8,13,0.32)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    function polygon(pts) {
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (var i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.closePath();
    }

    function quadHeight(q) {
      return Math.max(Math.hypot(q[3][0] - q[0][0], q[3][1] - q[0][1]),
                      Math.hypot(q[2][0] - q[1][0], q[2][1] - q[1][1]));
    }

    /* Fit text inside a tile by measuring it and scaling the font down until
     * it fits. Far tiles are small on screen, so a fixed font size either
     * overflows there or looks tiny up close. */
    function fitText(text, maxW, maxH) {
      var MIN = 9;      // below this an answer is unreadable on a far tile
      var MAX = 34;
      var size = Math.min(Math.max(MIN, maxH * 0.42), MAX);
      for (var guard = 0; guard < 14; guard++) {
        ctx.font = 'bold ' + size.toFixed(1) + 'px ui-monospace, Menlo, monospace';
        if (ctx.measureText(text).width <= maxW) break;
        if (size <= MIN + 0.01) break;          // already at the floor
        // Clamp on the way down: shrinking past MIN would trade an overflowing
        // label for an illegible one, which is a worse outcome.
        size = Math.max(MIN, size * 0.88);
      }
      return size;
    }

    function drawTiles(calib, tiles, showLabels, phase) {
      var pulse = 0.5 + 0.5 * Math.sin(Date.now() / 220);
      for (var i = 0; i < tiles.length; i++) {
        var t = tiles[i];
        var q = calib.tileCanvasQuad(t.row, t.col, 0.045);
        if (!q) continue;
        var style = TILE_STYLE[t.state] || TILE_STYLE.empty;
        var fill = style.fill.slice();

        if (t.state === 'danger') fill[3] = 0.2 + 0.25 * pulse;
        if (t.flash > 0) fill[3] = Math.min(0.85, fill[3] + t.flash * 0.4);

        ctx.fillStyle = 'rgba(' + fill[0] + ',' + fill[1] + ',' + fill[2] + ',' + fill[3] + ')';
        polygon(q);
        ctx.fill();
        ctx.strokeStyle = rgba(style.stroke, style.stroke[3]);
        ctx.lineWidth = style.line;
        polygon(q);
        ctx.stroke();

        if (t.state === 'broken' || t.state === 'cracked') drawCracks(q, t.row * 7 + t.col);

        var h = quadHeight(q);
        var w = Math.hypot(q[1][0] - q[0][0], q[1][1] - q[0][1]);
        var c = calib.tileCenterCanvas(t.row, t.col);
        if (!c) continue;

        if (t.symbol) {
          var size = fitText(t.symbol, w * 0.86, h);
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.lineWidth = Math.max(2, size * 0.16);
          ctx.strokeStyle = 'rgba(5,8,14,0.85)';
          ctx.strokeText(t.symbol, c[0], c[1]);
          ctx.fillStyle = t.state === 'correct' ? '#e9fff2'
            : (t.state === 'cracked' ? 'rgba(220,180,190,0.75)' : '#ffffff');
          ctx.fillText(t.symbol, c[0], c[1]);
        } else if (showLabels && h > 16) {
          ctx.font = Math.min(13, h * 0.26).toFixed(1) + 'px ui-monospace, Menlo, monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = 'rgba(190,225,245,0.42)';
          ctx.fillText(calib.label(t.row, t.col), c[0], c[1]);
        }
      }
      ctx.textBaseline = 'alphabetic';
    }

    /* Deterministic crack pattern per tile, so a broken tile does not shimmer
     * with a new set of cracks every frame. */
    function drawCracks(q, seed) {
      var cx = (q[0][0] + q[1][0] + q[2][0] + q[3][0]) / 4;
      var cy = (q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4;
      var s = seed * 2654435761 % 1000;
      ctx.strokeStyle = 'rgba(255,170,120,0.5)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (var i = 0; i < 4; i++) {
        var a = ((s + i * 97) % 360) * Math.PI / 180;
        var corner = q[i];
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + (corner[0] - cx) * 0.9 + Math.cos(a) * 3,
                   cy + (corner[1] - cy) * 0.9 + Math.sin(a) * 3);
      }
      ctx.stroke();
    }

    function drawCalibrationOverlay(quad, activeIndex, calib) {
      var labels = ['1 TOP LEFT', '2 TOP RIGHT', '3 BOTTOM RIGHT', '4 BOTTOM LEFT'];
      if (quad.length > 1) {
        ctx.strokeStyle = 'rgba(255,196,40,0.9)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(quad[0][0], quad[0][1]);
        for (var i = 1; i < quad.length; i++) ctx.lineTo(quad[i][0], quad[i][1]);
        if (quad.length === 4) ctx.closePath();
        ctx.stroke();
      }
      for (var k = 0; k < quad.length; k++) {
        var p = quad[k];
        ctx.beginPath();
        ctx.arc(p[0], p[1], k === activeIndex ? 11 : 8, 0, Math.PI * 2);
        ctx.fillStyle = k === activeIndex ? 'rgba(255,255,255,0.95)' : 'rgba(255,196,40,0.9)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(7,8,13,0.9)';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 13px ui-monospace, Menlo, monospace';
        ctx.textAlign = 'left';
        ctx.strokeStyle = 'rgba(7,8,13,0.9)';
        ctx.lineWidth = 3;
        ctx.strokeText(labels[k], p[0] + 14, p[1] + 4);
        ctx.fillText(labels[k], p[0] + 14, p[1] + 4);
      }
      if (calib) drawGridPreview(calib);
    }

    function drawGridPreview(calib) {
      ctx.strokeStyle = 'rgba(0,229,255,0.4)';
      ctx.lineWidth = 1;
      for (var r = 0; r <= calib.rows; r++) {
        var a = calib.toCanvas(0, r / calib.rows), b = calib.toCanvas(1, r / calib.rows);
        if (a && b) { ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
      }
      for (var c = 0; c <= calib.cols; c++) {
        var d = calib.toCanvas(c / calib.cols, 0), e = calib.toCanvas(c / calib.cols, 1);
        if (d && e) { ctx.beginPath(); ctx.moveTo(d[0], d[1]); ctx.lineTo(e[0], e[1]); ctx.stroke(); }
      }
    }

    function drawSkeleton(player, frameW, frameH, color) {
      var lm = player.landmarks;
      if (!lm) return;
      var E = ns.tracking.SKELETON_EDGES;
      ctx.strokeStyle = rgba(color, 0.55);
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (var i = 0; i < E.length; i++) {
        var a = lm[E[i][0]], b = lm[E[i][1]];
        if (!a || !b) continue;
        var va = a.visibility === undefined ? 1 : a.visibility;
        var vb = b.visibility === undefined ? 1 : b.visibility;
        if (va < ns.tracking.MIN_VIS || vb < ns.tracking.MIN_VIS) continue;
        ctx.moveTo(a.x * frameW, a.y * frameH);
        ctx.lineTo(b.x * frameW, b.y * frameH);
      }
      ctx.stroke();

      // Emphasise the landmarks the game actually depends on.
      var L = ns.tracking.LM;
      var key = [L.LEFT_ANKLE, L.RIGHT_ANKLE, L.LEFT_HEEL, L.RIGHT_HEEL,
                 L.LEFT_FOOT_INDEX, L.RIGHT_FOOT_INDEX];
      ctx.fillStyle = rgba(color, 0.9);
      for (var k = 0; k < key.length; k++) {
        var p = lm[key[k]];
        if (!p) continue;
        var v = p.visibility === undefined ? 1 : p.visibility;
        if (v < ns.tracking.MIN_VIS) continue;
        ctx.beginPath();
        ctx.arc(p.x * frameW, p.y * frameH, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    /* The circular marker under a player's feet.
     *
     * Built from a circle sampled in floor coordinates and projected, so it
     * lies flat on the ground and squashes with distance -- the thing that
     * makes it read as painted on the floor rather than stuck to the screen.
     */
    function drawPlayerMarker(calib, player, view, showSkeleton, frameW, frameH) {
      var color = playerColor(player.id);
      if (showSkeleton) drawSkeleton(player, frameW, frameH, color);
      if (!player.floor) return;

      var out = view && view.status === 'OUT';
      var alpha = out ? 0.3 : 1;
      var pulse = 0.5 + 0.5 * Math.sin(Date.now() / 300 + player.id);

      var ring = calib.floorCirclePolygon(player.floor.x, player.floor.y, 0.055, 24);
      if (ring) {
        ctx.fillStyle = rgba(color, 0.18 * alpha);
        polygon(ring); ctx.fill();
        ctx.strokeStyle = rgba(color, (0.75 + 0.25 * pulse) * alpha);
        ctx.lineWidth = 3;
        polygon(ring); ctx.stroke();
      }
      var inner = calib.floorCirclePolygon(player.floor.x, player.floor.y, 0.022, 16);
      if (inner) {
        ctx.fillStyle = rgba(color, 0.55 * alpha);
        polygon(inner); ctx.fill();
      }

      // Tie the marker to the body so it is obvious who it belongs to.
      var lm = player.landmarks;
      var L = ns.tracking.LM;
      if (lm && player.pixel && lm[L.LEFT_HIP] && lm[L.RIGHT_HIP]) {
        var hx = (lm[L.LEFT_HIP].x + lm[L.RIGHT_HIP].x) / 2 * frameW;
        var hy = (lm[L.LEFT_HIP].y + lm[L.RIGHT_HIP].y) / 2 * frameH;
        ctx.strokeStyle = rgba(color, 0.3 * alpha);
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(player.pixel.x, player.pixel.y);
        ctx.lineTo(hx, hy);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Nameplate above the head.
      var head = lm && lm[L.NOSE] ? { x: lm[L.NOSE].x * frameW, y: lm[L.NOSE].y * frameH - 28 }
                                  : { x: player.pixel.x, y: player.pixel.y - 120 };
      var text = 'P' + player.id + '  ' + (view ? view.tileLabel : '--') +
                 (view ? '  ' + view.score : '');
      ctx.font = 'bold 14px ui-monospace, Menlo, monospace';
      var w = ctx.measureText(text).width + 14;
      ctx.fillStyle = 'rgba(7,8,13,' + (0.72 * alpha) + ')';
      ctx.fillRect(head.x - w / 2, head.y - 15, w, 21);
      ctx.strokeStyle = rgba(color, 0.8 * alpha);
      ctx.lineWidth = 1.5;
      ctx.strokeRect(head.x - w / 2, head.y - 15, w, 21);
      ctx.fillStyle = rgba(color, alpha);
      ctx.textAlign = 'center';
      ctx.fillText(text, head.x, head.y);

      if (view && view.lives < 3) {
        var hearts = '';
        for (var i = 0; i < 3; i++) hearts += i < view.lives ? '\u2665' : '\u2661';
        ctx.font = '13px ui-monospace, Menlo, monospace';
        ctx.fillStyle = 'rgba(255,90,90,' + alpha + ')';
        ctx.fillText(hearts, head.x, head.y + 18);
      }
    }

    function burst(x, y, color, count) {
      for (var i = 0; i < (count || 18); i++) {
        var a = Math.random() * Math.PI * 2;
        var sp = 40 + Math.random() * 180;
        particles.push({
          x: x, y: y,
          vx: Math.cos(a) * sp, vy: Math.sin(a) * sp - 60,
          life: 0.5 + Math.random() * 0.5, max: 1,
          color: color, size: 2 + Math.random() * 3
        });
      }
    }

    function stepParticles(dt) {
      for (var i = particles.length - 1; i >= 0; i--) {
        var p = particles[i];
        p.life -= dt;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.vy += 420 * dt;
        if (p.life <= 0) particles.splice(i, 1);
      }
    }

    function drawParticles() {
      for (var i = 0; i < particles.length; i++) {
        var p = particles[i];
        var a = Math.max(0, p.life / p.max);
        ctx.fillStyle = rgba(p.color, a);
        ctx.fillRect(p.x, p.y, p.size, p.size);
      }
    }

    function drawRoundTimer(state) {
      if (state.phase !== 'active' || !state.roundDuration) return;
      var frac = Math.max(0, state.timeLeft / state.roundDuration);
      var h = 8;
      ctx.fillStyle = 'rgba(7,8,13,0.6)';
      ctx.fillRect(0, 0, canvas.width, h);
      ctx.fillStyle = frac < 0.3 ? 'rgba(255,80,70,0.95)'
                                 : (state.levelType === 'question' ? 'rgba(0,229,255,0.9)'
                                                                   : 'rgba(255,196,40,0.9)');
      ctx.fillRect(0, 0, canvas.width * frac, h);
    }

    function drawBanner(lines, color) {
      if (!lines || !lines.length) return;
      ctx.textAlign = 'center';
      for (var i = 0; i < lines.length; i++) {
        var size = i === 0 ? 40 : 20;
        ctx.font = 'bold ' + size + 'px ui-monospace, Menlo, monospace';
        var y = canvas.height * 0.42 + i * (size + 12);
        ctx.lineWidth = 6;
        ctx.strokeStyle = 'rgba(7,8,13,0.8)';
        ctx.strokeText(lines[i], canvas.width / 2, y);
        ctx.fillStyle = i === 0 ? color : '#f5f8ff';
        ctx.fillText(lines[i], canvas.width / 2, y);
      }
    }

    return {
      ctx: ctx,
      clear: clear,
      drawVideo: drawVideo,
      drawTiles: drawTiles,
      drawGridPreview: drawGridPreview,
      drawCalibrationOverlay: drawCalibrationOverlay,
      drawPlayerMarker: drawPlayerMarker,
      drawRoundTimer: drawRoundTimer,
      drawBanner: drawBanner,
      burst: burst,
      stepParticles: stepParticles,
      drawParticles: drawParticles,
      playerColor: playerColor,
      get particleCount() { return particles.length; }
    };
  }

  ns.render = { createRenderer: createRenderer, playerColor: playerColor, PLAYER_COLORS: PLAYER_COLORS };
})(typeof globalThis !== 'undefined' ? globalThis : this);
