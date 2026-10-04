/* Floor calibration geometry: camera pixels <-> floor grid.
 *
 * The camera looks at the floor from an angle, so a square tile is a
 * trapezoid on screen and tiles further away are smaller. A linear scale
 * cannot express that; a projective one can. Given the four corners of the
 * playable floor as clicked on the video, we solve for the homography that
 * maps that quadrilateral onto the unit square:
 *
 *     camera pixels --H--> floor [0,1] x [0,1] --> grid (row, col)
 *
 * and the inverse, which is what lets the renderer paint tiles that sit flat
 * on the real floor:
 *
 *     grid (row, col) --> floor [0,1]^2 --Hinv--> camera pixels
 *
 * Both directions are solved independently from the same four
 * correspondences rather than inverting one matrix. It is the same amount of
 * work, it avoids compounding the inverse's numerical error, and it gives the
 * tests a genuine consistency check: transform a point one way, then the
 * other, and it must land back where it started.
 *
 * Attaches to a global so Node can load this file for the tests without a
 * bundler or module loader.
 */
(function (root) {
  'use strict';

  var ns = root.CVFloor = root.CVFloor || {};

  /* Solve A x = b by Gauss-Jordan with partial pivoting.
   * Returns null when the system is singular -- which for calibration means
   * the user clicked a degenerate quad (three points in a line, say), and the
   * caller must refuse it rather than render nonsense. */
  function solveLinear(A, b) {
    var n = A.length;
    var m = [];
    for (var i = 0; i < n; i++) m.push(A[i].slice().concat([b[i]]));
    for (var col = 0; col < n; col++) {
      var pivot = col;
      for (var r = col + 1; r < n; r++) {
        if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
      }
      if (Math.abs(m[pivot][col]) < 1e-12) return null;
      var tmp = m[col]; m[col] = m[pivot]; m[pivot] = tmp;
      var pv = m[col][col];
      for (var c = col; c <= n; c++) m[col][c] /= pv;
      for (var r2 = 0; r2 < n; r2++) {
        if (r2 === col) continue;
        var f = m[r2][col];
        if (f === 0) continue;
        for (var c2 = col; c2 <= n; c2++) m[r2][c2] -= f * m[col][c2];
      }
    }
    var x = [];
    for (var k = 0; k < n; k++) x.push(m[k][n]);
    return x;
  }

  /* Homography taking src[i] -> dst[i] for exactly four point pairs.
   *
   * With h22 fixed at 1 there are 8 unknowns and each pair contributes two
   * equations, so this is an exact 8x8 solve -- no least squares needed.
   *
   * Source coordinates are shifted and scaled to be centred and
   * order-1 before solving. Raw pixel values in the hundreds (squared in the
   * projective terms) make the matrix badly conditioned, and the resulting
   * grid visibly bends away from the floor it is supposed to be glued to.
   */
  function homographyFrom4(src, dst) {
    if (!src || !dst || src.length !== 4 || dst.length !== 4) return null;

    var cx = 0, cy = 0;
    for (var i = 0; i < 4; i++) { cx += src[i][0]; cy += src[i][1]; }
    cx /= 4; cy /= 4;
    var scale = 0;
    for (var j = 0; j < 4; j++) {
      scale += Math.hypot(src[j][0] - cx, src[j][1] - cy);
    }
    scale = scale > 1e-9 ? (4 / scale) : 1;

    var A = [], b = [];
    for (var k = 0; k < 4; k++) {
      var x = (src[k][0] - cx) * scale;
      var y = (src[k][1] - cy) * scale;
      var u = dst[k][0], v = dst[k][1];
      A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
      b.push(u);
      A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
      b.push(v);
    }
    var h = solveLinear(A, b);
    if (!h) return null;
    for (var q = 0; q < 8; q++) if (!isFinite(h[q])) return null;

    // H_normalised, then fold the normalising similarity back in:
    // H = Hn * S, where S translates by -c and scales by `scale`.
    var Hn = [
      [h[0], h[1], h[2]],
      [h[3], h[4], h[5]],
      [h[6], h[7], 1]
    ];
    var S = [
      [scale, 0, -scale * cx],
      [0, scale, -scale * cy],
      [0, 0, 1]
    ];
    return matmul3(Hn, S);
  }

  function matmul3(a, b) {
    var out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (var i = 0; i < 3; i++) {
      for (var j = 0; j < 3; j++) {
        out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
      }
    }
    return out;
  }

  /* Map a point through a homography. Returns null behind the plane's horizon
   * (w <= 0), where the projection is meaningless. */
  function applyH(H, x, y) {
    var w = H[2][0] * x + H[2][1] * y + H[2][2];
    if (!isFinite(w) || Math.abs(w) < 1e-12) return null;
    return [
      (H[0][0] * x + H[0][1] * y + H[0][2]) / w,
      (H[1][0] * x + H[1][1] * y + H[1][2]) / w
    ];
  }

  var UNIT_SQUARE = [[0, 0], [1, 0], [1, 1], [0, 1]];

  function quadIsSane(quad) {
    if (!quad || quad.length !== 4) return false;
    for (var i = 0; i < 4; i++) {
      if (!quad[i] || !isFinite(quad[i][0]) || !isFinite(quad[i][1])) return false;
    }
    // Reject a self-intersecting or collapsed quad: the signed area must be
    // non-trivial and the winding consistent.
    var area = 0;
    for (var k = 0; k < 4; k++) {
      var p = quad[k], q = quad[(k + 1) % 4];
      area += p[0] * q[1] - q[0] * p[1];
    }
    // A play area smaller than roughly 50x50 px cannot hold a readable grid,
    // and almost always means a misclick rather than a real calibration.
    if (Math.abs(area / 2) < 2500) return false;
    var sign = 0;
    for (var m = 0; m < 4; m++) {
      var a = quad[m], bb = quad[(m + 1) % 4], c = quad[(m + 2) % 4];
      var cross = (bb[0] - a[0]) * (c[1] - bb[1]) - (bb[1] - a[1]) * (c[0] - bb[0]);
      if (Math.abs(cross) < 1e-9) continue;
      var s = cross > 0 ? 1 : -1;
      if (sign === 0) sign = s;
      else if (s !== sign) return false;          // concave / crossed
    }
    return true;
  }

  /* Column letters go A, B, ... Z, AA, AB ... so an oversized grid still gets
   * unique labels instead of silently colliding. */
  function rowLetter(row) {
    var s = '';
    var n = row;
    do {
      s = String.fromCharCode(65 + (n % 26)) + s;
      n = Math.floor(n / 26) - 1;
    } while (n >= 0);
    return s;
  }

  /* The calibration object the rest of the app talks to.
   *
   * `quad` is [topLeft, topRight, bottomRight, bottomLeft] in *camera pixel*
   * coordinates. Floor coordinates run x: 0 (left) -> 1 (right) and
   * y: 0 (far edge) -> 1 (near edge), so row 0 is the far row -- which is what
   * a person watching the screen expects row "A" to be.
   */
  function createCalibration(quad, cols, rows) {
    cols = cols || 8;
    rows = rows || 8;
    if (!quadIsSane(quad)) return null;
    var H = homographyFrom4(quad, UNIT_SQUARE);          // pixels -> floor
    var Hinv = homographyFrom4(UNIT_SQUARE, quad);       // floor -> pixels
    if (!H || !Hinv) return null;

    function toFloor(px, py) { return applyH(H, px, py); }
    function toCanvas(fx, fy) { return applyH(Hinv, fx, fy); }

    /* Which tile is this floor point in? Returns null when the point lies
     * outside the calibrated area -- a player standing off the playfield has
     * no tile, and inventing one would make the game lie. */
    function tileAtFloor(fx, fy) {
      if (fx < 0 || fx >= 1 || fy < 0 || fy >= 1) return null;
      var col = Math.floor(fx * cols);
      var row = Math.floor(fy * rows);
      if (col < 0) col = 0; else if (col >= cols) col = cols - 1;
      if (row < 0) row = 0; else if (row >= rows) row = rows - 1;
      return { row: row, col: col };
    }

    function tileAtPixel(px, py) {
      var f = toFloor(px, py);
      return f ? tileAtFloor(f[0], f[1]) : null;
    }

    /* How far inside its tile a point sits, as a fraction of the tile where
     * 0 is exactly on an edge and 0.5 is dead centre. The tile-change
     * hysteresis is built on this. */
    function insetInTile(fx, fy) {
      var tx = fx * cols, ty = fy * rows;
      var dx = Math.min(tx - Math.floor(tx), 1 - (tx - Math.floor(tx)));
      var dy = Math.min(ty - Math.floor(ty), 1 - (ty - Math.floor(ty)));
      return Math.min(dx, dy);
    }

    function tileFloorQuad(row, col, shrink) {
      var s = shrink || 0;
      var x0 = (col + s) / cols, x1 = (col + 1 - s) / cols;
      var y0 = (row + s) / rows, y1 = (row + 1 - s) / rows;
      return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    }

    /* Tile outline in camera pixels: this is what makes the virtual grid look
     * glued to the real floor. */
    function tileCanvasQuad(row, col, shrink) {
      var fq = tileFloorQuad(row, col, shrink);
      var out = [];
      for (var i = 0; i < 4; i++) {
        var p = toCanvas(fq[i][0], fq[i][1]);
        if (!p) return null;
        out.push(p);
      }
      return out;
    }

    function tileCenterFloor(row, col) {
      return [(col + 0.5) / cols, (row + 0.5) / rows];
    }

    function tileCenterCanvas(row, col) {
      var c = tileCenterFloor(row, col);
      return toCanvas(c[0], c[1]);
    }

    /* A circle on the *floor* becomes an ellipse on screen. Rather than guess
     * at the ellipse, sample the circle in floor space and push each sample
     * through the homography -- the perspective then comes out right for free,
     * including the near edge being larger than the far edge. */
    function floorCirclePolygon(fx, fy, radiusFloor, segments) {
      var n = segments || 20;
      var pts = [];
      for (var i = 0; i < n; i++) {
        var a = (i / n) * Math.PI * 2;
        var p = toCanvas(fx + Math.cos(a) * radiusFloor, fy + Math.sin(a) * radiusFloor);
        if (p) pts.push(p);
      }
      return pts.length >= 3 ? pts : null;
    }

    function label(row, col) { return rowLetter(row) + String(col + 1); }

    return {
      quad: quad.map(function (p) { return [p[0], p[1]]; }),
      cols: cols,
      rows: rows,
      H: H,
      Hinv: Hinv,
      toFloor: toFloor,
      toCanvas: toCanvas,
      tileAtFloor: tileAtFloor,
      tileAtPixel: tileAtPixel,
      insetInTile: insetInTile,
      tileFloorQuad: tileFloorQuad,
      tileCanvasQuad: tileCanvasQuad,
      tileCenterFloor: tileCenterFloor,
      tileCenterCanvas: tileCenterCanvas,
      floorCirclePolygon: floorCirclePolygon,
      label: label
    };
  }

  ns.geometry = {
    solveLinear: solveLinear,
    homographyFrom4: homographyFrom4,
    applyH: applyH,
    matmul3: matmul3,
    quadIsSane: quadIsSane,
    rowLetter: rowLetter,
    createCalibration: createCalibration,
    UNIT_SQUARE: UNIT_SQUARE
  };
})(typeof globalThis !== 'undefined' ? globalThis : this);
