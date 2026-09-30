/* The Syzygy integer path, ported to JavaScript.
 *
 * Copied exactly from SuperInstance/Syzygy's C headers. Every operation is an integer
 * operation with no floating point, matching the C, so what the page shows is what the
 * engine computes -- not an approximation of it.
 *
 *   syz_luma8         (77*R + 150*G + 29*B) >> 8     BT.601, weights sum exactly 256
 *   syz_braille_bit   left col rows 0,1,2,3 -> bits 0,1,2,6 ; right -> 3,4,5,7
 *                      a subpixel lights when luma STRICTLY exceeds the threshold
 *   SYZ_STE_BASIS     the 4x4 tone basis
 */
const LUMA_WR = [77, 150, 29];
const BRAILLE_LEFT  = [0, 1, 2, 6];
const BRAILLE_RIGHT = [3, 4, 5, 7];
const STE_BASIS = [[32, 64, 64, 32], [64, -32, -32, 64],
                   [16, 96, 96, 16], [-64, 12, 12, -64]];

export const luma8 = (r, g, b) => ((r * LUMA_WR[0] + g * LUMA_WR[1] + b * LUMA_WR[2]) >> 8) & 0xFF;
export const brailleBit = (row, col) => (col === 0 ? BRAILLE_LEFT : BRAILLE_RIGHT)[row & 3];

export function braillePack(block, threshold) {
  let m = 0;
  for (let row = 0; row < 4; row++)
    for (let col = 0; col < 2; col++)
      if (block[row][col] > threshold) m |= 1 << brailleBit(row, col);
  return m;
}
export const brailleChar = (mask) => String.fromCodePoint(0x2800 + (mask & 0xff));

export function steSelect4(feat) {
  let bestK = 0, bestV = null;
  for (let k = 0; k < 4; k++) {
    let v = 0;
    for (let i = 0; i < 4; i++) v += feat[i] * STE_BASIS[k][i];
    if (bestV === null || v > bestV) { bestV = v; bestK = k; }
  }
  return bestK;
}
const RAMP = " .:-=+*#%@";
export const tone = (l, g = 0) => {
  const idx = Math.floor((l * 9) / 255) + steSelect4([l, g, 0, 0]);
  return RAMP[idx > 9 ? 9 : idx];
};

/** One fused pass. frame is rows x cols of [r,g,b]; returns { text, masks, lumas }. */
export function renderText(frame, cols, rows, threshold = 110) {
  const lum = [];
  for (let y = 0; y < rows; y++) {
    lum.push([]);
    for (let x = 0; x < cols; x++) lum[y].push(luma8(...frame[y][x]));
  }
  const lines = [];
  const masks = [];
  for (let cy = 0; cy < Math.floor(rows / 4); cy++) {
    let line = "", rowMasks = [];
    for (let cx = 0; cx < Math.floor(cols / 2); cx++) {
      const blk = [];
      for (let r = 0; r < 4; r++) blk.push([lum[cy * 4 + r][cx * 2], lum[cy * 4 + r][cx * 2 + 1]]);
      const m = braillePack(blk, threshold);
      rowMasks.push(m);
      line += brailleChar(m);
    }
    lines.push(line);
    masks.push(rowMasks);
  }
  return { text: lines.join("\n"), masks, lum };
}
