"""A faithful Python port of the Syzygy integer path.

Ported from SuperInstance/Syzygy's C headers so that what is measured here is Syzygy and
not an approximation of it. The constants are copied exactly and every operation is an
integer operation, matching the C:

    syz_luma8          (77*R + 150*G + 29*B) >> 8          (BT.601, weights sum 256)
    syz_braille_bit    left col rows 0,1,2,3 -> bits 0,1,2,6 ; right -> 3,4,5,7
                       a subpixel lights when luma STRICTLY exceeds the threshold
    SYZ_STE_BASIS      the 4x4 tone basis
    tone index         ((luma * 9) / 255) + ste_select

If this port disagrees with the C, the disagreement is the finding -- not a nuisance.
"""
from __future__ import annotations

LUMA_WR = (77, 150, 29)                 # BT.601, sum exactly 256
RAMP = " .:-=+*#%@"                    # 9 density characters
BRAILLE_LEFT = (0, 1, 2, 6)
BRAILLE_RIGHT = (3, 4, 5, 7)
STE_BASIS = ((32, 64, 64, 32), (64, -32, -32, 64),
             (16, 96, 96, 16), (-64, 12, 12, -64))


def luma8(r: int, g: int, b: int) -> int:
    return ((r * LUMA_WR[0] + g * LUMA_WR[1] + b * LUMA_WR[2]) >> 8) & 0xFF


def braille_bit(row: int, col: int) -> int:
    return (BRAILLE_LEFT if col == 0 else BRAILLE_RIGHT)[row & 3]


def braille_pack(block, threshold: int) -> int:
    """block is 4 rows x 2 cols of luma. Strictly-greater, as in the C."""
    m = 0
    for row in range(4):
        for col in range(2):
            if block[row][col] > threshold:
                m |= 1 << braille_bit(row, col)
    return m


def braille_char(mask: int) -> str:
    return chr(0x2800 + (mask & 0xFF))


def ste_select4(feat) -> int:
    """Argmax over the 4 basis rows, first maximum wins on a strict '>' scan."""
    best_k, best_v = 0, None
    for k in range(4):
        row = STE_BASIS[k]
        v = sum(feat[i] * row[i] for i in range(4))
        if best_v is None or v > best_v:
            best_v, best_k = v, k
    return best_k


def tone(l: int, g: int = 0) -> str:
    idx = ((l * 9) // 255) + ste_select4((l, g, 0, 0))
    return RAMP[9 if idx > 9 else idx]


def sobel(block3) -> int:
    """3x3 Sobel magnitude, integer. Kept separate so a grid border is not an issue."""
    k = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    gx = gy = 0
    for r in range(3):
        for c in range(3):
            gx += block3[r][c] * k[r][c]
            gy += block3[r][c] * k[c][r]
    return abs(gx) + abs(gy)


def render_text(frame, cols, rows, threshold=110):
    """One fused pass: frame is rows x cols of (r,g,b). Returns the text rows.

    Cells are 2 px wide by 4 px tall, exactly the Braille dot layout, so a frame of
    cols x rows pixels becomes (cols//2) x (rows//4) characters.
    """
    lum = [[luma8(*frame[y][x]) for x in range(cols)] for y in range(rows)]
    out = []
    for cy in range(rows // 4):
        line = []
        for cx in range(cols // 2):
            blk = [[lum[cy * 4 + r][cx * 2 + c] for c in range(2)] for r in range(4)]
            line.append(braille_char(braille_pack(blk, threshold)))
        out.append("".join(line))
    return "\n".join(out)
