"""
Rock contour tuner — comic-book line art
=========================================
Loads the SAM-isolated rock image (produced by option5), detects edges at
every detail level — silhouette, cracks, texture, grain — and renders them
as black ink contours on a white canvas: a classic comic-book outline look.

Pipeline: blur → Canny edges → dilate (connect/thicken broken lines) →
find EVERY contour (not just the outer silhouette) → drop ones shorter than
MIN_LEN → draw the rest in black on white.

Sliders:
  BLUR_SIZE  — Gaussian pre-blur kernel (odd) — higher = fewer fine details
  CANNY_LOW  — Canny lower hysteresis threshold
  CANNY_HIGH — Canny upper hysteresis threshold
  DILATE_K   — dilation kernel (odd) — connects broken edges & thickens lines
  MIN_LEN    — minimum contour perimeter in px — discards speckle/noise

Close the window to print the final values.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

ISOLATION_PATH = r"D:\Machine_Learning\Stecci\Manually_Testing_Stuff\step2_output_combined_sam_rock_isolation.png"

INIT_BLUR       = 5
INIT_CANNY_LOW  = 50
INIT_CANNY_HIGH = 150
INIT_DILATE_K   = 1
INIT_MIN_LEN    = 20

LINE_COLOR     = (0, 0, 0)
LINE_THICKNESS = 1


def load_isolated_rgb():
    bgr = cv2.imread(ISOLATION_PATH)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {ISOLATION_PATH}\nRun option5 first to generate it.")
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def find_detail_contours(gray, blur_size, canny_low, canny_high, dilate_k, min_len):
    """
    Edge-detect at every detail level and keep every contour long enough to
    matter — the silhouette, cracks, texture lines, everything Canny picks
    up — for a comic-style "inked" rendering (as opposed to just the single
    outer rock boundary).
    """
    blur_size = int(blur_size) | 1
    dilate_k  = max(1, int(dilate_k)) | 1

    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    edges   = cv2.Canny(blurred, int(canny_low), int(canny_high))

    if dilate_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        edges  = cv2.dilate(edges, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.arcLength(c, True) >= min_len]


def draw_comic(shape, contours):
    """Render contours as black ink lines on a white canvas."""
    canvas = np.full(shape, 255, dtype=np.uint8)
    cv2.drawContours(canvas, contours, -1, LINE_COLOR, LINE_THICKNESS)
    return canvas


def main():
    bgr, rgb = load_isolated_rgb()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    canvas_shape = rgb.shape

    fig, ax = plt.subplots(figsize=(9, 8))
    plt.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.38)

    init_contours = find_detail_contours(
        gray, INIT_BLUR, INIT_CANNY_LOW, INIT_CANNY_HIGH, INIT_DILATE_K, INIT_MIN_LEN)
    im = ax.imshow(draw_comic(canvas_shape, init_contours))
    ax.set_title("Comic-style outline (all details)", fontsize=11)
    ax.axis("off")

    sup = fig.suptitle("", fontsize=11, fontweight="bold")

    def refresh_title(bs, cl, ch, dk, ml, n):
        sup.set_text(
            f"BLUR={bs}   CANNY=({cl}, {ch})   DILATE_K={dk}   "
            f"MIN_LEN={ml}px   →  {n} contour(s)"
        )

    refresh_title(INIT_BLUR, INIT_CANNY_LOW, INIT_CANNY_HIGH, INIT_DILATE_K, INIT_MIN_LEN, len(init_contours))

    ax_blur   = plt.axes([0.10, 0.28, 0.80, 0.03])
    ax_clow   = plt.axes([0.10, 0.22, 0.80, 0.03])
    ax_chigh  = plt.axes([0.10, 0.16, 0.80, 0.03])
    ax_dilate = plt.axes([0.10, 0.10, 0.80, 0.03])
    ax_minlen = plt.axes([0.10, 0.04, 0.80, 0.03])

    sl_blur   = Slider(ax_blur,   "BLUR_SIZE",    1,   31,  valinit=INIT_BLUR,       valstep=2, color="steelblue")
    sl_clow   = Slider(ax_clow,   "CANNY_LOW",    0,   255, valinit=INIT_CANNY_LOW,  valstep=1, color="mediumpurple")
    sl_chigh  = Slider(ax_chigh,  "CANNY_HIGH",   0,   400, valinit=INIT_CANNY_HIGH, valstep=1, color="mediumpurple")
    sl_dilate = Slider(ax_dilate, "DILATE_K",     1,   15,  valinit=INIT_DILATE_K,   valstep=2, color="steelblue")
    sl_minlen = Slider(ax_minlen, "MIN_LEN (px)", 0,   300, valinit=INIT_MIN_LEN,    valstep=5, color="steelblue")

    def current_params():
        return (
            int(sl_blur.val)   | 1,
            int(sl_clow.val),
            int(sl_chigh.val),
            int(sl_dilate.val) | 1,
            int(sl_minlen.val),
        )

    def update(_):
        bs, cl, ch, dk, ml = current_params()
        contours = find_detail_contours(gray, bs, cl, ch, dk, ml)
        im.set_data(draw_comic(canvas_shape, contours))
        refresh_title(bs, cl, ch, dk, ml, len(contours))
        fig.canvas.draw_idle()

    sl_blur.on_changed(update)
    sl_clow.on_changed(update)
    sl_chigh.on_changed(update)
    sl_dilate.on_changed(update)
    sl_minlen.on_changed(update)

    def on_close(_):
        bs, cl, ch, dk, ml = current_params()
        print("\n── Final parameter values ────────────────────────────")
        print(f"  BLUR_SIZE  = {bs}")
        print(f"  CANNY_LOW  = {cl}")
        print(f"  CANNY_HIGH = {ch}")
        print(f"  DILATE_K   = {dk}")
        print(f"  MIN_LEN    = {ml}px")
        print("──────────────────────────────────────────────────────\n")

    fig.canvas.mpl_connect("close_event", on_close)
    plt.show()


if __name__ == "__main__":
    main()
