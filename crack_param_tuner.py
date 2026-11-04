"""
Crack detection parameter tuner  (rock-masked)
===============================================
Loads the isolated rock image produced by option5 (or runs GDINO+SAM
automatically if those files don't exist yet).  Crack detection runs only
inside the rock bounding box, tiled for local Otsu thresholding, and the
result is ANDed with the rock mask so grass/background is excluded.

Sliders:
  Tile grid (N×N)   — 1 = whole rock box, 2 = 2×2, 3 = 3×3, …
  CRACK_KERNEL_SIZE — black-hat morphology kernel (odd)
  CRACK_BLUR_SIZE   — Gaussian pre-blur kernel (odd)
  CRACK_CLOSE_K     — closing kernel to bridge gaps
  CRACK_MIN_AREA    — minimum crack blob area (px²)

Close the window to print the final values.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch

IMAGE_PATH = "stecci_1_raw.jpeg"

# Saved outputs from option5 (loaded automatically if present)
ISOLATION_RGB_PATH  = "step2_output_combined_sam_rock_isolation.png"
ROCK_MASK_PATH      = "step2_output_combined_sam_rock_mask.png"

# option5 SAM / DINO settings (only used when running the pipeline from scratch)
SAM_CHECKPOINT  = r"D:\Machine_Learning\Stecci\App_v0\model_ckpts\sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE  = "vit_h"
ROCK_PROMPT     = "rock"
BOX_THRESHOLD   = 0.25
TEXT_THRESHOLD  = 0.20
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

INIT_TILE_N      = 6 # initial tile grid (N×N) — 1 = whole rock box, 2 = 2×2, 3 = 3×3, …
INIT_KERNEL_SIZE = 13
INIT_BLUR_SIZE   = 41
INIT_CLOSE_K     = 1
INIT_MIN_AREA    = 750
TILE_OVERLAP     = 0.2  # fraction of tile size to overlap (0.0–0.5) — helps continuity across tile borders


# ── Rock isolation ─────────────────────────────────────────────────────────────

def load_rock_isolation():
    """
    Try loading pre-saved files from option5.  If missing, run GDINO+SAM once.
    Returns (image_rgb, isolated_rgb, rock_mask bool H×W, rock_box int[4]).
    """
    import os
    from PIL import Image as PILImage

    bgr = cv2.imread(IMAGE_PATH)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {IMAGE_PATH}")
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if os.path.exists(ISOLATION_RGB_PATH) and os.path.exists(ROCK_MASK_PATH):
        print("Loading pre-saved isolation outputs …")
        iso_bgr      = cv2.imread(ISOLATION_RGB_PATH)
        mask_gray    = cv2.imread(ROCK_MASK_PATH, cv2.IMREAD_GRAYSCALE)
        isolated_rgb = cv2.cvtColor(iso_bgr, cv2.COLOR_BGR2RGB)
        rock_mask    = mask_gray > 127
    else:
        print("Pre-saved isolation files not found — running GDINO + SAM …")
        isolated_rgb, rock_mask = _run_gdino_sam(image_rgb)

    # Bounding box of the rock mask
    ys, xs   = np.where(rock_mask)
    rock_box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
    print(f"Rock box: {rock_box}   mask px: {rock_mask.sum():,}")
    return image_rgb, isolated_rgb, rock_mask, rock_box


def _run_gdino_sam(image_rgb: np.ndarray):
    """Run Grounding DINO + SAM isolation pipeline inline."""
    from PIL import Image as PILImage
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from segment_anything import sam_model_registry, SamPredictor

    image_pil = PILImage.fromarray(image_rgb)
    h, w      = image_rgb.shape[:2]

    # GDINO
    print("  Loading Grounding DINO …")
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    gdino     = AutoModelForZeroShotObjectDetection.from_pretrained(
                    "IDEA-Research/grounding-dino-base").to(DEVICE)
    gdino.eval()

    hf_prompt = ROCK_PROMPT.rstrip(".") + "."
    inputs    = processor(images=image_pil, text=hf_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = gdino(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
        target_sizes=[(h, w)])[0]

    boxes = results["boxes"].cpu().numpy()
    if len(boxes) == 0:
        raise RuntimeError("Grounding DINO found no rock — lower BOX_THRESHOLD.")

    # Pick box closest to image centre
    cx, cy   = w / 2, h / 2
    centres  = boxes[:, :2] / 2 + boxes[:, 2:] / 2
    dists    = np.linalg.norm(centres - np.array([cx, cy]), axis=1)
    rock_box = boxes[dists.argmin()]

    # SAM
    print("  Loading SAM …")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.eval().to(DEVICE)
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    box_t = torch.tensor(rock_box, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    box_t = predictor.transform.apply_boxes_torch(box_t, image_rgb.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None, point_labels=None,
        boxes=box_t, multimask_output=False)
    rock_mask = masks[0, 0].cpu().numpy().astype(bool)

    isolated_rgb           = image_rgb.copy()
    isolated_rgb[~rock_mask] = 0
    return isolated_rgb, rock_mask


# ── Crack detection ────────────────────────────────────────────────────────────

def _crack_single_tile(gray_tile: np.ndarray,
                        kernel_size: int, blur_size: int,
                        close_k: int, min_area: int) -> np.ndarray:
    blurred  = cv2.GaussianBlur(gray_tile, (blur_size, blur_size), 0)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
    _, mask  = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if close_k > 1:
        kc   = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = [c for c in contours if cv2.contourArea(c) >= min_area]
    out  = np.zeros_like(mask)
    cv2.drawContours(out, kept, -1, 255, -1)
    return out


def detect_cracks_tiled(isolated_gray: np.ndarray,
                          rock_mask: np.ndarray,
                          rock_box: np.ndarray,
                          kernel_size: int, blur_size: int,
                          close_k: int, min_area: int,
                          tile_n: int) -> tuple[np.ndarray, int]:
    """
    Tile the rock bounding box, detect cracks per tile, stitch back to full
    image coords, then AND with rock_mask to strip non-rock pixels.
    """
    kernel_size = int(kernel_size) | 1
    blur_size   = int(blur_size)   | 1
    close_k     = max(1, int(close_k))

    bx1, by1, bx2, by2 = map(int, rock_box)
    box_gray = isolated_gray[by1:by2, bx1:bx2]
    bh, bw   = box_gray.shape

    box_mask = np.zeros((bh, bw), dtype=np.uint8)
    th, tw   = bh / tile_n, bw / tile_n
    pad_h, pad_w = int(th * TILE_OVERLAP), int(tw * TILE_OVERLAP)

    for row in range(tile_n):
        for col in range(tile_n):
            y1 = int(row * th);  y2 = int((row + 1) * th) if row < tile_n - 1 else bh
            x1 = int(col * tw);  x2 = int((col + 1) * tw) if col < tile_n - 1 else bw

            y1e = max(0, y1 - pad_h);  y2e = min(bh, y2 + pad_h)
            x1e = max(0, x1 - pad_w);  x2e = min(bw, x2 + pad_w)

            tile     = box_gray[y1e:y2e, x1e:x2e]
            tile_out = _crack_single_tile(tile, kernel_size, blur_size, close_k, min_area)

            dy1, dx1 = y1 - y1e, x1 - x1e
            box_mask[y1:y2, x1:x2] = np.maximum(
                box_mask[y1:y2, x1:x2],
                tile_out[dy1:dy1 + (y2 - y1), dx1:dx1 + (x2 - x1)])

    # Remap to full image and restrict to rock mask
    h_full, w_full = isolated_gray.shape
    full_mask      = np.zeros((h_full, w_full), dtype=np.uint8)
    full_mask[by1:by2, bx1:bx2] = box_mask
    full_mask = cv2.bitwise_and(full_mask, rock_mask.astype(np.uint8) * 255)

    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return full_mask, len(contours)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    image_rgb, isolated_rgb, rock_mask, rock_box = load_rock_isolation()
    isolated_gray = cv2.cvtColor(
        cv2.cvtColor(isolated_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)

    def make_overlay(crack_mask: np.ndarray) -> np.ndarray:
        out = image_rgb.astype(np.float32).copy()
        out[crack_mask > 0] = (out[crack_mask > 0] * 0.35
                               + np.array([220, 60, 60], np.float32) * 0.65)
        return np.clip(out, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(11, 8))
    plt.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.38)

    init_mask, init_n = detect_cracks_tiled(
        isolated_gray, rock_mask, rock_box,
        INIT_KERNEL_SIZE, INIT_BLUR_SIZE, INIT_CLOSE_K, INIT_MIN_AREA, INIT_TILE_N)

    im = ax.imshow(make_overlay(init_mask))
    ax.set_title("Overlay  (red = crack, rock-masked)", fontsize=11)
    ax.axis("off")

    sup = fig.suptitle("", fontsize=11, fontweight="bold")

    def refresh_title(tn, ks, bs, ck, ma, n):
        sup.set_text(
            f"Tile grid={tn}×{tn}   KERNEL={ks}   BLUR={bs}   "
            f"CLOSE_K={ck}   MIN_AREA={ma}   →  {n} region(s)"
        )

    refresh_title(INIT_TILE_N, INIT_KERNEL_SIZE, INIT_BLUR_SIZE,
                  INIT_CLOSE_K, INIT_MIN_AREA, init_n)

    ax_tile   = plt.axes([0.10, 0.28, 0.80, 0.03])
    ax_kernel = plt.axes([0.10, 0.22, 0.80, 0.03])
    ax_blur   = plt.axes([0.10, 0.16, 0.80, 0.03])
    ax_close  = plt.axes([0.10, 0.10, 0.80, 0.03])
    ax_area   = plt.axes([0.10, 0.04, 0.80, 0.03])

    sl_tile   = Slider(ax_tile,   "Tile grid (N×N)",      1,    6,    valinit=INIT_TILE_N,      valstep=1,  color="mediumpurple")
    sl_kernel = Slider(ax_kernel, "CRACK_KERNEL_SIZE",     3,    51,   valinit=INIT_KERNEL_SIZE, valstep=2,  color="steelblue")
    sl_blur   = Slider(ax_blur,   "CRACK_BLUR_SIZE",       3,    101,  valinit=INIT_BLUR_SIZE,   valstep=2,  color="steelblue")
    sl_close  = Slider(ax_close,  "CRACK_CLOSE_K",         1,    25,   valinit=INIT_CLOSE_K,     valstep=1,  color="steelblue")
    sl_area   = Slider(ax_area,   "CRACK_MIN_AREA (px²)",  0,    5000, valinit=INIT_MIN_AREA,    valstep=50, color="steelblue")

    def update(_):
        tn = int(sl_tile.val)
        ks = int(sl_kernel.val) | 1
        bs = int(sl_blur.val)   | 1
        ck = int(sl_close.val)
        ma = int(sl_area.val)
        mask, n = detect_cracks_tiled(isolated_gray, rock_mask, rock_box,
                                       ks, bs, ck, ma, tn)
        im.set_data(make_overlay(mask))
        refresh_title(tn, ks, bs, ck, ma, n)
        fig.canvas.draw_idle()

    sl_tile.on_changed(update)
    sl_kernel.on_changed(update)
    sl_blur.on_changed(update)
    sl_close.on_changed(update)
    sl_area.on_changed(update)

    def on_close(_):
        tn = int(sl_tile.val)
        ks = int(sl_kernel.val) | 1
        bs = int(sl_blur.val)   | 1
        ck = int(sl_close.val)
        ma = int(sl_area.val)
        print("\n── Final parameter values ────────────────────────────")
        print(f"  TILE_N            = {tn}  ({tn}×{tn} grid)")
        print(f"  CRACK_KERNEL_SIZE = {ks}")
        print(f"  CRACK_BLUR_SIZE   = {bs}")
        print(f"  CRACK_CLOSE_K     = {ck}")
        print(f"  CRACK_MIN_AREA    = {ma}")
        print("──────────────────────────────────────────────────────\n")

    fig.canvas.mpl_connect("close_event", on_close)
    plt.show()


if __name__ == "__main__":
    main()
