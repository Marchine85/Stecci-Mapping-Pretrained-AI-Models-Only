"""
Option 5: Grounding DINO → SAM isolation → Classic CV cracks → SAM auto-seg
============================================================================
Pipeline:
  1. Grounding DINO (HF) finds the main rock structure in the centre of the image.
     The detection closest to the image centre is kept as the rock bounding box.
  2. SAM (SamPredictor + DINO box) produces a single precise rock mask.
     Everything outside the rock is zeroed out.  Saves: isolation PNG + mask PNG.
  2b. Comic-style outline: Canny edges + contour tracing on the isolated rock
      pick out every detail (silhouette, cracks, texture, grain) and render
      them as black ink lines on white — see rock_contour_tuner.py to tune
      the COMIC_* parameters interactively.  Saves: outline PNG.
  3. Classic CV (black-hat + Otsu) detects cracks on the isolated rock gray image.
     The result is ANDed with the rock mask to strip border artefacts.
     Saves: binary crack mask PNG + overlay PNG + one PNG per crack region.
  4. SAM (SamAutomaticMaskGenerator) auto-segments the masked rock crop into up to
     SAM_MAX_SEGMENTS parts.  Masks that mostly fall outside the rock are discarded.
     4b. CLIP classifies each segment (currently just "biological colonization").
     Saves: overlay PNG + individual-mask grid PNG + bio-colonization overlay PNG.
  5. Final overlay: every layer (segments, bio colonization, cracks, borders)
     flattened onto the isolated rock at fixed default opacities.  Saves: one PNG.
  6. Interactive figure: same layers as step 5, with sliders/radio buttons to
     adjust opacities and background live.  Not saved — closing the window exits.

Requirements:
    pip install torch torchvision transformers
    pip install segment-anything
    pip install opencv-python numpy matplotlib Pillow
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, RadioButtons
from pathlib import Path
from PIL import Image

import torch
import open_clip
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from skimage import measure

# Reuse the Frangi multi-scale crack pipeline from crack_detection.py so the
# "frangi_multiscale" CRACK_METHOD stays in sync with that script/tuner.
from crack_detection import (
    preprocess as crack_preprocess,
    suppress_shadows as crack_suppress_shadows,
    detect_cracks_multiscale,
    segment_cracks as crack_segment,
    filter_by_shape as crack_filter_by_shape,
    classify_size,
)

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_PATH  = "stecci_1_raw.jpeg" #"stecci_1_raw.jpeg"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

SAM_MODEL = "vit_h"  # "vit_b" | "vit_l" | "vit_h"

if SAM_MODEL == "vit_b":
    SAM_CHECKPOINT = r"sam_model_ckpts\sam_vit_b_01ec64.pth"
    SAM_MODEL_TYPE = "vit_b"
elif SAM_MODEL == "vit_l":
    SAM_CHECKPOINT = r"sam_model_ckpts\sam_vit_l_0b3195.pth"
    SAM_MODEL_TYPE = "vit_l"
elif SAM_MODEL == "vit_h":
    SAM_CHECKPOINT = r"sam_model_ckpts\sam_vit_h_4b8939.pth"
    SAM_MODEL_TYPE = "vit_h"
else:
    raise ValueError(f"Unknown SAM_MODEL: {SAM_MODEL!r} " 
                     f"(expected 'vit_b', 'vit_l', or 'vit_h')")

# SAM auto-segmentation tuning
SAM_MAX_SEGMENTS        = -1    # keep only the largest N masks (-1 = keep all)
SAM_POINTS_PER_SIDE     = 32   # higher = more candidate points
SAM_PRED_IOU_THRESH     = 0.1 # 0.88 is a good starting point for cleaner masks; lower to get more masks with looser fit
SAM_STABILITY_THRESH    = 0.1 # 0.90 is a good starting point for stable masks; lower = more masks, including less stable ones
SAM_MIN_MASK_AREA       = 10000  # px² — discard tiny fragments
SAM_ROCK_OVERLAP_THRESH = 0.5  # fraction of a mask that must lie inside the rock

# CLIP classification
CLIP_MODEL      = "ViT-B-32"
CLIP_PRETRAINED = "openai"
MIN_CONFIDENCE  = 0.10   # masks below this confidence fall back to "rock_surface"
CONTEXT_PADDING = 10     # extra pixels added around each mask bbox for CLIP crop

# Categories: name → (R, G, B) display color + list of text prompts for CLIP
CATEGORIES = {
    #"rock_surface":            ((160, 140, 120), ["bare rock surface", "stone texture", "grey rock"]),
    #"crack":                   (( 40,  40,  40), ["a crack in rock", "dark fissure", "linear fracture in stone"]),
    #"pitting":                 ((180, 100, 200), ["surface pitting on rock", "small holes in stone", "erosion pits"]),
    "biological colonization": (( 80, 180,  80), ["green biological ", "moss", "grass", "algae"]),
    #"lichen":                  (( 80, 180,  80), ["lichen on rock", "green biological growth on stone", "algae on rock", "green biological growth on stone"]),
    #"white_encrustation":      ((240, 240, 240), ["white mineral deposit on rock", "salt crust", "white encrustation"]),
}

# Grounding DINO — prompt to isolate the central rock structure
ROCK_PROMPT         = "rock"
ROCK_BOX_THRESHOLD  = 0.25
ROCK_TEXT_THRESHOLD = 0.20

# Crack detection — pick a method:
#   "tiled_blackhat"    – local-Otsu black-hat per tile (see crack_param_tuner.py);
#                         CRACK_TILE_N>1 adapts to lighting that varies across a large rock face
#   "frangi_multiscale" – multi-scale Frangi ridge filter + shape filtering
#                         (see crack_detection.py); favours long, thin, elongated cracks
CRACK_METHOD = "tiled_blackhat"   # "tiled_blackhat" | "frangi_multiscale"

# tiled_blackhat parameters
CRACK_TILE_N       = 6     # tile grid N×N over the rock bounding box (1 = whole box, like before)
CRACK_TILE_OVERLAP = 0.2   # fractional tile overlap (0.0–0.5) — improves continuity across tile borders
CRACK_KERNEL_SIZE  = 11    # black-hat kernel (must be odd)
CRACK_BLUR_SIZE    = 41    # pre-blur to suppress micro-texture
CRACK_CLOSE_K      = 1     # closing to bridge gaps
CRACK_MIN_AREA     = 500   # discard blobs smaller than this (px²)

# frangi_multiscale parameters — tune interactively in crack_detection.py
FRANGI_SIGMAS     = np.arange(1.0, 7.1, 1.0)  # sigma ≈ half the crack width
FRANGI_MIN_ASPECT = 2.5    # major/minor axis ratio — higher = stricter elongation requirement
FRANGI_MIN_LENGTH = 100    # minimum crack length in px
FRANGI_MIN_AREA   = 1000    # minimum crack area in px²

# Comic-style contour outline parameters — tune interactively in rock_contour_tuner.py
# (run it on the same isolation PNG, then copy the values it prints back here)
COMIC_BLUR           = 5     # Gaussian pre-blur kernel (odd)
COMIC_CANNY_LOW      = 50    # Canny lower hysteresis threshold
COMIC_CANNY_HIGH     = 150   # Canny upper hysteresis threshold
COMIC_DILATE_K       = 1     # dilation kernel (odd) — connects/thickens lines
COMIC_MIN_LEN        = 20    # minimum contour perimeter in px — discards speckle/noise
COMIC_LINE_COLOR     = (0, 0, 0)
COMIC_LINE_THICKNESS = 1

# Output paths
OUT_ROCK_BOX             = "step1_rock_box.png"
OUT_SAM_ISOLATION_FIGURE = "step2_sam_rock_isolation_figure.png"
OUT_SAM_ISOLATION        = "step2_sam_rock_isolation.png"
OUT_SAM_ROCK_MASK        = "step2_sam_rock_mask.png"
OUT_COMIC_OUTLINE        = "step2b_comic_outline.png"
OUT_CRACK_MASK           = "step3_crack_mask.png"
OUT_CRACK_OVERLAY        = "step3_crack_overlay.png"
OUT_CRACK_SIZE_OVERLAY   = "step3_crack_size_overlay.png"
OUT_SAM_ALL_SEGMENTS     = "step4a_sam_all_segments.png"
OUT_SAM_OVERLAY          = "step4b_sam_overlay.png"
OUT_BIO_OVERLAY          = "step4b_bio_colonization_overlay.png"
OUT_FINAL_OVERLAY        = "step5_final_overlay.png"
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 – Grounding DINO: find the central rock box
# ══════════════════════════════════════════════════════════════════════════════

def load_gdino_hf(device: str):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    model_id  = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model     = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return model, processor


def detect_rock_box(model, processor, image_pil: Image.Image,
                    device: str) -> np.ndarray:
    """
    Run Grounding DINO and return the single box (xyxy, absolute px) whose
    centre is closest to the image centre.  Raises RuntimeError if nothing found.
    """
    hf_prompt = ROCK_PROMPT.rstrip(".") + "."
    inputs    = processor(images=image_pil, text=hf_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    w, h = image_pil.size
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=ROCK_BOX_THRESHOLD,
        text_threshold=ROCK_TEXT_THRESHOLD,
        target_sizes=[(h, w)],
    )[0]

    boxes  = results["boxes"].cpu().numpy()   # (N, 4) xyxy
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]

    if len(boxes) == 0:
        raise RuntimeError(
            "Grounding DINO found no rock. Try lowering ROCK_BOX_THRESHOLD."
        )

    print(f"Grounding DINO — {len(boxes)} detection(s):")
    for lbl, sc, bx in zip(labels, scores, boxes):
        print(f"  {lbl:20s}  conf={sc:.3f}  box={bx.astype(int).tolist()}")

    # Pick the box whose centre is closest to the image centre
    img_cx, img_cy = w / 2, h / 2
    box_centres    = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                                (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
    dists          = np.linalg.norm(box_centres - np.array([img_cx, img_cy]), axis=1)
    best           = int(np.argmin(dists))

    print(f"\nSelected: '{labels[best]}'  conf={scores[best]:.3f}  "
          f"box={boxes[best].astype(int).tolist()}")
    return boxes[best].astype(np.float32)   # (4,) xyxy


# ══════════════════════════════════════════════════════════════════════════════
# Step 2b – Comic-style contour outline of every detail
# ══════════════════════════════════════════════════════════════════════════════

def detect_comic_contours(gray: np.ndarray) -> list:
    """
    Edge-detect at every detail level — silhouette, cracks, texture, grain —
    and keep every contour long enough to matter, for a comic-style "inked"
    rendering (as opposed to just the single outer rock boundary).

    Mirrors find_detail_contours() in rock_contour_tuner.py; tune the
    COMIC_* constants there interactively (it shows live parameter values
    and contour counts), then copy the final numbers back here.
    """
    blur_size = COMIC_BLUR | 1
    dilate_k  = max(1, COMIC_DILATE_K) | 1

    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    edges   = cv2.Canny(blurred, COMIC_CANNY_LOW, COMIC_CANNY_HIGH)

    if dilate_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        edges  = cv2.dilate(edges, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.arcLength(c, True) >= COMIC_MIN_LEN]


def save_comic_outline(isolated_rgb: np.ndarray, contours: list):
    """Render every detected detail contour as black ink on white, plot + save."""
    canvas = np.full(isolated_rgb.shape, 255, dtype=np.uint8)
    cv2.drawContours(canvas, contours, -1, COMIC_LINE_COLOR, COMIC_LINE_THICKNESS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(isolated_rgb)
    axes[0].set_title("Isolated rock")
    axes[0].axis("off")
    axes[1].imshow(canvas)
    axes[1].set_title(f"Comic-style outline ({len(contours)} contours)")
    axes[1].axis("off")
    plt.suptitle("Step 2b – Comic-Style Contour Outline", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_COMIC_OUTLINE, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_COMIC_OUTLINE}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 – Classic CV: detect cracks inside the rock ROI
# ══════════════════════════════════════════════════════════════════════════════

def _crack_blackhat_tile(gray_tile: np.ndarray, kernel_size: int, blur_size: int,
                         close_k: int, min_area: int) -> np.ndarray:
    """Black-hat + local-Otsu crack mask for a single tile — see crack_param_tuner.py."""
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


def detect_cracks_tiled_blackhat(isolated_gray: np.ndarray, rock_mask: np.ndarray,
                                  rock_box_xyxy: np.ndarray) -> tuple[np.ndarray, list]:
    """
    Black-hat crack detection over a CRACK_TILE_N×CRACK_TILE_N grid spanning
    the rock bounding box, each tile thresholded with its own local Otsu value
    and stitched back together with overlap — ported from crack_param_tuner.py
    so lighting that varies across a large rock face doesn't wash out a single
    global threshold.  CRACK_TILE_N=1 reduces to one global threshold over the
    whole box (the previous behaviour).  rock_mask (bool H×W) is ANDed at the
    end to strip any border artefacts.
    Returns (crack_mask – full-image uint8 mask (255=crack),
             contours   – crack contours in full-image coords).
    """
    bx1, by1, bx2, by2 = map(int, rock_box_xyxy)
    box_gray = isolated_gray[by1:by2, bx1:bx2]
    bh, bw   = box_gray.shape

    tile_n   = max(1, int(CRACK_TILE_N))
    box_mask = np.zeros((bh, bw), dtype=np.uint8)
    th, tw   = bh / tile_n, bw / tile_n
    pad_h, pad_w = int(th * CRACK_TILE_OVERLAP), int(tw * CRACK_TILE_OVERLAP)

    for row in range(tile_n):
        for col in range(tile_n):
            y1 = int(row * th);  y2 = int((row + 1) * th) if row < tile_n - 1 else bh
            x1 = int(col * tw);  x2 = int((col + 1) * tw) if col < tile_n - 1 else bw

            y1e = max(0, y1 - pad_h);  y2e = min(bh, y2 + pad_h)
            x1e = max(0, x1 - pad_w);  x2e = min(bw, x2 + pad_w)

            tile     = box_gray[y1e:y2e, x1e:x2e]
            tile_out = _crack_blackhat_tile(tile, CRACK_KERNEL_SIZE, CRACK_BLUR_SIZE,
                                             CRACK_CLOSE_K, CRACK_MIN_AREA)

            dy1, dx1 = y1 - y1e, x1 - x1e
            box_mask[y1:y2, x1:x2] = np.maximum(
                box_mask[y1:y2, x1:x2],
                tile_out[dy1:dy1 + (y2 - y1), dx1:dx1 + (x2 - x1)])

    full_mask = np.zeros(isolated_gray.shape, dtype=np.uint8)
    full_mask[by1:by2, bx1:bx2] = box_mask
    full_mask = cv2.bitwise_and(full_mask, rock_mask.astype(np.uint8) * 255)

    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= CRACK_MIN_AREA]
    print(f"[cracks:tiled_blackhat] {tile_n}×{tile_n} grid → "
          f"{len(contours)} region(s) after rock-mask clip")
    return full_mask, contours


def detect_cracks_frangi(isolated_rgb: np.ndarray, rock_mask: np.ndarray) -> tuple[np.ndarray, list]:
    """
    Multi-scale Frangi ridge-filter crack detection — runs the *exact* same
    pipeline as crack_detection.py's run_pipeline() on the SAM-isolated rock
    image: CLAHE preprocessing → shadow suppression → multiscale Frangi
    vesselness → Otsu segmentation → elongation/length/area shape filtering →
    AND with rock_mask (mirroring that script's __main__ masking step), so
    results stay directly comparable to crack_detection.py / its tuner.

    crack_detection.py loads its isolated image from the saved
    OUT_SAM_ISOLATION file, whose background is filled with
    dark grey [30, 30, 30] (see isolate_rock_with_sam). The in-memory
    isolated_rgb used elsewhere in this script fills the background with pure
    black instead — a brightness difference right at the silhouette edge that
    shifts the Frangi response enough to change the final overlay. Rebuilding
    that [30, 30, 30] fill here makes the pixel data byte-identical to what
    crack_detection.py sees (PNG round-trips losslessly), so results match.

    Returns (crack_mask – full-image uint8 mask (255=crack),
             contours   – crack contours in full-image coords).
    """
    frangi_rgb = isolated_rgb.copy()
    frangi_rgb[~rock_mask] = (30, 30, 30)
    isolated_bgr    = cv2.cvtColor(frangi_rgb, cv2.COLOR_RGB2BGR)
    blurred, _      = crack_preprocess(isolated_bgr)
    keep_mask, _    = crack_suppress_shadows(isolated_bgr, blurred)
    response        = detect_cracks_multiscale(blurred, sigmas=FRANGI_SIGMAS)
    binary, _       = crack_segment(response, keep_mask)
    filtered, stats = crack_filter_by_shape(
        binary, min_aspect_ratio=FRANGI_MIN_ASPECT,
        min_length=FRANGI_MIN_LENGTH, min_area=FRANGI_MIN_AREA)

    full_mask = cv2.bitwise_and(filtered, rock_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[cracks:frangi_multiscale] {len(stats)} candidate(s) → "
          f"{len(contours)} region(s) after rock-mask clip")
    return full_mask, contours


def detect_cracks(isolated_rgb: np.ndarray, isolated_gray: np.ndarray,
                   rock_box_xyxy: np.ndarray, rock_mask: np.ndarray) -> tuple[np.ndarray, list]:
    """Dispatch to the crack-detection method selected via CRACK_METHOD."""
    if CRACK_METHOD == "tiled_blackhat":
        return detect_cracks_tiled_blackhat(isolated_gray, rock_mask, rock_box_xyxy)
    if CRACK_METHOD == "frangi_multiscale":
        return detect_cracks_frangi(isolated_rgb, rock_mask)
    raise ValueError(f"Unknown CRACK_METHOD: {CRACK_METHOD!r} "
                     f"(expected 'tiled_blackhat' or 'frangi_multiscale')")


def save_crack_outputs(image_rgb: np.ndarray, crack_mask: np.ndarray,
                       contours: list, rock_box_xyxy: np.ndarray):
    # Binary mask file
    cv2.imwrite(OUT_CRACK_MASK, crack_mask)
    print(f"Saved → {OUT_CRACK_MASK}")

    # Overlay: red cracks + green contours on original, boxed rock in orange
    overlay = image_rgb.copy()
    overlay[crack_mask > 0] = (
        np.array(image_rgb[crack_mask > 0], dtype=np.float32) * 0.5 +
        np.array([255, 0, 0], dtype=np.float32) * 0.5
    ).astype(np.uint8)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    x1, y1, x2, y2 = map(int, rock_box_xyxy)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 165, 0), 3)
    cv2.putText(overlay, "rock (DINO)", (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    axes[0].imshow(image_rgb);      axes[0].set_title("Original")
    axes[1].imshow(crack_mask, cmap="gray"); axes[1].set_title("Crack Mask (CV)")
    axes[2].imshow(overlay);        axes[2].set_title("Overlay  (red=crack, green=contour)")
    for ax in axes:
        ax.axis("off")
    plt.suptitle("Step 2 – Classic CV Crack Detection", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_CRACK_OVERLAY, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_CRACK_OVERLAY}")
    plt.close(fig)


def visualise_crack_sizes(image_rgb: np.ndarray, crack_mask: np.ndarray, save_path: str):
    """
    "Final overlay" — colour-code each detected crack by width, mirroring
    crack_detection.py's visualise(): hairline / medium / wide.

    Classifies every connected component directly from its own
    minor_axis_length on the mask that's actually being displayed (rather than
    looking it up via a label captured during an earlier filtering pass, which
    is what made crack_detection.py's version silently drop most cracks — see
    the fix in that file). That guarantees every crack present in crack_mask
    gets coloured here, regardless of which CRACK_METHOD produced it.
    """
    color_map = {"hairline": (255, 220, 0), "medium": (255, 140, 0), "wide": (255, 30, 30)}
    overlay = image_rgb.copy()
    labeled = measure.label(crack_mask > 0)
    regions = measure.regionprops(labeled)
    for region in regions:
        overlay[labeled == region.label] = color_map[classify_size(region.minor_axis_length)]

    legend_patches = [
        mpatches.Patch(color=np.array(color_map["hairline"]) / 255, label="Hairline crack (<3px)"),
        mpatches.Patch(color=np.array(color_map["medium"])   / 255, label="Medium crack (3–8px)"),
        mpatches.Patch(color=np.array(color_map["wide"])     / 255, label="Wide crack (≥8px)"),
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(overlay)
    ax.set_title(f"Final overlay — cracks coloured by width  ({len(regions)} region(s))",
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9,
              framealpha=0.7, facecolor="black", labelcolor="white")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 – SAM two-pass rock segmentation
#   3a. SamPredictor + DINO box  → single precise rock mask
#   3b. SamAutomaticMaskGenerator on masked crop → up to SAM_MAX_SEGMENTS parts
# ══════════════════════════════════════════════════════════════════════════════

def load_sam(checkpoint: str, model_type: str, device: str):
    print(f"\nLoading SAM ({model_type}) …")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.eval().to(device)
    return sam


def isolate_rock_with_sam(sam, image_rgb: np.ndarray,
                           rock_box_xyxy: np.ndarray) -> np.ndarray:
    """
    Step 3a — use SamPredictor with the DINO box to get one clean rock mask.
    Returns a (H, W) bool array; saves OUT_SAM_ISOLATION.
    """
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    box_t = torch.tensor(rock_box_xyxy[None], dtype=torch.float32).to(predictor.device)
    box_t = predictor.transform.apply_boxes_torch(box_t, image_rgb.shape[:2])
    masks, scores, _ = predictor.predict_torch(
        point_coords=None, point_labels=None,
        boxes=box_t, multimask_output=False,
    )
    rock_mask = masks[0, 0].cpu().numpy().astype(bool)   # (H, W)
    print(f"[SAM 3a] Rock isolation mask: {rock_mask.sum():,} px  "
          f"(score={scores[0, 0].item():.3f})")

    # Visualise and save
    isolated = image_rgb.copy()
    isolated[~rock_mask] = [30, 30, 30]   # dark background outside rock
    overlay = image_rgb.astype(np.float32).copy()
    overlay[rock_mask] = overlay[rock_mask] * 0.5 + np.array([70, 130, 180], np.float32) * 0.5
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("Rock mask overlay")
    axes[1].axis("off")
    axes[2].imshow(isolated)
    axes[2].set_title("Rock isolated (dark bg)")
    axes[2].axis("off")
    plt.suptitle("Step 3a – SAM Rock Isolation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_SAM_ISOLATION_FIGURE, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_SAM_ISOLATION_FIGURE}")
    plt.close(fig)

    cv2.imwrite(OUT_SAM_ISOLATION, cv2.cvtColor(isolated, cv2.COLOR_RGB2BGR))
    print(f"Saved → {OUT_SAM_ISOLATION}")

    cv2.imwrite(OUT_SAM_ROCK_MASK, (rock_mask * 255).astype(np.uint8))
    print(f"Saved → {OUT_SAM_ROCK_MASK}")

    return rock_mask


def segment_rock_auto(sam, image_rgb: np.ndarray,
                       rock_mask: np.ndarray,
                       rock_box_xyxy: np.ndarray) -> list[dict]:
    """
    Step 3b — mask the image to the rock, crop to the DINO box, then run
    SamAutomaticMaskGenerator.  Keep up to SAM_MAX_SEGMENTS largest masks
    and remap them back to full-image coordinates.
    """
    # Black out everything outside the rock before auto-segmentation
    masked_image = image_rgb.copy()
    masked_image[~rock_mask] = 0

    x1, y1, x2, y2 = map(int, rock_box_xyxy)
    crop = masked_image[y1:y2, x1:x2]

    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=SAM_POINTS_PER_SIDE,
        pred_iou_thresh=SAM_PRED_IOU_THRESH,
        stability_score_thresh=SAM_STABILITY_THRESH,
        min_mask_region_area=SAM_MIN_MASK_AREA,
    )

    print(f"[SAM 3b] Auto-segmenting masked rock crop ({x2-x1}×{y2-y1} px) …")
    raw_masks = generator.generate(crop)
    print(f"[SAM 3b] {len(raw_masks)} segments found before filtering")

    # Discard masks that mostly fall outside the rock (black border artefacts)
    rock_mask_crop = rock_mask[y1:y2, x1:x2]
    filtered = []
    for m in raw_masks:
        seg = m["segmentation"]
        overlap = float((seg & rock_mask_crop).sum()) / max(seg.sum(), 1)
        if overlap >= SAM_ROCK_OVERLAP_THRESH:
            filtered.append(m)
    print(f"[SAM 3b] {len(filtered)} segments after rock-overlap filter "
          f"(threshold={SAM_ROCK_OVERLAP_THRESH:.0%}) → keeping top {SAM_MAX_SEGMENTS}")

    top_n     = SAM_MAX_SEGMENTS if SAM_MAX_SEGMENTS > 0 else len(filtered)
    raw_masks = sorted(filtered, key=lambda m: m["area"], reverse=True)[:top_n]

    h_full, w_full = image_rgb.shape[:2]
    full_masks = []
    for m in raw_masks:
        full_seg = np.zeros((h_full, w_full), dtype=bool)
        full_seg[y1:y2, x1:x2] = m["segmentation"]
        full_masks.append({
            "segmentation":  full_seg,
            "area":          int(full_seg.sum()),
            "predicted_iou": m["predicted_iou"],
        })

    print(f"[SAM 3b] Returning {len(full_masks)} segment(s)")
    return full_masks


# ══════════════════════════════════════════════════════════════════════════════
# Step 4b – CLIP: classify each SAM segment
# ══════════════════════════════════════════════════════════════════════════════

def load_clip(device: str):
    print(f"Loading CLIP ({CLIP_MODEL}, {CLIP_PRETRAINED}) …")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    return model, preprocess, tokenizer


def encode_text_prompts(clip_model, tokenizer, device: str) -> dict:
    embeddings = {}
    with torch.no_grad():
        for cat_name, (_, prompts) in CATEGORIES.items():
            tokens = tokenizer(prompts).to(device)
            embeds = clip_model.encode_text(tokens)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            embeddings[cat_name] = embeds.mean(dim=0)
    return embeddings


def extract_segment_crop(image_rgb: np.ndarray, seg: np.ndarray) -> np.ndarray | None:
    """
    Tight crop of image_rgb to the segment bounding box (+ CONTEXT_PADDING).
    All pixels outside the segment are set to neutral grey so CLIP only sees
    the segment itself.  Returns None if the segment is empty.
    """
    ys, xs = np.where(seg)
    if len(xs) == 0:
        return None
    x1 = max(0, int(xs.min()) - CONTEXT_PADDING)
    y1 = max(0, int(ys.min()) - CONTEXT_PADDING)
    x2 = min(image_rgb.shape[1], int(xs.max()) + CONTEXT_PADDING)
    y2 = min(image_rgb.shape[0], int(ys.max()) + CONTEXT_PADDING)
    crop = image_rgb[y1:y2, x1:x2].copy()
    crop[~seg[y1:y2, x1:x2]] = 128
    return crop


def classify_crop(crop: np.ndarray, clip_model, preprocess,
                  text_embeddings: dict, device: str) -> tuple[str, float]:
    """Run CLIP on a pre-extracted segment crop and return (category, confidence)."""
    tensor = preprocess(Image.fromarray(crop)).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = clip_model.encode_image(tensor)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    scores   = {cat: (img_emb @ emb.unsqueeze(-1)).item()
                for cat, emb in text_embeddings.items()}
    best_cat = max(scores, key=scores.__getitem__)
    vals     = np.array(list(scores.values()))
    softmax  = np.exp(vals) / np.exp(vals).sum()
    conf     = float(softmax[list(scores.keys()).index(best_cat)])
    return best_cat, conf


def save_segments_grid(image_rgb: np.ndarray, masks: list[dict]):
    rng = np.random.default_rng(42)

    # All-masks overlay
    overlay = image_rgb.astype(np.float32).copy()
    for m in masks:
        color = rng.integers(50, 230, size=3).astype(np.float32)
        overlay[m["segmentation"]] = overlay[m["segmentation"]] * 0.4 + color * 0.6
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig_ov, ax_ov = plt.subplots(figsize=(10, 7))
    ax_ov.imshow(overlay)
    ax_ov.set_title(f"All SAM Segments ({len(masks)} found)")
    ax_ov.axis("off")
    plt.suptitle("Step 4a – SAM Auto-Segments", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_ov.savefig(OUT_SAM_ALL_SEGMENTS, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_SAM_ALL_SEGMENTS}")
    plt.close(fig_ov)


def classify_all_masks(image_rgb: np.ndarray, masks: list[dict],
                       clip_model, preprocess, text_embeddings: dict,
                       device: str) -> list[dict]:
    print(f"Classifying {len(masks)} segment(s) individually with CLIP …")
    results = []
    for i, m in enumerate(masks):
        crop = extract_segment_crop(image_rgb, m["segmentation"])
        if crop is None:
            cat, conf = next(iter(CATEGORIES)), 0.0
        else:
            cat, conf = classify_crop(crop, clip_model, preprocess, text_embeddings, device)
        sz = f"{crop.shape[1]}×{crop.shape[0]}" if crop is not None else "empty"
        results.append({**m, "category": cat, "confidence": conf})
        print(f"  seg {i:02d} [crop {sz:>12s}]: {cat:28s}  conf={conf:.2f}")
    return results


def save_classified_outputs(image_rgb: np.ndarray, classified: list[dict],
                             rock_mask: np.ndarray):
    h, w      = image_rgb.shape[:2]
    cat_names = list(CATEGORIES.keys())
    palette   = np.array([CATEGORIES[c][0] for c in cat_names], dtype=np.uint8)

    # -1 = unassigned; only pixels inside the rock mask get a label
    label_map  = np.full((h, w), fill_value=-1, dtype=np.int32)
    cover_area = np.full((h, w), 10**9)
    for m in sorted(classified, key=lambda x: x["area"], reverse=True):
        seg  = m["segmentation"] & rock_mask          # restrict to rock only
        area = m["area"]
        cat  = m["category"] if m["confidence"] >= MIN_CONFIDENCE else cat_names[0]
        idx  = cat_names.index(cat)
        update = seg & (area < cover_area)
        label_map[update]  = idx
        cover_area[update] = area

    # Colored label map: dark grey outside rock, category color inside
    colored = np.full((h, w, 3), 40, dtype=np.uint8)          # dark grey bg
    rock_labeled = rock_mask & (label_map >= 0)
    colored[rock_labeled] = palette[label_map[rock_labeled]]

    # Overlay: original outside rock, semi-transparent category color inside
    overlay = image_rgb.copy().astype(np.float32)
    if rock_labeled.any():
        cat_color = palette[label_map[rock_labeled]].astype(np.float32)
        overlay[rock_labeled] = overlay[rock_labeled] * 0.45 + cat_color * 0.55
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Legend: only categories that actually appear in results
    seen_cats = {
        (m["category"] if m["confidence"] >= MIN_CONFIDENCE else cat_names[0])
        for m in classified
    }
    patches = [
        mpatches.Patch(color=np.array(CATEGORIES[c][0]) / 255,
                       label=c.replace("_", " ").title())
        for c in cat_names if c in seen_cats
    ]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(colored)
    axes[1].set_title("CLIP Label Map (rock only)")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (rock only)")
    axes[2].axis("off")

    if patches:
        fig.legend(handles=patches, loc="lower center", ncol=len(patches),
                   fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.04))
    plt.suptitle("Step 4 – SAM + CLIP Classification", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_SAM_OVERLAY, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_SAM_OVERLAY}")
    plt.close(fig)


def save_bio_overlay(image_rgb: np.ndarray, classified: list[dict], save_path: str):
    """Single-panel overlay showing only the CLIP-classified biological colonisation segments."""
    bio_cat  = "biological colonization"
    bio_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
    for m in classified:
        if m["category"] == bio_cat and m["confidence"] >= MIN_CONFIDENCE:
            bio_mask |= m["segmentation"]
    bio_color = np.array(
        CATEGORIES[bio_cat][0] if bio_cat in CATEGORIES else (80, 180, 80),
        dtype=np.float32)

    overlay = image_rgb.copy().astype(np.float32)
    if bio_mask.any():
        overlay[bio_mask] = overlay[bio_mask] * 0.45 + bio_color * 0.55
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(overlay)
    ax.set_title(f"Biological colonization overlay  ({int(bio_mask.sum()):,} px)",
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    legend_patches = [mpatches.Patch(color=bio_color / 255, label="Biological colonization (CLIP)")]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9,
              framealpha=0.7, facecolor="black", labelcolor="white")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Combined overlay (shared by the static final-overlay image and the interactive figure)
# ══════════════════════════════════════════════════════════════════════════════

def build_overlay_compose(image_rgb: np.ndarray, isolated_rgb: np.ndarray,
                           crack_mask: np.ndarray, classified: list[dict],
                           sam_masks: list[dict]):
    """
    Precompute every overlay layer — SAM segments, segment borders, CLIP
    biological colonisation, classic-CV cracks — and return a
    compose(bg_key, alpha_bg, alpha_seg, alpha_bio, alpha_crack, alpha_border)
    function that flattens them onto either background image at the given
    opacities. Shared by save_final_overlay() (one static PNG) and
    show_interactive_figure() (same layers, live sliders).
    """
    h, w = image_rgb.shape[:2]

    backgrounds = {
        "Isolated rock": isolated_rgb.astype(np.float32),
        "Original":      image_rgb.astype(np.float32),
    }
    neutral_canvas = np.full((h, w, 3), 255, dtype=np.float32)  # faded-out background = white

    # Precompute segment fill layer (consistent random colours)
    rng = np.random.default_rng(42)
    seg_colors  = np.zeros((h, w, 3), dtype=np.float32)
    seg_present = np.zeros((h, w), dtype=bool)
    for m in sam_masks:
        color = rng.integers(50, 230, size=3).astype(np.float32)
        seg_colors[m["segmentation"]]  = color
        seg_present                   |= m["segmentation"]

    # Precompute segment border layer — outlines only (see option1_sam.py
    # draw_mask_borders), drawn in black so they read clearly against any background.
    border_canvas = np.zeros((h, w), dtype=np.uint8)
    for m in sam_masks:
        seg_u8 = m["segmentation"].astype(np.uint8) * 255
        contours, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(border_canvas, contours, -1, 255, 2)
    border_mask  = border_canvas > 0
    border_color = np.zeros(3, dtype=np.float32)

    # Precompute biological colonisation layer
    bio_cat  = "biological colonization"
    bio_mask = np.zeros((h, w), dtype=bool)
    for m in classified:
        if m["category"] == bio_cat and m["confidence"] >= MIN_CONFIDENCE:
            bio_mask |= m["segmentation"]
    bio_color = np.array(
        CATEGORIES[bio_cat][0] if bio_cat in CATEGORIES else (80, 180, 80),
        dtype=np.float32)

    # Precompute crack layer
    crack_bool  = crack_mask > 0
    crack_color = np.array([220, 60, 60], dtype=np.float32)

    def compose(bg_key: str, alpha_bg: float,
                alpha_seg: float, alpha_bio: float, alpha_crack: float,
                alpha_border: float) -> np.ndarray:
        result = backgrounds[bg_key] * alpha_bg + neutral_canvas * (1 - alpha_bg)

        if alpha_seg > 0 and seg_present.any():
            result[seg_present] = (result[seg_present] * (1 - alpha_seg)
                                   + seg_colors[seg_present] * alpha_seg)
        if alpha_bio > 0 and bio_mask.any():
            result[bio_mask] = (result[bio_mask] * (1 - alpha_bio)
                                + bio_color * alpha_bio)
        if alpha_crack > 0 and crack_bool.any():
            result[crack_bool] = (result[crack_bool] * (1 - alpha_crack)
                                  + crack_color * alpha_crack)
        if alpha_border > 0 and border_mask.any():
            result[border_mask] = (result[border_mask] * (1 - alpha_border)
                                   + border_color * alpha_border)

        return np.clip(result, 0, 255).astype(np.uint8)

    return compose


# Default layer opacities — shared by the static final-overlay image and the
# interactive figure's initial slider positions, so the static PNG matches
# what the interactive window shows on open.
OVERLAY_DEFAULT_BG          = "Isolated rock"
OVERLAY_DEFAULT_BG_OPACITY  = 1.0
OVERLAY_DEFAULT_SEG_OPACITY = 0.4
OVERLAY_DEFAULT_BIO_OPACITY = 0.6
OVERLAY_DEFAULT_CRACK_OPACITY   = 0.6
OVERLAY_DEFAULT_BORDER_OPACITY  = 0.6

OVERLAY_LEGEND_PATCHES = [
    mpatches.Patch(color=np.array([220, 60, 60]) / 255,    label="Cracks (CV)"),
    mpatches.Patch(color=np.array([80, 180, 80]) / 255,    label="Biological colonization (CLIP)"),
    mpatches.Patch(color=np.array([120, 120, 200]) / 255,  label="Segments (SAM, filled)"),
    mpatches.Patch(color="black",                          label="Segment borders (SAM, outline)"),
]


def save_final_overlay(image_rgb: np.ndarray, isolated_rgb: np.ndarray,
                        crack_mask: np.ndarray, classified: list[dict],
                        sam_masks: list[dict], save_path: str):
    """Static PNG combining every layer — segments, bio colonisation, cracks, borders — in one image."""
    compose = build_overlay_compose(image_rgb, isolated_rgb, crack_mask, classified, sam_masks)
    result  = compose(OVERLAY_DEFAULT_BG, OVERLAY_DEFAULT_BG_OPACITY,
                       OVERLAY_DEFAULT_SEG_OPACITY, OVERLAY_DEFAULT_BIO_OPACITY,
                       OVERLAY_DEFAULT_CRACK_OPACITY, OVERLAY_DEFAULT_BORDER_OPACITY)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(result)
    ax.set_title("Final overlay — all layers combined", fontsize=13, fontweight="bold")
    ax.axis("off")
    ax.legend(handles=OVERLAY_LEGEND_PATCHES, loc="upper right", fontsize=10, framealpha=0.85)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Interactive figure
# ══════════════════════════════════════════════════════════════════════════════

def show_interactive_figure(image_rgb: np.ndarray, isolated_rgb: np.ndarray,
                             crack_mask: np.ndarray, classified: list[dict],
                             sam_masks: list[dict]):
    """
    Interactive matplotlib window with opacity sliders for every layer, plus
    radio buttons to pick the background image and the segment-border colour:
      - Background     : isolated rock or original image, with its own opacity
      - Segments       : all SAM auto-seg masks, filled (random colours)
      - Bio. colon.    : CLIP-classified biological colonisation segments (green)
      - Cracks         : classic-CV crack mask (red)
      - Borders        : SAM segment outlines only (random colours, or black)
    """
    compose = build_overlay_compose(image_rgb, isolated_rgb, crack_mask, classified, sam_masks)

    INIT_BG, INIT_BG_OP            = OVERLAY_DEFAULT_BG, OVERLAY_DEFAULT_BG_OPACITY
    INIT_SEG, INIT_BIO, INIT_CRACK = OVERLAY_DEFAULT_SEG_OPACITY, OVERLAY_DEFAULT_BIO_OPACITY, OVERLAY_DEFAULT_CRACK_OPACITY
    INIT_BORDER_OP                 = OVERLAY_DEFAULT_BORDER_OPACITY

    fig, ax = plt.subplots(figsize=(14, 10))
    plt.subplots_adjust(left=0.06, right=0.96, top=0.93, bottom=0.38)

    im = ax.imshow(compose(INIT_BG, INIT_BG_OP, INIT_SEG, INIT_BIO, INIT_CRACK,
                           INIT_BORDER_OP))
    ax.set_title("Interactive Rock Analysis  –  use sliders / radio buttons to adjust the view",
                 fontsize=13, fontweight="bold")
    ax.axis("off")

    ax.legend(handles=OVERLAY_LEGEND_PATCHES, loc="upper right", fontsize=10,
              framealpha=0.85)

    # ── Sliders (left column) ────────────────────────────────────────────────
    # Left edge pushed in from 0.10 to 0.24 so the slider labels (e.g.
    # "Background opacity") have room to render without falling off the figure.
    ax_bg_op  = plt.axes([0.24, 0.30, 0.48, 0.025])
    ax_seg    = plt.axes([0.24, 0.26, 0.48, 0.025])
    ax_bio    = plt.axes([0.24, 0.22, 0.48, 0.025])
    ax_crack  = plt.axes([0.24, 0.18, 0.48, 0.025])
    ax_border = plt.axes([0.24, 0.14, 0.48, 0.025])

    sl_bg_op  = Slider(ax_bg_op,  "Background opacity", 0.0, 1.0, valinit=INIT_BG_OP,     color="lightgrey")
    sl_seg    = Slider(ax_seg,    "Segments",           0.0, 1.0, valinit=INIT_SEG,       color="lightblue")
    sl_bio    = Slider(ax_bio,    "Bio. colon.",        0.0, 1.0, valinit=INIT_BIO,       color="lightgreen")
    sl_crack  = Slider(ax_crack,  "Crack opacity",      0.0, 1.0, valinit=INIT_CRACK,     color="salmon")
    sl_border = Slider(ax_border, "Border opacity",     0.0, 1.0, valinit=INIT_BORDER_OP, color="khaki")

    # ── Radio buttons (right column) ─────────────────────────────────────────
    ax_bg_radio = plt.axes([0.80, 0.14, 0.10, 0.16])
    ax_bg_radio.set_title("Background", fontsize=9)

    radio_bg = RadioButtons(ax_bg_radio, ("Isolated rock", "Original"), active=0)

    def update(_=None):
        im.set_data(compose(
            radio_bg.value_selected, sl_bg_op.val,
            sl_seg.val, sl_bio.val, sl_crack.val,
            sl_border.val,
        ))
        fig.canvas.draw_idle()

    for slider in (sl_bg_op, sl_seg, sl_bio, sl_crack, sl_border):
        slider.on_changed(update)
    radio_bg.on_clicked(update)

    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Load image
    bgr = cv2.imread(IMAGE_PATH)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {IMAGE_PATH}")
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    gray      = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w      = gray.shape

    # ── Step 1: Grounding DINO → rock bounding box ───────────────────────────
    print("=" * 60)
    print("Step 1 – Grounding DINO: locating central rock …")
    gdino_model, processor = load_gdino_hf(DEVICE)
    rock_box = detect_rock_box(gdino_model, processor, image_pil, DEVICE)

    # Draw and save the raw detection for reference
    boxed = image_rgb.copy()
    x1, y1, x2, y2 = map(int, rock_box)
    cv2.rectangle(boxed, (x1, y1), (x2, y2), (255, 165, 0), 3)
    cv2.putText(boxed, "rock (DINO)", (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
    cv2.imwrite(OUT_ROCK_BOX, cv2.cvtColor(boxed, cv2.COLOR_RGB2BGR))
    print(f"Saved → {OUT_ROCK_BOX}")

    # ── Step 2: SAM — isolate the rock precisely ─────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2 – SAM: isolating the rock …")
    sam       = load_sam(SAM_CHECKPOINT, SAM_MODEL_TYPE, DEVICE)
    rock_mask = isolate_rock_with_sam(sam, image_rgb, rock_box)

    # Build isolated grayscale for classic CV (non-rock pixels are black=0)
    isolated_rgb  = image_rgb.copy()
    isolated_rgb[~rock_mask] = 0
    isolated_gray = cv2.cvtColor(isolated_rgb, cv2.COLOR_RGB2GRAY)

    # ── Step 2b: Comic-style outline of every detail on the isolated rock ────
    print("\n" + "=" * 60)
    print("Step 2b – Comic-style outline: tracing every detail contour …")
    comic_contours = detect_comic_contours(isolated_gray)
    print(f"[comic] {len(comic_contours)} contour(s) kept (≥{COMIC_MIN_LEN}px perimeter)")
    save_comic_outline(isolated_rgb, comic_contours)

    # ── Step 3: Classic CV cracks on the isolated rock image ─────────────────
    print("\n" + "=" * 60)
    print(f"Step 3 – Classic CV: detecting cracks on isolated rock (method={CRACK_METHOD!r}) …")
    crack_mask, crack_contours = detect_cracks(isolated_rgb, isolated_gray, rock_box, rock_mask)
    save_crack_outputs(isolated_rgb, crack_mask, crack_contours, rock_box)
    visualise_crack_sizes(isolated_rgb, crack_mask, OUT_CRACK_SIZE_OVERLAY)

    # ── Step 4: SAM auto-seg + CLIP classification ───────────────────────────
    print("\n" + "=" * 60)
    print("Step 4a – SAM: auto-segmenting the masked rock …")
    sam_masks = segment_rock_auto(sam, image_rgb, rock_mask, rock_box)
    save_segments_grid(image_rgb, sam_masks)

    mean_area   = float(np.mean([m["area"] for m in sam_masks]))
    clip_masks  = [m for m in sam_masks if m["area"] < mean_area]
    print(f"\nMean segment area: {mean_area:,.0f} px²  "
          f"→  {len(clip_masks)}/{len(sam_masks)} segments below mean passed to CLIP")

    print("\n" + "=" * 60)
    print("Step 4b – CLIP: classifying segments smaller than mean …")
    clip_model, clip_preprocess, tokenizer = load_clip(DEVICE)
    text_embeddings = encode_text_prompts(clip_model, tokenizer, DEVICE)
    classified = classify_all_masks(
        isolated_rgb, clip_masks, clip_model, clip_preprocess, text_embeddings, DEVICE)
    save_classified_outputs(image_rgb, classified, rock_mask)
    save_bio_overlay(image_rgb, classified, OUT_BIO_OVERLAY)

    # ── Step 5: Final combined overlay (all layers in one static image) ──────
    print("\n" + "=" * 60)
    print("Step 5 – Saving final overlay (segments + bio colonization + cracks + borders) …")
    save_final_overlay(image_rgb, isolated_rgb, crack_mask, classified, sam_masks, OUT_FINAL_OVERLAY)

    # ── Step 6: Interactive figure ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 6 – Interactive figure (close window to exit) …")
    show_interactive_figure(image_rgb, isolated_rgb, crack_mask, classified, sam_masks)

    print("\nDone.")


if __name__ == "__main__":
    main()
