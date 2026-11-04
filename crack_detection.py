"""
Rock crack detection pipeline (classical, no deep learning)
Requires: pip install opencv-python scikit-image numpy matplotlib scipy

Edit the CONFIG block below to point at your image and tune parameters.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure, filters, morphology, measure, color
from skimage.filters import frangi, hessian, meijering, sato
from scipy import ndimage


# ─────────────────────────────────────────────
# 0. Demo image generator (no real image needed)
# ─────────────────────────────────────────────

def make_demo_image(h=512, w=512):
    """Synthesise a rock-like surface with cracks of different widths and a shadow blob."""
    rng = np.random.default_rng(42)

    # Rock texture: Perlin-ish noise via layered Gaussians
    base = rng.normal(160, 20, (h, w)).astype(np.float32)
    for sigma in [4, 8, 16]:
        base += rng.normal(0, 10, (h, w)) * (sigma / 16)
    base = np.clip(base, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    # Thin crack (2 px wide, diagonal)
    cv2.line(img, (60, 80), (300, 430), (30, 25, 20), 2)
    # Medium crack (4 px, curved – approximated with polyline)
    pts = np.array([[200, 50], [240, 150], [220, 280], [260, 400], [310, 480]], np.int32)
    cv2.polylines(img, [pts], False, (25, 20, 15), 4)
    # Wide crack (8 px)
    cv2.line(img, (380, 30), (460, 500), (20, 15, 10), 8)

    # Shadow region: large, smooth, dark blob (not a crack)
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (130, 350), (90, 60), 30, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (61, 61), 20)
    img = img.astype(np.float32)
    img -= mask[:, :, None] * 80
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img


# ─────────────────────────────────────────────
# 1. Preprocessing
# ─────────────────────────────────────────────

def preprocess(img_bgr):
    """
    Convert to grayscale and apply CLAHE to normalise uneven lighting.
    A mild Gaussian blur suppresses fine grain / noise.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE – adaptive contrast enhancement, tile size 8x8 is a good default
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Mild blur to reduce texture speckle without smearing crack edges
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return blurred, gray


# ─────────────────────────────────────────────
# 2. Shadow suppression
# ─────────────────────────────────────────────

def suppress_shadows(img_bgr, blurred_gray):
    """
    Shadows are dark AND low local-variance AND low saturation.
    Cracks are dark but have high local contrast/variance.

    Returns a binary mask: 1 = possible crack region, 0 = shadow / background
    """
    # --- (a) Saturation mask ---
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    # Shadows have low saturation; a threshold around 20-40 works for most rocks
    # (cracks are also achromatic so they'll have low sat too – we combine with variance)
    low_sat = saturation < 30  # True = achromatic (could be crack OR shadow)

    # --- (b) Local variance ---
    # Compute local standard deviation using a sliding window (approx with Gaussian)
    gray_f = blurred_gray.astype(np.float32)
    mu  = cv2.GaussianBlur(gray_f, (21, 21), 5)
    mu2 = cv2.GaussianBlur(gray_f ** 2, (21, 21), 5)
    local_var = np.sqrt(np.maximum(mu2 - mu ** 2, 0))

    # High variance = crack-like edge; low variance = smooth shadow
    high_var = local_var > np.percentile(local_var, 40)

    # --- (c) Darkness ---
    dark = blurred_gray < 140  # only dark pixels are candidates

    # Shadow mask: dark + low-variance + achromatic  →  NOT a crack
    shadow_mask = dark & (~high_var) & low_sat

    # Morphological closing to fill gaps in shadow blobs
    shadow_mask = morphology.binary_closing(shadow_mask, morphology.disk(8))

    # Invert: 1 = keep (could be crack), 0 = shadow
    keep_mask = ~shadow_mask

    return keep_mask.astype(np.uint8), shadow_mask.astype(np.uint8)


# ─────────────────────────────────────────────
# 3. Multi-scale crack detection (Frangi + pyramid)
# ─────────────────────────────────────────────

def detect_cracks_multiscale(blurred_gray, sigmas=(0.5, 1, 2, 3, 5)):
    """
    Frangi vesselness filter highlights elongated, ridge-like structures
    across multiple scales (sigma controls sensitivity to crack width).

    sigmas:
        ~0.5–1  → hairline / thin cracks
        ~2–3    → medium cracks
        ~4–6    → wide cracks / fractures
    """
    gray_f = blurred_gray.astype(np.float32) / 255.0

    # scikit-image frangi: black_ridges=True detects dark lines on bright bg
    #
    # Call frangi separately per sigma rather than passing the whole list at
    # once: with gamma=None, frangi auto-estimates gamma only from the FIRST
    # sigma it processes and then reuses that single value for every other
    # scale. Small sigmas yield much larger Hessian norms than large sigmas,
    # so a gamma derived from a small sigma silently suppresses the response
    # of larger sigmas (and vice versa). Per-sigma calls let each scale get
    # its own correctly-matched gamma; we then combine via per-pixel max,
    # mirroring frangi's own multiscale combination.
    responses = [
        frangi(
            gray_f,
            sigmas=[sigma],
            black_ridges=True,
            alpha=0.5,   # controls sensitivity to blob vs. plate-like structures
            beta=1.0,    # controls sensitivity to the background
            gamma=None,  # auto-estimated per scale
        )
        for sigma in sigmas
    ]
    response = np.maximum.reduce(responses)

    # Normalise to 0-255 for display / further processing
    r_min, r_max = response.min(), response.max()
    if r_max > r_min:
        response_norm = ((response - r_min) / (r_max - r_min) * 255).astype(np.uint8)
    else:
        response_norm = np.zeros_like(blurred_gray)

    return response_norm


# ─────────────────────────────────────────────
# 4. Thresholding & binary segmentation
# ─────────────────────────────────────────────

def segment_cracks(frangi_response, keep_mask):
    """
    Otsu threshold on the Frangi response, then apply the shadow keep-mask.
    Morphological cleaning to remove isolated specks.
    """
    # Otsu on the Frangi response
    _, binary = cv2.threshold(
        frangi_response, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    # Apply shadow suppression mask
    binary = binary & (keep_mask * 255)

    # Remove tiny isolated blobs (min area = 50 px)
    binary_bool = binary > 0
    labeled = measure.label(binary_bool)
    for region in measure.regionprops(labeled):
        if region.area < 50:
            binary_bool[labeled == region.label] = False

    # Skeletonise to get 1-px wide crack centre-lines (optional but useful)
    skeleton = morphology.skeletonize(binary_bool)

    return (binary_bool * 255).astype(np.uint8), (skeleton * 255).astype(np.uint8)


# ─────────────────────────────────────────────
# 5. Shape-based filtering (crack vs. residual shadow)
# ─────────────────────────────────────────────

def filter_by_shape(binary, min_aspect_ratio=3.0, min_length=20, min_area=300):
    """
    True cracks are elongated (high aspect ratio), reasonably long, and have
    sufficient area.  All three gates must pass to avoid small false positives.

    Returns:
        filtered  – binary image with only crack-like components
        stats     – list of dicts with per-crack properties
    """
    labeled = measure.label(binary > 0)
    filtered = np.zeros_like(binary)
    stats = []

    for region in measure.regionprops(labeled):
        if region.minor_axis_length < 1:
            continue
        aspect = region.major_axis_length / region.minor_axis_length
        length = region.major_axis_length
        solidity = region.solidity

        if aspect >= min_aspect_ratio and length >= min_length and region.area >= min_area:
            filtered[labeled == region.label] = 255
            stats.append({
                "label":        region.label,
                "length_px":    round(length, 1),
                "width_px":     round(region.minor_axis_length, 1),
                "aspect_ratio": round(aspect, 2),
                "solidity":     round(solidity, 3),
                "area_px":      region.area,
                "centroid":     (round(region.centroid[1]), round(region.centroid[0])),
            })

    return filtered, stats


# ─────────────────────────────────────────────
# 6. Classify crack size
# ─────────────────────────────────────────────

def classify_size(width_px):
    """Simple rule-based size classifier (adjust thresholds to your scale)."""
    if width_px < 3:
        return "hairline"
    elif width_px < 8:
        return "medium"
    else:
        return "wide"


# ─────────────────────────────────────────────
# 7. Visualise results
# ─────────────────────────────────────────────

def visualise(img_bgr, shadow_mask, frangi_response, binary, filtered, save_path=None):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Build overlay
    #
    # NOTE: `filtered` only contains a SUBSET of the regions in `binary`, so
    # relabelling it assigns fresh sequential IDs that no longer match the
    # `label` values stored in `stats` (those came from labelling `binary`
    # inside filter_by_shape). Looking cracks up via `s["label"]` therefore
    # missed most of them. Instead, classify each region directly from its
    # own `minor_axis_length` on the relabelled `filtered` mask — every
    # shape-filtered crack is then guaranteed to get coloured in.
    overlay = img_rgb.copy()
    color_map = {"hairline": (255, 220, 0), "medium": (255, 140, 0), "wide": (255, 30, 30)}
    labeled = measure.label(filtered > 0)
    for region in measure.regionprops(labeled):
        overlay[labeled == region.label] = color_map[classify_size(region.minor_axis_length)]

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=np.array([255, 220, 0]) / 255, label="Hairline crack"),
        Patch(facecolor=np.array([255, 140, 0]) / 255, label="Medium crack"),
        Patch(facecolor=np.array([255, 30, 30]) / 255, label="Wide crack"),
    ]
    '''
    # ── Figure 1: pipeline steps ──────────────────────────────────────────────
    fig1, axes1 = plt.subplots(1, 4, figsize=(22, 6))
    fig1.suptitle("Crack Detection – Pipeline Steps", fontsize=13, fontweight="bold")
    for ax, data, title, kw in zip(
        axes1,
        [img_rgb, shadow_mask, frangi_response, binary],
        ["Original image", "Shadow mask (rejected)", "Frangi vesselness", "Binary segmentation"],
        [{}, {"cmap": "hot"}, {"cmap": "inferno"}, {"cmap": "gray"}],
    ):
        ax.imshow(data, **kw)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    '''

    # ── Figure 2: final results ───────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 7))
    fig2.suptitle("Crack Detection – Final Results", fontsize=13, fontweight="bold")

    axes2[0].imshow(filtered, cmap="gray")
    axes2[0].set_title("Shape-filtered cracks", fontsize=11, fontweight="bold")
    axes2[0].axis("off")

    axes2[1].imshow(overlay)
    axes2[1].set_title("Final overlay", fontsize=11, fontweight="bold")
    axes2[1].axis("off")
    axes2[1].legend(handles=legend_elements, loc="lower right", fontsize=9,
                    framealpha=0.7, facecolor="black", labelcolor="white")

    plt.tight_layout()

    if save_path:
        fig2.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Result saved to {save_path}")

    plt.show()


# ─────────────────────────────────────────────
# 8. Main pipeline
# ─────────────────────────────────────────────

def run_pipeline(img_bgr, verbose=True,
                 frangi_sigmas=(0.5, 1, 2, 3, 5),
                 min_aspect_ratio=3.0,
                 min_length=20,
                 min_area=300):
    """End-to-end crack detection. Returns result dict."""

    print("Step 1/5 – Preprocessing …")
    blurred, gray = preprocess(img_bgr)

    print("Step 2/5 – Shadow suppression …")
    keep_mask, shadow_mask = suppress_shadows(img_bgr, blurred)

    print("Step 3/5 – Multi-scale Frangi filter …")
    frangi_response = detect_cracks_multiscale(blurred, sigmas=frangi_sigmas)

    print("Step 4/5 – Segmentation …")
    binary, skeleton = segment_cracks(frangi_response, keep_mask)

    print("Step 5/5 – Shape filtering …")
    filtered, stats = filter_by_shape(binary, min_aspect_ratio=min_aspect_ratio, min_length=min_length, min_area=min_area)

    if verbose:
        print(f"\n{'─'*40}")
        print(f"Found {len(stats)} crack(s):\n")
        for s in stats:
            size = classify_size(s["width_px"])
            print(f"  Crack #{s['label']:02d} | {size:8s} | "
                  f"length={s['length_px']:6.1f}px  "
                  f"width={s['width_px']:5.1f}px  "
                  f"aspect={s['aspect_ratio']:5.1f}  "
                  f"centroid={s['centroid']}")
        print(f"{'─'*40}\n")

    return {
        "blurred":        blurred,
        "shadow_mask":    shadow_mask,
        "frangi":         frangi_response,
        "binary":         binary,
        "skeleton":       skeleton,
        "filtered":       filtered,
        "stats":          stats,
    }


# ─────────────────────────────────────────────
# CONFIG – edit these before running
# ─────────────────────────────────────────────

ISOLATION_PATH = r"D:\Machine_Learning\Stecci\Manually_Testing_Stuff\step2_output_combined_sam_rock_isolation.png"
ROCK_MASK_PATH = r"D:\Machine_Learning\Stecci\Manually_Testing_Stuff\step2_output_combined_sam_rock_mask.png"
SAVE_PATH      = r"D:\Machine_Learning\Stecci\Manually_Testing_Stuff\stecci_1_result.png"
COMPARISON_SAVE_PATH = r"D:\Machine_Learning\Stecci\Manually_Testing_Stuff\stecci_1_filter_comparison.png"
MIN_ASPECT    = 2.5    # major/minor axis ratio — higher = stricter elongation requirement
MIN_LENGTH    = 100     # minimum crack length in px — filters small false-positive blobs
MIN_AREA      = 1000    # minimum crack area in px² — second guard against small speckles
FRANGI_SIGMAS = np.arange(1.0, 7.1, 1.0)  # sigma ≈ half the crack width; extend upward for wider macro cracks


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    isolated_bgr = cv2.imread(ISOLATION_PATH)
    if isolated_bgr is None:
        raise FileNotFoundError(f"Cannot read: {ISOLATION_PATH}\nRun option5 first to generate it.")

    rock_mask_gray = cv2.imread(ROCK_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if rock_mask_gray is None:
        raise FileNotFoundError(f"Cannot read: {ROCK_MASK_PATH}\nRun option5 first to generate it.")
    rock_mask = (rock_mask_gray > 127).astype(np.uint8)

    print(f"Loaded: {ISOLATION_PATH}")

    result = run_pipeline(
        isolated_bgr,
        verbose=True,
        frangi_sigmas=FRANGI_SIGMAS,
        min_aspect_ratio=MIN_ASPECT,
        min_length=MIN_LENGTH,
        min_area=MIN_AREA,
    )

    # Strip anything outside the rock from every output mask
    result["shadow_mask"] = cv2.bitwise_and(result["shadow_mask"], rock_mask)
    result["binary"]      = cv2.bitwise_and(result["binary"],      rock_mask * 255)
    result["filtered"]    = cv2.bitwise_and(result["filtered"],    rock_mask * 255)

    visualise(
        isolated_bgr,
        result["shadow_mask"],
        result["frangi"],
        result["binary"],
        result["filtered"],
        save_path=SAVE_PATH,
    )

