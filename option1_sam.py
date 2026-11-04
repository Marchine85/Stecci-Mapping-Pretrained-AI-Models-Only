"""
Option 1: SAM (Segment Anything Model) - Automatic Mask Generation
==================================================================
Segments the image into regions automatically, without knowing what they are.
Each mask is a distinct region; you can then inspect/classify them manually.

Requirements:
    pip install segment-anything torch torchvision opencv-python matplotlib numpy

Model checkpoint:
    Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    Or smaller variants:
      vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
      vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_PATH      = "stecci_1_raw.jpeg"
CHECKPOINT_PATH = r"sam_model_ckpts\sam_vit_h_4b8939.pth"  # change to vit_l / vit_b if needed
MODEL_TYPE      = "vit_h"                   # "vit_h" | "vit_l" | "vit_b"
OUTPUT_PATH     = "output_sam_masks.png"

# Mask generator tuning (adjust to get more/fewer segments)
POINTS_PER_SIDE       = 32    # higher = more candidate points = more masks
PRED_IOU_THRESH       = 0.90  # keep masks with predicted IoU above this
STABILITY_SCORE_THRESH= 0.90  # keep only stable masks
MIN_MASK_REGION_AREA  = 1000   # discard tiny masks (pixels)
# ──────────────────────────────────────────────────────────────────────────────


def load_image(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def generate_masks(image_rgb: np.ndarray) -> list[dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM model on {device.upper()} …")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device)
    sam.eval()

    mask_gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=POINTS_PER_SIDE,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_SCORE_THRESH,
        min_mask_region_area=MIN_MASK_REGION_AREA,
    )

    print("Generating masks …")
    masks = mask_gen.generate(image_rgb)
    print(f"Found {len(masks)} masks.")
    return masks


def colorize_masks(image_rgb: np.ndarray, masks: list[dict]) -> np.ndarray:
    """Overlay semi-transparent random colors on each mask."""
    overlay = image_rgb.copy().astype(np.float32)
    rng = np.random.default_rng(42)

    # Sort by area descending so large masks don't hide small ones
    sorted_masks = sorted(masks, key=lambda m: m["area"], reverse=True)

    for mask_info in sorted_masks:
        mask = mask_info["segmentation"]          # bool array H×W
        color = rng.integers(50, 255, size=3)     # random RGB
        overlay[mask] = overlay[mask] * 0.4 + color * 0.6

    return np.clip(overlay, 0, 255).astype(np.uint8)


def draw_mask_borders(image_rgb: np.ndarray, masks: list[dict]) -> np.ndarray:
    """Draw each mask's outline only (no fill) on top of the original image."""
    overlay = image_rgb.copy()
    rng = np.random.default_rng(42)

    for mask_info in sorted(masks, key=lambda m: m["area"], reverse=True):
        mask = mask_info["segmentation"].astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = rng.integers(50, 255, size=3).tolist()
        cv2.drawContours(overlay, contours, -1, color, 2)

    return overlay


def plot_results(image_rgb: np.ndarray, masks: list[dict], colored: np.ndarray, borders: np.ndarray):
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.suptitle("SAM – Automatic Segmentation", fontsize=16, fontweight="bold")

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(colored)
    axes[1].set_title(f"All Masks ({len(masks)} regions)")
    axes[1].axis("off")

    axes[2].imshow(borders)
    axes[2].set_title(f"Segment Borders ({len(masks)} regions)")
    axes[2].axis("off")

    # Show individual masks as a grid (first 12 max)
    n = min(len(masks), 12)
    cols = 4
    rows = (n + cols - 1) // cols
    fig2, axes2 = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    fig2.suptitle("Individual Masks (sorted by area, largest first)", fontsize=14)
    sorted_masks = sorted(masks, key=lambda m: m["area"], reverse=True)

    for i, ax in enumerate(axes2.flat):
        if i < n:
            mask = sorted_masks[i]["segmentation"]
            crop = image_rgb.copy()
            crop[~mask] = [220, 220, 220]  # grey out everything outside mask
            ax.imshow(crop)
            ax.set_title(f"Mask {i+1}  ({sorted_masks[i]['area']:,} px)", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    return fig, fig2


def save_masks_as_numpy(masks: list[dict], path: str = "sam_masks.npz"):
    """Save all boolean masks to a compressed numpy file for later use."""
    arrays = {f"mask_{i:03d}": m["segmentation"] for i, m in enumerate(masks)}
    areas  = np.array([m["area"] for m in masks])
    np.savez_compressed(path, areas=areas, **arrays)
    print(f"Saved {len(masks)} masks → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    image_rgb = load_image(IMAGE_PATH)
    masks     = generate_masks(image_rgb)
    colored   = colorize_masks(image_rgb, masks)
    borders   = draw_mask_borders(image_rgb, masks)

    fig1, fig2 = plot_results(image_rgb, masks, colored, borders)

    fig1.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    fig2.savefig("output_sam_individual_masks.png", dpi=150, bbox_inches="tight")
    save_masks_as_numpy(masks)

    print(f"Saved overview  → {OUTPUT_PATH}")
    print("Saved individual masks → output_sam_individual_masks.png")
    plt.show()
