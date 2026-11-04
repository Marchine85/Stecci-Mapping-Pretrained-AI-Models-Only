"""
Option 4: Grounding DINO + SAM – Open-Vocabulary Segmentation
=============================================================
Pipeline:
  1. Grounding DINO (HuggingFace) detects bounding boxes for text-prompted categories
     (e.g. "crack . lichen . moss . rock")
  2. SAM uses those boxes as prompts to generate precise pixel masks

Requirements:
    pip install torch torchvision transformers
    pip install segment-anything
    pip install opencv-python numpy matplotlib Pillow

  SAM checkpoint:
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pathlib import Path
from PIL import Image

import torch
from segment_anything import sam_model_registry, SamPredictor

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_PATH   = "stecci_1_raw.jpeg"
OUTPUT_PATH  = "output_grounding_dino_sam.png"
DEVICE       = "cuda"     # change to "cuda" if available

# SAM checkpoint
SAM_CHECKPOINT = r"sam_model_ckpts\sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"   # "vit_h" | "vit_l" | "vit_b"

# ── Prompts ───────────────────────────────────────────────────────────────────
# Grounding DINO takes a dot-separated string of categories.
# Each token becomes a class the detector looks for.
# Keep phrases short (1–3 words) and period-separated, no trailing period.
TEXT_PROMPT = "crack . lichen . moss . pitting . rock surface . white encrustation"

# Detection thresholds (lower = more detections, higher = fewer but more confident)
BOX_THRESHOLD  = 0.30
TEXT_THRESHOLD = 0.25

# ── Category display colors (RGB 0–255) ───────────────────────────────────────
CATEGORY_COLORS = {
    "crack":               ( 40,  40,  40),
    "lichen":              ( 80, 180,  80),
    "moss":                ( 50, 150,  50),
    "pitting":             (180, 100, 200),
    "rock surface":        (160, 140, 120),
    "white encrustation":  (240, 240, 240),
}
# Fallback color for any label not in the dict above
DEFAULT_COLOR = (200, 100,  50)
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# Grounding DINO – HuggingFace Transformers
# ══════════════════════════════════════════════════════════════════════════════

def load_gdino_hf(device: str):
    """Load Grounding DINO from HuggingFace (downloads automatically)."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model     = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return model, processor


def predict_gdino_hf(model, processor, image_pil: Image.Image,
                     text_prompt: str,
                     box_threshold: float,
                     text_threshold: float,
                     device: str):
    """
    Returns:
        boxes_xyxy – (N, 4) float32 numpy array, absolute pixel coords [x1,y1,x2,y2]
        scores     – (N,)   float32 numpy array
        labels     – list[str]
    """
    # HF expects the prompt as "category1. category2. category3."
    hf_prompt = text_prompt.replace(" . ", ". ") + "."

    inputs  = processor(images=image_pil, text=hf_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    w, h = image_pil.size
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(h, w)],
    )[0]

    boxes  = results["boxes"].cpu().numpy()    # (N,4) xyxy absolute
    scores = results["scores"].cpu().numpy()   # (N,)
    labels = results["labels"]                 # list[str]
    return boxes, scores, labels


# ══════════════════════════════════════════════════════════════════════════════
# SAM mask generation from boxes
# ══════════════════════════════════════════════════════════════════════════════

def load_sam_predictor(checkpoint: str, model_type: str, device: str) -> SamPredictor:
    print(f"Loading SAM ({model_type}) …")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.eval().to(device)
    return SamPredictor(sam)


def predict_masks_from_boxes(predictor: SamPredictor,
                              image_rgb: np.ndarray,
                              boxes_xyxy: np.ndarray) -> np.ndarray:
    """
    Run SAM with box prompts.
    Returns masks array of shape (N, H, W) bool.
    """
    predictor.set_image(image_rgb)

    if len(boxes_xyxy) == 0:
        return np.empty((0, *image_rgb.shape[:2]), dtype=bool)

    input_boxes = torch.tensor(boxes_xyxy, dtype=torch.float32).to(predictor.device)
    transformed = predictor.transform.apply_boxes_torch(input_boxes, image_rgb.shape[:2])

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed,
        multimask_output=False,   # one mask per box
    )
    # masks shape: (N, 1, H, W) → squeeze to (N, H, W)
    return masks.squeeze(1).cpu().numpy().astype(bool)


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def get_color(label: str) -> tuple:
    label_lower = label.lower().strip()
    for key, color in CATEGORY_COLORS.items():
        if key in label_lower or label_lower in key:
            return color
    return DEFAULT_COLOR


def draw_boxes(image_rgb: np.ndarray, boxes_xyxy: np.ndarray,
               labels: list, scores: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    for box, label, score in zip(boxes_xyxy, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        color_rgb = get_color(label)
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2)
        cv2.putText(out, f"{label} {score:.2f}",
                    (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2)
    return out


def build_overlay(image_rgb: np.ndarray,
                  masks: np.ndarray,
                  labels: list,
                  alpha: float = 0.50) -> np.ndarray:
    overlay = image_rgb.astype(np.float32).copy()
    for mask, label in zip(masks, labels):
        color = np.array(get_color(label), dtype=np.float32)
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def build_label_map(masks: np.ndarray, labels: list,
                    h: int, w: int) -> tuple[np.ndarray, list]:
    """
    Build a single-channel label map.
    Smaller masks (higher specificity) win overlapping pixels.
    """
    unique_labels = list(dict.fromkeys(labels))   # preserves order, no dupes
    label_map  = np.zeros((h, w), dtype=np.int32)
    cover_area = np.full((h, w), fill_value=h * w + 1)

    for mask, label in zip(masks, labels):
        area   = mask.sum()
        idx    = unique_labels.index(label) + 1   # 0 = background
        update = mask & (area < cover_area)
        label_map[update]  = idx
        cover_area[update] = area

    return label_map, unique_labels


def plot_results(image_rgb, boxed_image, overlay, masks, labels, scores,
                 label_map, unique_labels, output_path):

    n_masks  = len(masks)
    grid_cols = 4
    grid_rows = max(1, (n_masks + grid_cols - 1) // grid_cols)

    fig = plt.figure(figsize=(26, 10 + 4 * grid_rows))
    gs  = fig.add_gridspec(2 + grid_rows, grid_cols, hspace=0.35, wspace=0.15)

    # Row 0: original + box detections + overlay + label map
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(image_rgb);     ax_orig.set_title("Original Image"); ax_orig.axis("off")

    ax_box = fig.add_subplot(gs[0, 1])
    ax_box.imshow(boxed_image);    ax_box.set_title("DINO Detections"); ax_box.axis("off")

    ax_ov = fig.add_subplot(gs[0, 2])
    ax_ov.imshow(overlay);         ax_ov.set_title("SAM Masks Overlay"); ax_ov.axis("off")

    # Legend
    patches = []
    seen    = set()
    for label in labels:
        if label not in seen:
            c = np.array(get_color(label)) / 255
            patches.append(mpatches.Patch(color=c, label=label.title()))
            seen.add(label)
    ax_leg = fig.add_subplot(gs[0, 3])
    ax_leg.legend(handles=patches, loc="center", fontsize=12, frameon=False)
    ax_leg.set_title("Legend"); ax_leg.axis("off")

    # Row 1: individual masks
    for i in range(n_masks):
        row = 1 + i // grid_cols
        col = i % grid_cols
        ax  = fig.add_subplot(gs[row, col])
        crop = image_rgb.copy()
        crop[~masks[i]] = [220, 220, 220]
        color_rgb = np.array(get_color(labels[i])) / 255
        ax.imshow(crop)
        ax.set_title(f"{labels[i]}\n{scores[i]:.2f}", fontsize=9,
                     color=color_rgb if sum(color_rgb) < 2.5 else "black")
        ax.axis("off")

    fig.suptitle("Grounding DINO + SAM – Open-Vocabulary Segmentation",
                 fontsize=17, fontweight="bold")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.show()


def print_stats(masks: np.ndarray, labels: list, total_px: int):
    from collections import defaultdict
    cat_px = defaultdict(int)
    for mask, label in zip(masks, labels):
        cat_px[label] += int(mask.sum())

    print("\n── Detection / coverage statistics ──────")
    print(f"  Total detections : {len(labels)}")
    for cat, px in sorted(cat_px.items()):
        print(f"  {cat:28s}: {px:8,} px  ({100*px/total_px:.1f} %)")
    print("─────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # 1. Load image
    bgr = cv2.imread(IMAGE_PATH)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {IMAGE_PATH}")
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    h, w      = image_rgb.shape[:2]

    # 2. Run Grounding DINO
    print("Loading Grounding DINO via HuggingFace Transformers …")
    gdino_model, processor = load_gdino_hf(DEVICE)
    boxes_xyxy, scores, labels = predict_gdino_hf(
        gdino_model, processor, image_pil,
        TEXT_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, DEVICE)

    print(f"\nGrounding DINO found {len(labels)} detections:")
    for label, score, box in zip(labels, scores, boxes_xyxy):
        print(f"  {label:25s}  conf={score:.3f}  box={box.astype(int).tolist()}")

    if len(labels) == 0:
        print("No detections — try lowering BOX_THRESHOLD / TEXT_THRESHOLD.")
        return

    # 3. SAM mask prediction from boxes
    sam_predictor = load_sam_predictor(SAM_CHECKPOINT, SAM_MODEL_TYPE, DEVICE)
    masks = predict_masks_from_boxes(sam_predictor, image_rgb, boxes_xyxy)

    # 4. Stats
    print_stats(masks, labels, h * w)

    # 5. Visualise
    boxed_image = draw_boxes(image_rgb, boxes_xyxy, labels, scores)
    overlay     = build_overlay(image_rgb, masks, labels)
    label_map, unique_labels = build_label_map(masks, labels, h, w)

    plot_results(image_rgb, boxed_image, overlay, masks, labels, scores,
                 label_map, unique_labels, OUTPUT_PATH)

    # 6. Save individual masks
    for i, (mask, label) in enumerate(zip(masks, labels)):
        fname = f"output_gdino_mask_{i:02d}_{label.replace(' ', '_')}.png"
        cv2.imwrite(fname, (mask * 255).astype(np.uint8))
        print(f"  saved → {fname}")


if __name__ == "__main__":
    main()
