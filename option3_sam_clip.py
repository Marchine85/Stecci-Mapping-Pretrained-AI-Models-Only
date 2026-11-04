"""
Option 3: SAM + CLIP – Zero-Shot Semantic Segmentation
=======================================================
Pipeline:
  1. SAM generates candidate region masks automatically
  2. Each mask region is cropped from the image
  3. CLIP encodes the crop and a set of text category labels
  4. The crop is assigned to the category with highest cosine similarity

This gives you LABELED segments without any training data.

Requirements:
    pip install segment-anything open-clip-torch torch torchvision
    pip install opencv-python numpy matplotlib Pillow

Model checkpoints:
  SAM: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  CLIP weights are downloaded automatically by open_clip.

Tip: run on GPU for speed (`device = "cuda"`).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import torch
import open_clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_PATH       = "stecci_1_raw.jpeg"
SAM_CHECKPOINT   = r"D:\Machine_Learning\Stecci\App_v0\model_ckpts\sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE   = "vit_b"              # "vit_h" | "vit_l" | "vit_b"
CLIP_MODEL       = "ViT-B-32"           # open_clip model name
CLIP_PRETRAINED  = "openai"             # pretrained weights tag
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_PATH      = "output_sam_clip.png"

# ── Category definitions ──────────────────────────────────────────────────────
# Each key is the category name; value is a list of text prompts fed to CLIP.
# More prompts → more robust classification. Be as descriptive as possible.
CATEGORIES = {
    "rock_surface",
    "biological_colonisation",
    "white_lichen",
    "crack",
    "pitting",
}

# Display colors per category (RGB 0–255)
CATEGORY_COLORS = {
    "rock_surface":            (160, 140, 120),
    "biological_colonisation": (80,  180,  80),
    "white_lichen":            (240, 240, 240),
    "crack":                   (40,   40,  40),
    "pitting":                 (180, 100, 200),
}

# SAM mask generator settings
SAM_POINTS_PER_SIDE        = 32
SAM_PRED_IOU_THRESH        = 0.86
SAM_STABILITY_SCORE_THRESH = 0.92
SAM_MIN_MASK_REGION_AREA   = 800    # pixels; increase to ignore tiny regions

# CLIP classification settings
MIN_CONFIDENCE   = 0.20    # discard masks where top-class prob < this
CONTEXT_PADDING  = 10      # pixels of context added around each crop bbox
# ──────────────────────────────────────────────────────────────────────────────


# ── CLIP helpers ──────────────────────────────────────────────────────────────

def load_clip(model_name: str, pretrained: str, device: str):
    print(f"Loading CLIP ({model_name}, {pretrained}) …")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def encode_text_prompts(model, tokenizer, categories: dict, device: str) -> dict:
    """
    Returns {category_name: mean_text_embedding} for all prompt lists.
    Averaging multiple prompts per category improves robustness.
    """
    text_embeddings = {}
    with torch.no_grad():
        for cat_name, prompts in categories.items():
            tokens = tokenizer(prompts).to(device)
            embeds = model.encode_text(tokens)          # (n_prompts, D)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            text_embeddings[cat_name] = embeds.mean(dim=0)  # (D,)
    return text_embeddings


def classify_crop(image_rgb: np.ndarray, bbox: tuple,
                  model, preprocess, text_embeddings: dict,
                  device: str, padding: int = CONTEXT_PADDING) -> tuple[str, float]:
    """
    Classify a single crop (defined by bbox = x, y, w, h) using CLIP.
    Returns (category_name, confidence_score).
    """
    h_img, w_img = image_rgb.shape[:2]
    x, y, w, h = bbox
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)

    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return "rock_surface", 0.0

    pil_crop = Image.fromarray(crop)
    tensor   = preprocess(pil_crop).unsqueeze(0).to(device)

    with torch.no_grad():
        img_emb = model.encode_image(tensor)            # (1, D)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

    # Cosine similarities against each category's mean text embedding
    scores = {}
    for cat_name, txt_emb in text_embeddings.items():
        sim = (img_emb @ txt_emb.unsqueeze(-1)).item()
        scores[cat_name] = sim

    best_cat   = max(scores, key=scores.__getitem__)
    best_score = scores[best_cat]

    # Softmax for a proper probability
    vals    = np.array(list(scores.values()))
    softmax = np.exp(vals) / np.exp(vals).sum()
    confidence = float(softmax[list(scores.keys()).index(best_cat)])

    return best_cat, confidence


# ── SAM helpers ───────────────────────────────────────────────────────────────

def load_sam_and_generate_masks(image_rgb: np.ndarray) -> list[dict]:
    print("Loading SAM …")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE)
    sam.eval()

    gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=SAM_POINTS_PER_SIDE,
        pred_iou_thresh=SAM_PRED_IOU_THRESH,
        stability_score_thresh=SAM_STABILITY_SCORE_THRESH,
        min_mask_region_area=SAM_MIN_MASK_REGION_AREA,
    )
    print("Generating SAM masks …")
    masks = gen.generate(image_rgb)
    print(f"SAM produced {len(masks)} masks.")
    return masks


# ── Classification pipeline ───────────────────────────────────────────────────

def classify_all_masks(image_rgb, masks, model, preprocess,
                       text_embeddings, device) -> list[dict]:
    print("Classifying masks with CLIP …")
    results = []
    for i, m in enumerate(masks):
        cat, conf = classify_crop(
            image_rgb, m["bbox"], model, preprocess, text_embeddings, device)
        results.append({
            "segmentation": m["segmentation"],
            "bbox":         m["bbox"],
            "area":         m["area"],
            "category":     cat,
            "confidence":   conf,
        })
        if (i + 1) % 10 == 0:
            print(f"  … {i+1}/{len(masks)} done")
    print("Classification complete.")
    return results


def build_label_image(classified: list[dict], h: int, w: int,
                      categories: list[str]) -> np.ndarray:
    """
    Build a label map where each pixel is assigned the category of the
    smallest mask covering it (smaller masks = more specific / foreground).
    Low-confidence masks fall back to rock_surface.
    """
    label_map  = np.zeros((h, w), dtype=np.int32)        # 0 = rock_surface
    cover_area = np.full((h, w), fill_value=10**9)       # track covering mask area

    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # Process from largest to smallest so small masks win
    sorted_masks = sorted(classified, key=lambda m: m["area"], reverse=True)

    for m in sorted_masks:
        seg  = m["segmentation"]
        area = m["area"]
        cat  = m["category"] if m["confidence"] >= MIN_CONFIDENCE else "rock_surface"
        idx  = cat_to_idx[cat]

        # Only overwrite pixels where this mask is smaller than current
        update = seg & (area < cover_area)
        label_map[update]  = idx
        cover_area[update] = area

    return label_map


def colorize(label_map: np.ndarray, categories: list[str]) -> np.ndarray:
    palette = np.array([CATEGORY_COLORS[c] for c in categories], dtype=np.uint8)
    return palette[label_map]


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_results(image_rgb, classified, label_map, colored,
                 categories, output_path):

    overlay = np.clip(
        image_rgb.astype(np.float32) * 0.45 +
        colored.astype(np.float32) * 0.55, 0, 255
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    fig.suptitle("SAM + CLIP – Zero-Shot Semantic Segmentation", fontsize=16, fontweight="bold")

    axes[0].imshow(image_rgb);  axes[0].set_title("Original Image");  axes[0].axis("off")
    axes[1].imshow(colored);    axes[1].set_title("Label Map");        axes[1].axis("off")
    axes[2].imshow(overlay);    axes[2].set_title("Overlay");          axes[2].axis("off")

    patches = [
        mpatches.Patch(color=np.array(CATEGORY_COLORS[c]) / 255, label=c.replace("_", " ").title())
        for c in categories
    ]
    fig.legend(handles=patches, loc="lower center", ncol=len(categories),
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.show()


def print_stats(classified: list[dict], total_px: int):
    from collections import Counter
    cat_px = Counter()
    for m in classified:
        cat = m["category"] if m["confidence"] >= MIN_CONFIDENCE else "rock_surface"
        cat_px[cat] += m["area"]

    print("\n── Category coverage (by mask area) ─────")
    for cat, px in sorted(cat_px.items()):
        print(f"  {cat:30s}: {px:8,} px  ({100*px/total_px:.1f} %)")
    print("─────────────────────────────────────────\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Load image
    bgr = cv2.imread(IMAGE_PATH)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {IMAGE_PATH}")
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    # 2. Load CLIP and encode text prompts
    clip_model, clip_preprocess, tokenizer = load_clip(CLIP_MODEL, CLIP_PRETRAINED, DEVICE)
    text_embeddings = encode_text_prompts(clip_model, tokenizer, CATEGORIES, DEVICE)

    # 3. Generate SAM masks
    masks = load_sam_and_generate_masks(image_rgb)

    # 4. Classify each mask with CLIP
    classified = classify_all_masks(
        image_rgb, masks, clip_model, clip_preprocess, text_embeddings, DEVICE)

    # 5. Build label map and colorize
    categories = list(CATEGORIES.keys())
    label_map  = build_label_image(classified, h, w, categories)
    colored    = colorize(label_map, categories)

    # 6. Stats and output
    print_stats(classified, h * w)
    plot_results(image_rgb, classified, label_map, colored, categories, OUTPUT_PATH)

    # 7. Save per-category masks
    for i, cat in enumerate(categories):
        mask_bin = ((label_map == i) * 255).astype(np.uint8)
        out = f"output_clip_mask_{cat}.png"
        cv2.imwrite(out, mask_bin)
        print(f"  saved → {out}")
