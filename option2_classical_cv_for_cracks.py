import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_PATH  = "stecci_1_raw.jpeg"
OUTPUT_PATH = "output_cv_segmentation.png"

CRACK_KERNEL_SIZE = 13  # black-hat kernel — larger = coarser crack scale (must be odd)
CRACK_BLUR_SIZE   = 41  # pre-blur to suppress micro-texture
CRACK_CLOSE_K     = 1  # closing to bridge gaps along crack length
CRACK_MIN_AREA    = 750  # discard blobs smaller than this (px)

CRACK_COLOR = (255, 0, 0)   # red overlay for cracks
CONTOUR_COLOR = (0, 255, 0) # green for contour outlines
CONTOUR_THICKNESS = 3

# ROI — fractions of image width/height (0.0–1.0); restrict cracks to this region
ROI_X0 = 0.10   # left edge
ROI_X1 = 0.90   # right edge
ROI_Y0 = 0.10   # top edge
ROI_Y1 = 0.90   # bottom edge
# ──────────────────────────────────────────────────────────────────────────────


def load_image(path: str):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def detect_cracks(gray: np.ndarray):
    blurred = cv2.GaussianBlur(gray, (CRACK_BLUR_SIZE, CRACK_BLUR_SIZE), 0)

    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (CRACK_KERNEL_SIZE, CRACK_KERNEL_SIZE))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)

    otsu_val, m = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"[cracks] Otsu threshold: {otsu_val:.0f}  →  {np.sum(m > 0):,} px")

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (CRACK_CLOSE_K, CRACK_CLOSE_K))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(m)
    kept = [c for c in contours if cv2.contourArea(c) >= CRACK_MIN_AREA]
    cv2.drawContours(mask, kept, -1, 255, -1)
    print(f"[cracks] {len(kept)} regions kept after area filter")
    return mask, kept


def visualize(image_rgb, crack_mask, contours, output_path):
    overlay = image_rgb.copy()
    overlay[crack_mask > 0] = (
        np.array(image_rgb[crack_mask > 0], dtype=np.float32) * 0.5 +
        np.array(CRACK_COLOR, dtype=np.float32) * 0.5
    ).astype(np.uint8)
    cv2.drawContours(overlay, contours, -1, CONTOUR_COLOR, CONTOUR_THICKNESS)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[1].imshow(crack_mask, cmap="gray")
    axes[1].set_title("Crack Mask")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (red=crack, green=contour)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.show()


def find_rock_contour(gray: np.ndarray, h: int, w: int) -> np.ndarray:
    """Find the largest contour inside the rectangular ROI — the rock boundary."""
    x0, x1 = int(w * ROI_X0), int(w * ROI_X1)
    y0, y1 = int(h * ROI_Y0), int(h * ROI_Y1)

    roi_gray = gray[y0:y1, x0:x1]
    blurred  = cv2.GaussianBlur(roi_gray, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean up to get one solid rock region
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fall back to the full rectangle if nothing found
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        return mask

    # Largest contour = the rock
    rock_cnt = max(contours, key=cv2.contourArea)

    # Shift contour coordinates back to full-image space
    rock_cnt[:, 0, 0] += x0
    rock_cnt[:, 0, 1] += y0

    rock_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(rock_mask, [rock_cnt], -1, 255, -1)
    return rock_mask, rock_cnt


if __name__ == "__main__":
    bgr, rgb = load_image(IMAGE_PATH)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    crack_mask, _ = detect_cracks(gray)

    # Restrict cracks to the rock contour
    rock_mask, rock_cnt = find_rock_contour(gray, h, w)
    crack_mask = cv2.bitwise_and(crack_mask, rock_mask)

    # Recompute crack contours on the clipped mask
    contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rock boundary on the overlay for reference
    rgb_annotated = rgb.copy()
    cv2.drawContours(rgb_annotated, [rock_cnt], -1, (255, 165, 0), 3)

    visualize(rgb_annotated, crack_mask, contours, OUTPUT_PATH)
