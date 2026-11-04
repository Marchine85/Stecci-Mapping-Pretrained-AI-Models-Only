import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

images = {
    "Original":        "stecci_1_raw.jpeg",
    "Rock isolation":  "step2_output_combined_sam_rock_isolation.png",
    "Rock mask":       "step2_output_combined_sam_rock_mask.png",
}

available = {k: v for k, v in images.items() if os.path.exists(v)}
n = len(available)

fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
if n == 1:
    axes = [axes]

for ax, (title, path) in zip(axes, available.items()):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(rgb)
    ax.set_title(title, fontsize=13)
    ax.axis("off")

plt.tight_layout()
plt.show()
