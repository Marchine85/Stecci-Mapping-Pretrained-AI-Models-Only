# pip3 install segment-anything
# pip3 install opencv-python
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=r"D:\Machine_Learning\Stecci\App_v0\model_ckpts\sam_vit_b_01ec64.pth")
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

image = cv2.imread("stecci_1_raw.jpeg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image_rgb)


def show_masks(image, masks):
    overlay = image.copy()
    for mask in masks:
        color = np.random.randint(0, 255, 3, dtype=np.uint8)
        overlay[mask["segmentation"]] = color
    blended = cv2.addWeighted(image, 0.4, overlay, 0.6, 0)
    return blended

result = show_masks(image_rgb, masks)
result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imwrite("masks_output.jpg", result_bgr)
cv2.imshow("SAM Masks", result_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
