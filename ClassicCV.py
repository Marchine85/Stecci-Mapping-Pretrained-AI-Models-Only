import cv2
import numpy as np
# pip3 install scikit-image
from skimage import color, filters, morphology

image = cv2.imread("stecci_1_raw.jpeg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# --- Mask 1: Green biological colonisation (moss/lichen) ---
# Green hues in HSV
lower_green = np.array([25, 20, 30])
upper_green = np.array([90, 255, 200])
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, np.ones((5,5)))

# --- Mask 2: Cracks ---
# Cracks are dark, thin linear features
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Enhance cracks with morphological black-hat
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
_, mask_cracks = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)

# --- Mask 3: Rock surface (everything else) ---
# Combine all specific masks and invert
combined = cv2.bitwise_or(mask_green, mask_cracks)
mask_rock = cv2.bitwise_not(combined)

# Visualize
result = image.copy()
result[mask_green > 0] = [0, 200, 0]
result[mask_cracks > 0] = [0, 0, 0]
cv2.imwrite("segmented_output.jpg", result)