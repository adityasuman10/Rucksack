import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import img_to_array

# ---------- Step 1: Load your ISAR grayscale image ----------
img_path = r"C:/vscode/cimulon/images (5).jpeg"  # replace with your file name
img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Resize to standard CNN input size if needed
img_gray = cv2.resize(img_gray, (224, 224))

# ---------- Step 2: Apply radar-style color map ----------
# 'JET' gives the blue→green→yellow→red mapping seen in your reference image
img_colored = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)

# ---------- Step 3: Normalize for model use ----------
img_array = img_to_array(img_colored) / 255.0  # [0,1] float32 array

# ---------- Step 4: Display ----------
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original ISAR (Grayscale)")
plt.imshow(img_gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Colorized ISAR (Radar Style)")
plt.imshow(cv2.cvtColor(img_colored, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
