import cv2
import os
from tqdm import tqdm

# ---------- CONFIGURATION ----------
input_root = r"C:\vscode\cimulon\concave"        # your original dataset folder
output_root = r"C:\vscode\cimulon\marnberryland"    # where colorized images will be saved

# Create output root folder if not exists
os.makedirs(output_root, exist_ok=True)

# Walk through all subfolders (classes)
for class_name in os.listdir(input_root):
    class_path = os.path.join(input_root, class_name)
    if not os.path.isdir(class_path):
        continue  # skip non-folder files

    # Create corresponding output class folder
    output_class_path = os.path.join(output_root, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    # List images in class folder
    image_files = [f for f in os.listdir(class_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]

    print(f"Processing class '{class_name}' with {len(image_files)} images...")

    for file_name in tqdm(image_files, desc=f"Colorizing {class_name}", leave=False):
        img_path = os.path.join(class_path, file_name)
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img_gray is None:
            print(f"⚠️ Skipping unreadable file: {file_name}")
            continue

        # Apply radar-style color map (JET) — NO resizing
        img_colored = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)

        # Save to output folder (same filename)
        out_path = os.path.join(output_class_path, file_name)
        cv2.imwrite(out_path, img_colored)

print("✅ All ISAR class folders have been colorized and saved to:", output_root)
