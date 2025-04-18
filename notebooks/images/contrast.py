import cv2
import os

# --- Settings ---
input_dir = '../videos/rx_mean'              # Replace with your image path
output_dir = '../videos/high_contrast'       # Replace with your actual output folder

# Make sure output_dir exists
os.makedirs(output_dir, exist_ok=True)

# Process each image in input_dir
for filename in os.listdir(input_dir):
    if filename.lower().endswith('.jpg'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Load image in grayscale
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping unreadable file: {filename}")
            continue

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(img)

        # Convert grayscale to RGB
        rgb_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB)

        # Save to output_dir with same name
        cv2.imwrite(output_path, rgb_img)
        print(f"Saved (RGB): {output_path}")
