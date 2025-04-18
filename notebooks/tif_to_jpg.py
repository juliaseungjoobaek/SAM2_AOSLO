import os
from PIL import Image

# Define your directory containing the .tif files
video_dir = "./videos/rx_mean"

# Step 1: Delete all .jpg files in the directory
for file in os.listdir(video_dir):
    if file.lower().endswith(".jpg"):
        os.remove(os.path.join(video_dir, file))

# Step 2: Get all .tif files and sort them (you can adjust sorting as needed)
tiff_files = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1].lower() in [".tif", ".tiff"]
]

# Optional: Sort the TIFF files if needed (adjust the key function if required)
tiff_files.sort()

# Step 3: Convert TIFF to JPEG and rename as 00001.jpg, 00002.jpg, etc.
for idx, tiff_file in enumerate(tiff_files, start=1):
    tiff_path = os.path.join(video_dir, tiff_file)
    jpeg_filename = f"{idx:05d}.jpg"  # Formats the index as 00001, 00002, etc.
    jpeg_path = os.path.join(video_dir, jpeg_filename)

    # Open the TIFF file and convert it to JPEG
    with Image.open(tiff_path) as img:
        img.convert("RGB").save(jpeg_path, "JPEG", quality=100)

    print(f"Converted {tiff_file} to {jpeg_filename}")
