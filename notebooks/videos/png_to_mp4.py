import cv2
import os
from natsort import natsorted

# Directory with PNG frames
frame_folder = './overlayed_add_neg_v1'

# Get sorted list of PNG files
images = [img for img in os.listdir(frame_folder) if img.endswith(".png")]
images = natsorted(images)

# Read first image to get dimensions
frame_path = os.path.join(frame_folder, images[0])
frame = cv2.imread(frame_path)
height, width, layers = frame.shape

# Define video writer
video = cv2.VideoWriter('output_neg_v1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

# Write each frame
for image in images:
    frame = cv2.imread(os.path.join(frame_folder, image))
    video.write(frame)

video.release()
print("Video saved as output.mp4")
