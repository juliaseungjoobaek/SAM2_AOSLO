from video_predictor import SAM2VideoSegmenter
import numpy as np

# Setup paths
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
checkpoint_path =  "../checkpoints/sam2.1_hiera_large.pt"
sam2_module_path = "../sam2/sam2"
video_dir = "./videos/rx_mean"
output_dir = "./videos/overlayed_add_neg_test"

# Create the segmenter
segmenter = SAM2VideoSegmenter(
    model_cfg=model_cfg,
    checkpoint_path=checkpoint_path,
    sam2_module_path=sam2_module_path,
    video_dir=video_dir
)

# Annotation info
ann_frame_idx = 0
points = [[290, 33], [124, 68], [146, 110], [244, 140], [28, 146], [71, 208],
          [174, 229], [223, 250], [110, 278], [125, 280], [75, 309], [56, 374], [391, 196]]
labels = [1] * len(points)

# Run annotation + propagation
video_segments = segmenter.annotate_and_propagate(
    ann_frame_idx=ann_frame_idx,
    points=points,
    labels=labels,
    output_dir=output_dir,
    vis_frame_stride=1
)

# (Optional) Visualize
#segmenter.visualize_segmentation(video_segments, output_dir, vis_frame_stride=1)
# python run_model.py --path --output_dir ! --gpu 