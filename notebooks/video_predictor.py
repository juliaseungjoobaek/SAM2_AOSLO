import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Add sam2 module path dynamically
def add_sam2_to_sys_path(sam2_module_path):
    abs_path = Path(sam2_module_path).resolve()
    if str(abs_path) not in sys.path:
        sys.path.append(str(abs_path))
    return abs_path

class SAM2VideoSegmenter:
    def __init__(self, model_cfg, checkpoint_path, sam2_module_path, video_dir):
        # Add sam2 module to sys.path
        sam2_module_path = add_sam2_to_sys_path(sam2_module_path)
        from build_sam import build_sam2_video_predictor  # Must come after path append

        # Set environment and device
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        if torch.cuda.is_available():
            print(torch.version.cuda)
            print(torch.cuda.is_available())
            self.device = torch.device("cuda")
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS."
            )
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # Load predictor
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=self.device)

        # Init and reset inference state
        self.inference_state = self.predictor.init_state(video_path=video_dir)
        self.predictor.reset_state(self.inference_state)

    @staticmethod
    def show_mask(mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab20")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords, labels, ax, marker_size=200):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)

    @staticmethod
    def show_points_with_id(coords, labels, ax, obj_ids=None, marker_size=200):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*',
                   s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*',
                   s=marker_size, edgecolor='white', linewidth=1.25)
        if obj_ids is not None:
            for i, (x, y) in enumerate(coords):
                label = labels[i]
                color = 'green' if label == 1 else 'red'
                obj_id = obj_ids[i]
                ax.text(x + 5, y - 5, f"ID {obj_id}", color='white', fontsize=9,
                        bbox=dict(facecolor=color, alpha=0.6, boxstyle='round'))

    @staticmethod
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    


    def annotate_and_propagate(self, ann_frame_idx, points, labels, output_dir, vis_frame_stride=1):
        prompts = {}
        for i, point in enumerate(points):
            obj_id = i + 1
            prompts[obj_id] = point, labels[i]
            _, _, _ = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=ann_frame_idx,
                obj_id=obj_id,
                points=np.array([point], dtype=np.float32),
                labels=np.array([labels[i]], dtype=np.int32)
            )

        video_segments = {}
        active_objects = set()
        os.makedirs(output_dir, exist_ok=True)

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            frame_result = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                if mask.sum() == 0:
                    continue
                active_objects.add(out_obj_id)
                frame_result[out_obj_id] = mask

            if out_frame_idx > 0:
                for obj_id in list(active_objects):
                    if obj_id not in frame_result:
                        active_objects.remove(obj_id)

            video_segments[out_frame_idx] = frame_result

        return video_segments

    def visualize_segmentation(self, video_segments, output_dir, vis_frame_stride=1):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.close("all")
        for out_frame_idx in range(0, len(self.frame_names), vis_frame_stride):
            img_path = os.path.join(self.video_dir, self.frame_names[out_frame_idx])
            if not os.path.exists(img_path):
                print(f"Warning: frame {img_path} not found")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: unable to read image at {img_path}")
                continue

            plt.figure(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(img)

            if out_frame_idx in video_segments:
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    self.show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

            overlayed_img_path = os.path.join(output_dir, f"overlayed_frame_{out_frame_idx}.png")
            plt.axis('off')
            plt.savefig(overlayed_img_path, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close()
