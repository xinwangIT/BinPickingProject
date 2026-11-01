"""
main.py — orchestration. Edit the config block, then `python main.py`.
"""
from pathlib import Path

import cv2

from bin_picking_pipeline import (
    L515,
    load_models,
    find_depth_file, load_depth,
    process_frame, visualize_point_cloud,
    sweep_dbscan, print_sweep,
    process_ply, save_poses_to_csv,
    run_full_pipeline,
)


# ------------------------------------------------------------
# CONFIG (edit for your machine)
# ------------------------------------------------------------
CANDIDATE_BASES = [
    Path("C:/Users/ITSum/Desktop/RealSenseD405/l515_callibration_25"),
    Path("/Users/ITSum/Desktop/RealSenseD405/l515_callibration_25"),
]
base = next((p for p in CANDIDATE_BASES if p.exists()), CANDIDATE_BASES[0])
print("base:", base, "| exists:", base.exists())

model_dir   = base.parent / "BinPickingProject/vision_Notebook/runs/detect"
yolo_weights = model_dir / "train2/weights/best.pt"
sam_weights  = model_dir / "SAM/sam2.1_b.pt"
rgb_path     = base / "rgb_0000.png"

TOP_K = 1
EPS, MIN_POINTS, RANSAC_THRESH = 0.015, 20, 0.01


def run_one_liner():
    out = run_full_pipeline(
        base_dir=base,
        yolo_weights=yolo_weights, sam_weights=sam_weights,
        rgb_path=rgb_path,
        camera=L515, name="seg_pptc", top_k=TOP_K,
        eps=EPS, min_points=MIN_POINTS, ransac_thresh=RANSAC_THRESH,
        do_sweep=False, save_overlay=True, visualize=False,
    )
    print(out["poses_df"].to_string(index=False))


if __name__ == "__main__":
    run_one_liner()