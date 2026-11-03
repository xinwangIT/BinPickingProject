"""
main.py — orchestration. Edit the config block, then `python main.py`.
"""
from pathlib import Path

import cv2
import pandas as pd

from Dectection import (
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

model_dir    = base.parent / "BinPickingProject/vision_Notebook/runs/detect"
yolo_weights = model_dir / "train2/weights/best.pt"
sam_weights  = model_dir / "SAM/sam2.1_b.pt"
rgb_path     = base / "rgb_0000.png"

TOP_K = 1
EPS, MIN_POINTS, RANSAC_THRESH = 0.015, 80, 0.01   # MIN_POINTS: 20 -> 80
MAX_PX_DIST = 150.0                                 # match gate


def run_one_liner():
    out = run_full_pipeline(
        base_dir=base,
        yolo_weights=yolo_weights, sam_weights=sam_weights,
        rgb_path=rgb_path,
        camera=L515, name="seg_pptc", top_k=TOP_K,
        eps=EPS, min_points=MIN_POINTS, ransac_thresh=RANSAC_THRESH,
        max_px_dist=MAX_PX_DIST,          # <-- was missing
        do_sweep=False, save_overlay=True, visualize=False,
    )

    # Ensure the DataFrame prints all columns without truncation
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("\n--- Pipeline Results ---")
    print(out["poses_df"].to_string(index=False))

    # Highlight the pixel coords only if any were confidently matched
    df = out["poses_df"]
    if not df.empty and "center_u" in df.columns and df["center_u"].notna().any():
        print("\n--- Pixel Center Coordinates ---")
        cols = [c for c in ["cluster", "det_idx", "det_px_dist",
                            "center_u", "center_v"] if c in df.columns]
        print(df[cols].to_string(index=False))
    else:
        print("\n[Warning] No confident pixel-center matches were found.")




if __name__ == "__main__":
    run_one_liner()