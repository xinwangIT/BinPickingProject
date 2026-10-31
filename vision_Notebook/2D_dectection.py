import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO, SAM
from pathlib import Path
base = Path("/Users/ITSum/Desktop/RealSenseD405/l515_callibration_25")
model_path = Path("/Users/ITSum/Desktop/RealSenseD405/BinPickingProject/vision_Notebook/runs/detect/train2/weights/best.pt")
sam_path = "/Users/ITSum/Desktop/RealSenseD405/BinPickingProject/vision_Notebook/runs/detect/SAM/sam2.1_b.pt"
yolo_model = YOLO(model_path)
sam_model  = SAM(sam_path)

image_path = "/Users/ITSum/Desktop/RealSenseD405/l515_callibration_25/rgb_0000.png"
image = cv2.imread(image_path)
h, w = image.shape[:2]

yolo_results = yolo_model(image)
bboxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)

combined_mask   = np.zeros((h, w), dtype=np.uint8)
per_object_masks = []

if len(bboxes) > 0:
    print(f"YOLO detected {len(bboxes)} objects. Passing boxes to SAM...")
    sam_results = sam_model(image, bboxes=bboxes)

    r = sam_results[0]
    if r.masks is not None:
        masks = (r.masks.data.cpu().numpy() * 255).astype(np.uint8)   # (N, H, W)
        per_object_masks = [masks[i] for i in range(masks.shape[0])]
        combined_mask = masks.max(axis=0)                              # (H, W)

    # save for the depth step
    np.save(base, combined_mask)


    # visualize
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(yolo_results[0].plot(img=image), cv2.COLOR_BGR2RGB))
    plt.title("YOLO detections"); plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(combined_mask, cmap='gray')
    plt.title("SAM combined mask"); plt.axis('off')
    plt.show()

for i, conf in enumerate(yolo_results[0].boxes.conf):
    print(f"Detection {i+1}: {conf.item():.4f}")

# --- Save top-3 confidence parts as separate .npy files ---
base = Path("/Users/ITSum/Desktop/RealSenseD405/l515_callibration_25")
confs_all = yolo_results[0].boxes.conf.cpu().numpy()
bboxes_all = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)

TOP_K = 1
top_idx = np.argsort(confs_all)[::-1][:TOP_K]

top_bboxes = bboxes_all[top_idx]
top_confs  = confs_all[top_idx]

if len(top_bboxes) > 0:
    sam_top = sam_model(image, bboxes=top_bboxes)
    r_top = sam_top[0]
    if r_top.masks is not None:
        top_masks = (r_top.masks.data.cpu().numpy() * 255).astype(np.uint8)  # (K, H, W)
        for i in range(top_masks.shape[0]):
            out_path = f"{base}/top_config.npy"
            np.save(out_path, top_masks[i])
            print(f"Saved {out_path}")
img = np.load(base / "top_config.npy")
plt.imshow(img, cmap='gray')
plt.title("single depth"); plt.axis('off')
plt.show()

import numpy as np, open3d as o3d
from pathlib import Path

# ---------- CAMERA PROFILE (device-specific, set once) ----------
L515 = dict(fx=1358.8, fy=1358.3, cx=971.5, cy=534.0,
            unit_scale=1.0)        # 1.0 = meters, 0.001 = mm->m
# add more profiles as you go, e.g.
# D435 = dict(fx=..., fy=..., cx=..., cy=..., unit_scale=0.001)


def depth_to_partial_ply(depth_seg, camera, out_dir, name,
                         z_min=0.1, z_max=10.0, overwrite=True):
    """Back-project a segmented depth array and save it as a .ply."""
    fx, fy, cx, cy = camera["fx"], camera["fy"], camera["cx"], camera["cy"]

    H, W = depth_seg.shape
    z = depth_seg.astype(np.float32) * camera["unit_scale"]
    valid = np.isfinite(z) & (z > z_min) & (z < z_max)

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x[valid], y[valid], z[valid]], axis=1).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # >>> THIS is where the .ply is created <<<
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)          # ensure folder exists
    out_path = out_dir / f"{name}.ply"                  # (1) build the path
    if overwrite or not out_path.exists():
        o3d.io.write_point_cloud(str(out_path), pcd)    # (2) write bytes to disk
    return out_path


def sweep_dbscan(ply_path, eps_list, mp_list):
    """Load a .ply and sweep DBSCAN params."""
    cloud = o3d.io.read_point_cloud(str(ply_path))
    print(f"loaded {len(cloud.points)} points from {ply_path}")
    for eps in eps_list:
        for mp in mp_list:
            labels = np.asarray(cloud.cluster_dbscan(eps=eps, min_points=mp))
            ids = labels[labels >= 0]
            n = len(set(ids))
            sizes = sorted([int((labels == l).sum()) for l in set(ids)], reverse=True)
            print(f"eps={eps:.3f} mp={mp:3d} -> {n:2d} clusters | top10: {sizes[:10]}")