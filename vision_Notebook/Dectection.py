# ============================================================
# Bin picking: YOLO -> SAM mask -> apply to depth -> partial PLY -> DBSCAN
# ============================================================
import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from pathlib import Path
from ultralytics import YOLO, SAM

# ------------------------------------------------------------
# 1. PATHS  (edit these for your machine)
# ------------------------------------------------------------
# NOTE: your notebook had mixed macOS ("/Users/...") and Windows ("C:/Users/...")
# paths. Use ONE. This block picks the folder that exists.
CANDIDATE_BASES = [
    Path("C:/Users/ITSum/Desktop/RealSenseD405/l515_callibration_25"),
    Path("/Users/ITSum/Desktop/RealSenseD405/l515_callibration_25"),
]
base = next((p for p in CANDIDATE_BASES if p.exists()), CANDIDATE_BASES[0])
print("base:", base, "| exists:", base.exists())

# Model paths — adjust to your layout
model_path = base.parent / "BinPickingProject/vision_Notebook/runs/detect/train2/weights/best.pt"
sam_path   = base.parent / "BinPickingProject/vision_Notebook/runs/detect/SAM/sam2.1_b.pt"
rgb_path   = base / "rgb_0000.png"

# ------------------------------------------------------------
# 2. LOAD MODELS + RUN YOLO + SAM
# ------------------------------------------------------------
yolo_model = YOLO(str(model_path))
sam_model  = SAM(str(sam_path))

image = cv2.imread(str(rgb_path))
assert image is not None, f"could not read RGB image: {rgb_path}"
h, w = image.shape[:2]
print(f"image: {w}x{h}")

yolo_results = yolo_model(image)
bboxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
confs  = yolo_results[0].boxes.conf.cpu().numpy()
print(f"YOLO detected {len(bboxes)} objects")

if len(bboxes) == 0:
    raise SystemExit("no YOLO detections — nothing to segment")

# All-objects combined mask
sam_results = sam_model(image, bboxes=bboxes)
r = sam_results[0]
if r.masks is None:
    raise SystemExit("SAM returned no masks")

masks = (r.masks.data.cpu().numpy() * 255).astype(np.uint8)   # (N, H, W)
combined_mask = masks.max(axis=0)
np.save(base / "combined_mask.npy", combined_mask)            # BUG FIX: dir -> file path
print(f"saved combined_mask.npy  |  {np.count_nonzero(combined_mask)} px set")

# Top-K by confidence -> per-object masks
TOP_K = 1
top_idx = np.argsort(confs)[::-1][:TOP_K]
top_bboxes = bboxes[top_idx]
top_confs  = confs[top_idx]

sam_top = sam_model(image, bboxes=top_bboxes)
r_top = sam_top[0]
if r_top.masks is None:
    raise SystemExit("SAM (top-K) returned no masks")

top_masks = (r_top.masks.data.cpu().numpy() * 255).astype(np.uint8)   # (K, H, W)
for i in range(top_masks.shape[0]):
    out_path = base / f"top_mask_{i}.npy"                     # BUG FIX: unique name per mask
    np.save(out_path, top_masks[i])
    print(f"saved {out_path.name}  conf={top_confs[i]:.4f}  "
          f"px={np.count_nonzero(top_masks[i])}")

# ------------------------------------------------------------
# 3. LOAD DEPTH FRAME (the missing step!)
# ------------------------------------------------------------
def find_depth_file(folder: Path):
    """Find a depth image in folder. RealSense depth is 16-bit."""
    patterns = ["depth_*.png", "depth_*.npy", "depth_*.tiff", "depth_*.tif",
                "*depth*.png", "*depth*.npy", "*depth*.tiff"]
    for pat in patterns:
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    # fallback: any png that isn't the rgb
    for p in sorted(folder.glob("*.png")):
        if "rgb" not in p.name.lower() and "color" not in p.name.lower():
            return p
    return None

depth_path = find_depth_file(base)
if depth_path is None:
    print("\nAvailable files in", base)
    for p in sorted(base.iterdir()):
        print(f"  {p.name:40s} {p.stat().st_size:>12d} bytes")
    raise SystemExit(
        "No depth file found. RealSense depth is usually saved as depth_XXXX.png "
        "(16-bit). If it's still in the .bag/.rosbag, extract it first."
    )
print(f"\ndepth file: {depth_path}")

if depth_path.suffix.lower() == ".npy":
    depth = np.load(depth_path)
else:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)   # preserve 16-bit!

assert depth is not None, f"could not read {depth_path}"
print(f"depth: dtype={depth.dtype} shape={depth.shape} "
      f"min={depth.min()} max={depth.max()} std={depth.std():.2f}")

if depth.dtype != np.uint16:
    print(f"WARNING: expected uint16 (mm) depth, got {depth.dtype}. "
          f"Verify units before trusting scale.")

# ------------------------------------------------------------
# 4. APPLY MASK TO DEPTH
# ------------------------------------------------------------
mask = top_masks[0] if TOP_K == 1 else combined_mask
m = mask > 0

if m.shape != depth.shape:
    # SAM may output a different resolution than the depth frame
    m = cv2.resize(m.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                   interpolation=cv2.INTER_NEAREST) > 0
    print(f"resized mask to depth shape {depth.shape}")

depth_seg = np.where(m, depth, 0)
print(f"\nsegmented depth: nonzero={np.count_nonzero(depth_seg)} "
      f"std={depth_seg.std():.2f} max={depth_seg.max()}")

if np.count_nonzero(depth_seg) < 100:
    raise SystemExit("segmented depth is empty — mask/depth mismatch?")
if depth_seg[depth_seg > 0].std() < 1.0:
    raise SystemExit("segmented depth is constant — wrong file or units?")

# ------------------------------------------------------------
# 5. BACK-PROJECT TO POINT CLOUD
# ------------------------------------------------------------
L515 = dict(fx=1358.8, fy=1358.3, cx=971.5, cy=534.0,
            unit_scale=0.001)   # BUG FIX: mm -> m

def depth_to_partial_ply(depth_seg, camera, out_dir, name,
                         z_min=0.1, z_max=10.0, overwrite=True):
    """Back-project a segmented depth array and save it as a .ply."""
    fx, fy, cx, cy = camera["fx"], camera["fy"], camera["cx"], camera["cy"]

    H, W = depth_seg.shape
    z = depth_seg.astype(np.float32) * camera["unit_scale"]   # -> meters
    valid = np.isfinite(z) & (z > z_min) & (z < z_max)

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x[valid], y[valid], z[valid]], axis=1).astype(np.float64)

    print(f"back-projected: {len(pts)} valid points "
          f"(z range {pts[:,2].min():.3f}..{pts[:,2].max():.3f} m)" if len(pts)
          else "back-projected: 0 valid points")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.ply"
    if overwrite or not out_path.exists():
        ok = o3d.io.write_point_cloud(str(out_path), pcd)
        if not ok:
            raise IOError(f"write_point_cloud failed for {out_path}")
    return out_path

ply = depth_to_partial_ply(depth_seg, L515, base, "seg_pptc")

# ------------------------------------------------------------
# 6. SANITY GATE + VIEW
# ------------------------------------------------------------
pcd = o3d.io.read_point_cloud(str(ply))
pts = np.asarray(pcd.points)
print(f"\nwrote: {ply}  ({ply.stat().st_size} bytes)")
print(f"points: {len(pts)}")

if len(pts) == 0:
    raise SystemExit("point cloud is empty — check depth_seg before PLY")
print(f"x span: {np.ptp(pts[:,0]):.4f} m")
print(f"y span: {np.ptp(pts[:,1]):.4f} m")
print(f"z span: {np.ptp(pts[:,2]):.4f} m   std={pts[:,2].std():.4f}")

if np.ptp(pts[:,2]) < 1e-3:
    raise SystemExit("z is flat — depth frame is constant, don't sweep")

# colorize by Z for visibility
zn = (pts[:,2] - pts[:,2].min()) / (np.ptp(pts[:,2]) + 1e-9)
pcd.colors = o3d.utility.Vector3dVector(np.stack([zn, 1-zn, 0*zn], axis=1))
o3d.visualization.draw_geometries(
    [pcd], window_name="partial point cloud",
    width=1280, height=720)

# ------------------------------------------------------------
# 7. DBSCAN SWEEP
# ------------------------------------------------------------
def sweep_dbscan(ply_path, eps_list, mp_list):
    cloud = o3d.io.read_point_cloud(str(ply_path))
    print(f"\nloaded {len(cloud.points)} points from {ply_path}")
    for eps in eps_list:
        for mp in mp_list:
            labels = np.asarray(cloud.cluster_dbscan(eps=eps, min_points=mp))
            ids = labels[labels >= 0]
            n = len(set(ids))
            sizes = sorted([int((labels == l).sum()) for l in set(ids)], reverse=True)
            noise = int((labels == -1).sum())
            print(f"eps={eps:.3f} mp={mp:3d} -> {n:2d} clusters | "
                  f"noise={noise:5d} | top10={sizes[:10]}")

sweep_dbscan(ply,
             eps_list=[0.005, 0.010, 0.015, 0.020, 0.025],
             mp_list=[10, 20, 40, 60])

import numpy as np
import open3d as o3d
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from pyransac3d import Cylinder

# ------------------------------------------------------------
# 1. Cluster the cloud (your sweep winner)
# ------------------------------------------------------------
def cluster_cloud(pcd, eps=0.015, min_points=20, voxel=0.003):
    pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    return pcd, labels

# ------------------------------------------------------------
# 2. Fit a cylinder to one cluster, return a pose dict
# ------------------------------------------------------------
def fit_cylinder_pose(pts, thresh=0.01, max_iter=1000):
    """Fit a cylinder and return a pose dict (or None if fit fails)."""
    if len(pts) < 30:
        return None

    cyl = Cylinder()
    axis, center, radius, inliers = cyl.fit(pts, thresh=thresh, maxIteration=max_iter)
    if inliers is None or len(inliers) < 20 or radius <= 0:
        return None

    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return None
    axis = axis / n

    # --- sign-normalize the axis (consistent across parts) ---
    # force z >= 0; if z is ~0, force x >= 0. Pick whatever convention you need.
    if axis[2] < 0 or (abs(axis[2]) < 1e-6 and axis[0] < 0):
        axis = -axis

    center = np.asarray(center, dtype=float)
    inlier_pts = pts[inliers]

    # --- reference point: project inlier centroid onto the axis ---
    # this gives a point on the axis *at the object*, not somewhere along infinity
    centroid = inlier_pts.mean(axis=0)
    v = centroid - center
    t = np.dot(v, axis)                       # parameter along axis
    axis_point = center + t * axis            # closest axis point to centroid

    # --- build a full rotation frame ---
    # z_axis = cylinder axis; x/y chosen via a reference "up" to fix the spin.
    z_axis = axis
    # pick any vector not parallel to z_axis
    ref = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(ref, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot = np.column_stack([x_axis, y_axis, z_axis])   # 3x3, columns = axes

    quat = R.from_matrix(rot).as_quat()               # (x, y, z, w)

    return dict(
        position=axis_point,
        axis=axis,
        rotation=rot,
        quat=quat,
        radius=float(radius),
        n_inliers=int(len(inliers)),
        n_points=int(len(pts)),
        inlier_ratio=float(len(inliers) / len(pts)),
        centroid=centroid,
        ransac_center=center,
    )

# ------------------------------------------------------------
# 3. Run over every cluster, collect rows
# ------------------------------------------------------------
def process_ply(ply_path, eps=0.015, min_points=20,
                thresh=0.01, save_overlay=True):
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        print(f"[skip] empty cloud: {ply_path}")
        return []

    cloud, labels = cluster_cloud(pcd, eps=eps, min_points=min_points)
    pts_all = np.asarray(cloud.points)

    rows = []
    cluster_ids = [c for c in sorted(set(labels)) if c >= 0]
    print(f"{ply_path.name}: {len(pts_all)} pts, {len(cluster_ids)} clusters")

    overlay = o3d.geometry.PointCloud()  # for visualization
    for cid in cluster_ids:
        pts = pts_all[labels == cid]
        pose = fit_cylinder_pose(pts, thresh=thresh)
        if pose is None:
            print(f"  cluster {cid:2d} ({len(pts):5d} pts): cylinder fit FAILED")
            continue

        print(f"  cluster {cid:2d} ({len(pts):5d} pts): "
              f"r={pose['radius']:.4f} m  "
              f"inl={pose['inlier_ratio']*100:.1f}%  "
              f"axis={pose['axis'].round(3)}")

        rows.append(dict(
            file=ply_path.name,
            cluster=cid,
            n_points=pose["n_points"],
            n_inliers=pose["n_inliers"],
            inlier_ratio=pose["inlier_ratio"],
            radius_m=pose["radius"],
            # position (point on axis nearest the object centroid)
            px=pose["position"][0], py=pose["position"][1], pz=pose["position"][2],
            # centroid (raw cluster mean, for cross-check)
            cx=pose["centroid"][0], cy=pose["centroid"][1], cz=pose["centroid"][2],
            # axis direction
            ax=pose["axis"][0], ay=pose["axis"][1], az=pose["axis"][2],
            # quaternion
            qx=pose["quat"][0], qy=pose["quat"][1],
            qz=pose["quat"][2], qw=pose["quat"][3],
            # rotation matrix (row-major flattened)
            **{f"R{r}{c}": pose["rotation"][r, c]
               for r in range(3) for c in range(3)},
        ))

        if save_overlay:
            cpcd = o3d.geometry.PointCloud()
            cpcd.points = o3d.utility.Vector3dVector(pts)
            cpcd.paint_uniform_color([0.2, 0.8, 0.2])   # green
            overlay += cpcd

    if save_overlay and len(overlay.points):
        out = ply_path.with_name(f"ransac_clusters_{ply_path.stem}.ply")
        o3d.io.write_point_cloud(str(out), overlay)
        print(f"  overlay saved -> {out.name}")

    return rows

# ------------------------------------------------------------
# 4. Run + write CSV
# ------------------------------------------------------------
base = Path("C:/Users/ITSum/Desktop/RealSenseD405/l515_callibration_25")
ply_path = base / "seg_pptc.ply"     # from your earlier pipeline

rows = process_ply(ply_path, eps=0.015, min_points=20, thresh=0.01)

if rows:
    df = pd.DataFrame(rows)
    csv_path = base / f"{ply_path.stem}_poses.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}  ({len(df)} rows)")
    print(df[["cluster", "radius_m", "px", "py", "pz",
              "ax", "ay", "az", "inlier_ratio"]].to_string(index=False))
else:
    print("no cylinder fits — nothing written")