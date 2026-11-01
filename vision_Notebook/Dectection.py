"""
bin_picking_pipeline.py

Modular functions for:
  - YOLO detection + SAM segmentation
  - Depth masking + back-projection to point cloud
  - DBSCAN clustering + RANSAC cylinder pose fitting

Import these from main.py (see main.py for usage).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation as R
from pyransac3d import Cylinder
from ultralytics import YOLO, SAM


# ============================================================
# Camera model
# ============================================================
@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    unit_scale: float = 0.001   # depth units -> meters (0.001 for mm)


# Defaults measured for your L515 — override if you re-calibrate
L515 = CameraIntrinsics(fx=1358.8, fy=1358.3, cx=971.5, cy=534.0)


# ============================================================
# 1. Model loading
# ============================================================
def load_models(yolo_weights, sam_weights):
    """Load YOLO and SAM models once. Returns (yolo, sam)."""
    yolo = YOLO(str(yolo_weights))
    sam = SAM(str(sam_weights))
    return yolo, sam


# ============================================================
# 2. Detection + segmentation
# ============================================================
def detect_objects(yolo_model, image):
    """Run YOLO on a BGR image. Returns (bboxes (N,4) int, confs (N,))."""
    results = yolo_model(image)
    bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    confs = results[0].boxes.conf.cpu().numpy()
    return bboxes, confs


def segment_objects(sam_model, image, bboxes):
    """Run SAM given bboxes. Returns masks (N, H, W) uint8, values {0,255}."""
    sam_results = sam_model(image, bboxes=bboxes)
    r = sam_results[0]
    if r.masks is None:
        raise RuntimeError("SAM returned no masks")
    return (r.masks.data.cpu().numpy() * 255).astype(np.uint8)


def segment_image(yolo_model, sam_model, image, top_k=1):
    """
    Full YOLO->SAM pipeline for one frame.

    Returns
    -------
    top_masks : (K, H, W) uint8  -- per-object masks, sorted by conf desc
    top_confs : (K,)             -- matching confidences
    combined_mask : (H, W) uint8 -- pixel-wise max over the top-K masks
                                    (FIX: original used *all* masks here;
                                    now it's top-K — set top_k large for "all")
    """
    bboxes, confs = detect_objects(yolo_model, image)
    if len(bboxes) == 0:
        raise RuntimeError("no YOLO detections — nothing to segment")

    top_idx = np.argsort(confs)[::-1][:top_k]
    top_bboxes = bboxes[top_idx]
    top_confs = confs[top_idx]

    top_masks = segment_objects(sam_model, image, top_bboxes)
    combined_mask = top_masks.max(axis=0) if top_masks.size else \
        np.zeros(image.shape[:2], dtype=np.uint8)
    return top_masks, top_confs, combined_mask


# ============================================================
# 3. Depth I/O
# ============================================================
def find_depth_file(folder, exclude=("rgb", "color")):
    """Find a depth image in a folder. Returns Path or None."""
    folder = Path(folder)
    patterns = [
        "depth_*.png", "depth_*.npy", "depth_*.tiff", "depth_*.tif",
        "*depth*.png", "*depth*.npy", "*depth*.tiff",
    ]
    for pat in patterns:
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    for p in sorted(folder.glob("*.png")):
        if not any(x in p.name.lower() for x in exclude):
            return p
    return None


def load_depth(path, verbose=True):
    """Load a depth frame preserving 16-bit. Warns on non-uint16 dtypes."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        depth = np.load(path)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise IOError(f"could not read depth file: {path}")
    if verbose and depth.dtype != np.uint16:
        print(f"[depth] WARNING: expected uint16 (mm), got {depth.dtype}. "
              f"Verify units before trusting scale.")
    if verbose:
        print(f"[depth] {path.name} dtype={depth.dtype} shape={depth.shape} "
              f"min={depth.min()} max={depth.max()} std={depth.std():.2f}")
    return depth


# ============================================================
# 4. Mask -> depth -> point cloud
# ============================================================
def apply_mask_to_depth(mask, depth):
    """Set depth=0 wherever mask is 0. Resizes mask to depth resolution if needed."""
    m = mask > 0
    if m.shape != depth.shape:
        m = cv2.resize(m.astype(np.uint8),
                       (depth.shape[1], depth.shape[0]),
                       interpolation=cv2.INTER_NEAREST) > 0
    return np.where(m, depth, 0)


def depth_to_points(depth_seg, camera: CameraIntrinsics,
                    z_min=0.1, z_max=10.0):
    """Back-project a segmented depth map to (N,3) float64 points in meters."""
    fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy
    H, W = depth_seg.shape
    z = depth_seg.astype(np.float32) * camera.unit_scale
    valid = np.isfinite(z) & (z > z_min) & (z < z_max)

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x[valid], y[valid], z[valid]], axis=1).astype(np.float64)


def points_to_pcd(pts, color_by_z=False):
    """Wrap an (N,3) numpy array in an Open3D PointCloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if color_by_z and len(pts):
        zn = (pts[:, 2] - pts[:, 2].min()) / (np.ptp(pts[:, 2]) + 1e-9)
        pcd.colors = o3d.utility.Vector3dVector(
            np.stack([zn, 1 - zn, 0 * zn], axis=1))
    return pcd


def save_point_cloud(pcd, out_path, overwrite=True):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and out_path.exists():
        return out_path
    ok = o3d.io.write_point_cloud(str(out_path), pcd)
    if not ok:
        raise IOError(f"write_point_cloud failed for {out_path}")
    return out_path


def visualize_point_cloud(pcd, window_name="point cloud",
                          width=1280, height=720):
    """Blocking Open3D viewer. Skip on headless machines."""
    o3d.visualization.draw_geometries([pcd], window_name=window_name,
                                      width=width, height=height)


# ============================================================
# 5. One-shot: image + depth -> segmented PLY
# ============================================================
def process_frame(yolo_model, sam_model, image, depth, camera: CameraIntrinsics,
                  out_dir, name="frame", top_k=1,
                  z_min=0.1, z_max=10.0,
                  save_masks=True, save_ply=True, verbose=True):
    """
    End-to-end per-frame pipeline.

    Returns a dict with:
        top_masks, top_confs, combined_mask,
        depth_seg, points, pcd, ply_path (or None if save_ply=False)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(depth, (str, Path)):
        depth = load_depth(depth, verbose=verbose)

    top_masks, top_confs, combined_mask = segment_image(
        yolo_model, sam_model, image, top_k=top_k)

    if verbose:
        print(f"[segment] kept {len(top_confs)} detections (top_k={top_k}) "
              f"conf={[f'{c:.3f}' for c in top_confs]}")

    if save_masks:
        np.save(out_dir / f"{name}_combined_mask.npy", combined_mask)
        for i, m in enumerate(top_masks):
            np.save(out_dir / f"{name}_top_mask_{i}.npy", m)

    depth_seg = apply_mask_to_depth(combined_mask, depth)
    nz = int(np.count_nonzero(depth_seg))
    if verbose:
        print(f"[depth]   segmented px: {nz}")
    if nz < 100:
        raise RuntimeError("segmented depth is empty — mask/depth mismatch?")
    if depth_seg[depth_seg > 0].std() < 1.0:
        raise RuntimeError("segmented depth is constant — wrong file or units?")

    pts = depth_to_points(depth_seg, camera, z_min=z_min, z_max=z_max)
    if verbose and len(pts):
        print(f"[backproj] {len(pts)} points | z range "
              f"{pts[:,2].min():.3f}..{pts[:,2].max():.3f} m")
    if len(pts) == 0:
        raise RuntimeError("back-projection produced 0 points")

    pcd = points_to_pcd(pts, color_by_z=True)
    ply_path = None
    if save_ply:
        ply_path = save_point_cloud(pcd, out_dir / f"{name}.ply")
        if verbose:
            print(f"[save]    {ply_path} "
                  f"({ply_path.stat().st_size} bytes)")

    return dict(
        top_masks=top_masks,
        top_confs=top_confs,
        combined_mask=combined_mask,
        depth_seg=depth_seg,
        points=pts,
        pcd=pcd,
        ply_path=ply_path,
    )


# ============================================================
# 6. Clustering + cylinder pose
# ============================================================
def cluster_cloud(pcd, eps=0.015, min_points=20, voxel=0.003):
    """Voxel-downsample, remove outliers, run DBSCAN. Returns (pcd, labels)."""
    pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    return pcd, labels


def fit_cylinder_pose(pts, thresh=0.01, max_iter=1000):
    """RANSAC-fit a cylinder to a cluster. Returns a pose dict or None."""
    if len(pts) < 30:
        return None

    cyl = Cylinder()
    axis, center, radius, inliers = cyl.fit(pts, thresh=thresh,
                                            maxIteration=max_iter)
    if inliers is None or len(inliers) < 20 or radius <= 0:
        return None

    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return None
    axis = axis / n

    # sign-normalize so the pose is consistent across parts
    if axis[2] < 0 or (abs(axis[2]) < 1e-6 and axis[0] < 0):
        axis = -axis

    center = np.asarray(center, dtype=float)
    inlier_pts = pts[inliers]
    centroid = inlier_pts.mean(axis=0)

    # reference point: projection of inlier centroid onto the axis
    t = float(np.dot(centroid - center, axis))
    axis_point = center + t * axis

    # orthonormal frame with z = cylinder axis
    z_axis = axis
    ref = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(ref, z_axis); x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot = np.column_stack([x_axis, y_axis, z_axis])
    quat = R.from_matrix(rot).as_quat()

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


def process_ply(ply_path, eps=0.015, min_points=20, thresh=0.01,
                save_overlay=True, verbose=True):
    """Cluster a PLY and fit a cylinder per cluster. Returns a list of row dicts."""
    ply_path = Path(ply_path)
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        if verbose:
            print(f"[skip] empty cloud: {ply_path}")
        return []

    cloud, labels = cluster_cloud(pcd, eps=eps, min_points=min_points)
    pts_all = np.asarray(cloud.points)
    cluster_ids = [c for c in sorted(set(labels)) if c >= 0]

    if verbose:
        print(f"{ply_path.name}: {len(pts_all)} pts, "
              f"{len(cluster_ids)} clusters")

    rows = []
    overlay = o3d.geometry.PointCloud()

    for cid in cluster_ids:
        pts = pts_all[labels == cid]
        pose = fit_cylinder_pose(pts, thresh=thresh)
        if pose is None:
            if verbose:
                print(f"  cluster {cid:2d} ({len(pts):5d} pts): fit FAILED")
            continue

        if verbose:
            print(f"  cluster {cid:2d} ({len(pts):5d} pts): "
                  f"r={pose['radius']:.4f} m  "
                  f"inl={pose['inlier_ratio']*100:.1f}%  "
                  f"axis={pose['axis'].round(3)}")

        row = dict(
            file=ply_path.name, cluster=cid,
            n_points=pose["n_points"], n_inliers=pose["n_inliers"],
            inlier_ratio=pose["inlier_ratio"], radius_m=pose["radius"],
            px=pose["position"][0], py=pose["position"][1],
            pz=pose["position"][2],
            cx=pose["centroid"][0], cy=pose["centroid"][1],
            cz=pose["centroid"][2],
            ax=pose["axis"][0], ay=pose["axis"][1], az=pose["axis"][2],
            qx=pose["quat"][0], qy=pose["quat"][1],
            qz=pose["quat"][2], qw=pose["quat"][3],
        )
        for r in range(3):
            for c in range(3):
                row[f"R{r}{c}"] = pose["rotation"][r, c]
        rows.append(row)

        if save_overlay:
            cpcd = o3d.geometry.PointCloud()
            cpcd.points = o3d.utility.Vector3dVector(pts)
            cpcd.paint_uniform_color([0.2, 0.8, 0.2])
            overlay += cpcd

    if save_overlay and len(overlay.points):
        out = ply_path.with_name(f"ransac_clusters_{ply_path.stem}.ply")
        o3d.io.write_point_cloud(str(out), overlay)
        if verbose:
            print(f"  overlay -> {out.name}")

    return rows


# ============================================================
# 7. DBSCAN sweep (for tuning eps/min_points)
# ============================================================
def sweep_dbscan(ply_path, eps_list, mp_list, voxel=0.003):
    """
    Try combinations of eps/min_points. Returns a list of result dicts.

    FIX: now applies the same preprocessing as cluster_cloud (voxel +
    statistical outlier removal), so the eps/mp you pick here transfers
    directly to process_ply(). Set voxel=0 to sweep the raw cloud.
    """
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        raise RuntimeError(f"empty cloud: {ply_path}")
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    results = []
    for eps in eps_list:
        for mp in mp_list:
            labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=mp))
            ids = [int(l) for l in set(labels) if l >= 0]
            sizes = sorted([int((labels == l).sum()) for l in ids], reverse=True)
            noise = int((labels == -1).sum())
            results.append(dict(
                eps=eps, min_points=mp,
                n_clusters=len(ids), noise=noise,
                sizes_top10=sizes[:10],
            ))
    return results


def print_sweep(results):
    for r in results:
        print(f"eps={r['eps']:.3f} mp={r['min_points']:3d} -> "
              f"{r['n_clusters']:2d} clusters | noise={r['noise']:5d} | "
              f"top10={r['sizes_top10']}")


# ============================================================
# 8. Save poses
# ============================================================
def save_poses_to_csv(rows, csv_path):
    """Write pose rows to CSV. Returns (df, csv_path_or_None)."""
    csv_path = Path(csv_path)
    df = pd.DataFrame(rows)
    if df.empty:
        print("[csv] no rows — nothing written")
        return df, None
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[csv] wrote {csv_path} ({len(df)} rows)")
    return df, csv_path


# ============================================================
# 9. Optional convenience: full pipeline end-to-end
# ============================================================
def run_full_pipeline(base_dir, yolo_weights, sam_weights,
                      rgb_path=None, depth_path=None,
                      camera: CameraIntrinsics = L515,
                      name="frame", top_k=1,
                      eps=0.015, min_points=20, ransac_thresh=0.01,
                      do_sweep=False, save_overlay=True,
                      visualize=False):
    """
    Runs the entire pipeline and returns a summary dict.
    Use individual functions instead if you need finer control.
    """
    base = Path(base_dir)
    rgb_path = Path(rgb_path) if rgb_path else base / "rgb_0000.png"

    image = cv2.imread(str(rgb_path))
    if image is None:
        raise IOError(f"could not read {rgb_path}")

    if depth_path is None:
        depth_path = find_depth_file(base)
    if depth_path is None:
        raise IOError(f"no depth file in {base}")
    depth = load_depth(depth_path)

    yolo, sam = load_models(yolo_weights, sam_weights)

    frame = process_frame(yolo, sam, image, depth, camera,
                          out_dir=base, name=name, top_k=top_k)
    ply_path = frame["ply_path"]

    if visualize:
        visualize_point_cloud(frame["pcd"])

    sweep = None
    if do_sweep:
        sweep = sweep_dbscan(ply_path,
                             eps_list=[0.005, 0.010, 0.015, 0.020, 0.025],
                             mp_list=[10, 20, 40, 60])
        print_sweep(sweep)

    rows = process_ply(ply_path, eps=eps, min_points=min_points,
                       thresh=ransac_thresh, save_overlay=save_overlay)
    df, csv_path = save_poses_to_csv(
        rows, base / f"{ply_path.stem}_poses.csv")

    return dict(
        ply_path=ply_path,
        sweep=sweep,
        poses_df=df,
        poses_csv=csv_path,
        frame=frame,
    )