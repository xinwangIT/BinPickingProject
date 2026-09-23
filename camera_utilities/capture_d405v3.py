"""
Capture aligned RGB + depth images from an Intel RealSense D405,
generate colored point clouds, and view them in 3D with Open3D.

Setup:
    pip install pyrealsense2 opencv-python numpy open3d

Usage:
    python capture_d405v3.py

Keys (with the OpenCV windows focused):
    SPACE : save current frame (rgb .png, depth .png/.npy, point cloud .ply)
    P     : open the current frame's point cloud in an Open3D 3D viewer
    Q/ESC : quit

Saves to captures/:
    rgb_XXXX.png      8-bit color
    depth_XXXX.png    16-bit raw depth (multiply by depth scale for meters)
    depth_XXXX.npy    raw depth array
    cloud_XXXX.ply    colored point cloud (meters, camera frame)
"""

import os
import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d

SAVE_DIR = "data_photo_1"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Configure pipeline -----------------------------------------------------
pipeline = rs.pipeline()
config = rs.config()

# D405: depth and color both come from the stereo module (color = left imager)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

profile = pipeline.start(config)

# Depth scale: multiply raw depth by this to get meters.
# NOTE: D405 default is 0.0001 (100 um units), not 0.001 like D415/D435.
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
print(f"Depth scale: {depth_scale} m/unit")

# Align depth to the color frame so pixels correspond 1:1
align = rs.align(rs.stream.color)

# Intrinsics of the COLOR stream (depth is aligned to it, so these are the
# correct intrinsics for deprojecting the aligned depth into 3D).
intr = profile.get_stream(rs.stream.color) \
              .as_video_stream_profile().get_intrinsics()
o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
)
print(f"Intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} "
      f"cx={intr.ppx:.1f} cy={intr.ppy:.1f}")

# Post-processing filters (reduce speckle noise on small parts)
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_fill = rs.hole_filling_filter()
colorizer = rs.colorizer()  # histogram-equalized depth visualization


def make_point_cloud(color_bgr, depth_raw):
    """Build a colored Open3D point cloud from aligned RGB + depth arrays."""
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(rgb)),
        o3d.geometry.Image(depth_raw),
        depth_scale=1.0 / depth_scale,   # raw units per meter (D405: 10000)
        depth_trunc=0.6,                 # ignore points beyond 0.6 m
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics)
    # Flip so the cloud appears upright in the Open3D viewer
    # (camera convention: +Z forward, +Y down -> viewer: +Y up)
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    return pcd


def view_point_cloud(pcd):
    """Open a blocking Open3D window. Mouse: rotate; wheel: zoom;
    Shift+drag: pan. Close the window to return to the live view."""
    # Light denoise for nicer viewing (keeps the saved cloud untouched)
    shown, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    frame_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    o3d.visualization.draw_geometries(
        [shown, frame_axes],
        window_name="D405 Point Cloud (close window to resume)",
        width=1280, height=720,
    )


frame_idx = 0
try:
    # Let auto-exposure settle
    for _ in range(15):
        pipeline.wait_for_frames()

    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = hole_fill.process(depth_frame)

        depth = np.asanyarray(depth_frame.get_data())   # uint16
        color = np.asanyarray(color_frame.get_data())   # uint8 BGR

        # Histogram-equalized depth visualization (auto-scales to scene)
        depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        cv2.imshow("RGB", color)
        cv2.imshow("Depth", depth_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            ts = f"{frame_idx:04d}"
            cv2.imwrite(f"{SAVE_DIR}/rgb_{ts}.png", color)
            cv2.imwrite(f"{SAVE_DIR}/depth_{ts}.png", depth)  # 16-bit PNG
            np.save(f"{SAVE_DIR}/depth_{ts}.npy", depth)
            pcd = make_point_cloud(color, depth)
            o3d.io.write_point_cloud(f"{SAVE_DIR}/cloud_{ts}.ply", pcd)
            print(f"Saved frame {ts}: {len(pcd.points)} points "
                  f"(median depth "
                  f"{np.median(depth[depth > 0]) * depth_scale:.3f} m)")
            frame_idx += 1
        elif key == ord("p"):
            print("Opening 3D viewer... close its window to resume capture.")
            view_point_cloud(make_point_cloud(color, depth))
        elif key in (ord("q"), 27):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()