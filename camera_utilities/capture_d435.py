"""
Capture aligned RGB + depth images from an Intel RealSense D435/D435i,
generate colored point clouds, and view them in 3D with Open3D.
 
Setup:
    pip install pyrealsense2 opencv-python numpy open3d
 
Usage:
    python capture_d435.py
 
Keys (with the OpenCV windows focused):
    SPACE : save current frame (rgb .png, depth .png/.npy, point cloud .ply)
    P     : open the current frame's point cloud in an Open3D 3D viewer
    E     : toggle the IR dot projector (emitter) on/off
    Q/ESC : quit
 
Saves to captures_d435/:
    rgb_XXXX.png      8-bit color
    depth_XXXX.png    16-bit raw depth (multiply by depth scale for meters)
    depth_XXXX.npy    raw depth array
    cloud_XXXX.ply    colored point cloud (meters, camera frame)
 
D435 notes (vs. D405):
  - Separate RGB sensor -> depth/color alignment is essential (done below).
  - Built-in IR dot projector -> good depth on textureless surfaces.
  - Default depth units: 1 mm (depth scale 0.001).
  - Working range ~0.3 m to several meters; ideal for a bin at 0.5-1.0 m.
"""
 
import os
import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d

 
SAVE_DIR ="calibrate_d435_25"
os.makedirs(SAVE_DIR, exist_ok=True)
 
MAX_DEPTH_M = 1.2   # truncate points beyond this (just past bin-floor distance)
 
# --- Configure pipeline -----------------------------------------------------
pipeline = rs.pipeline()
config = rs.config()
 
# 848x480 is the D435's optimal depth resolution (best accuracy per Intel).
# RGB runs at 1280x720; depth is aligned (reprojected) onto the color frame.
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
 
profile = pipeline.start(config)
device = profile.get_device()
depth_sensor = device.first_depth_sensor()
 
# Depth scale: multiply raw depth by this to get meters (D435 default: 0.001)
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth scale: {depth_scale} m/unit")
 
# Make sure the IR dot projector is ON and at full power -- this is what
# gives the D435 clean depth on smooth/textureless parts.
emitter_on = True
if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 1)
if depth_sensor.supports(rs.option.laser_power):
    rng = depth_sensor.get_option_range(rs.option.laser_power)
    depth_sensor.set_option(rs.option.laser_power, rng.max)
    print(f"Laser power set to {rng.max}")
 
# Align depth to the color frame (critical on D435: RGB is a separate sensor
# a few cm from the depth module, so unaligned streams do NOT correspond).
align = rs.align(rs.stream.color)
 
# Intrinsics of the COLOR stream (correct ones for the aligned depth)
intr = profile.get_stream(rs.stream.color) \
              .as_video_stream_profile().get_intrinsics()
o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
)
print(f"Intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} "
      f"cx={intr.ppx:.1f} cy={intr.ppy:.1f}")
 
# Post-processing filters
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_fill = rs.hole_filling_filter()
colorizer = rs.colorizer()
 
 
def make_point_cloud(color_bgr, depth_raw):
    """Build a colored Open3D point cloud from aligned RGB + depth arrays."""
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(rgb)),
        o3d.geometry.Image(depth_raw),
        depth_scale=1.0 / depth_scale,   # raw units per meter (D435: 1000)
        depth_trunc=MAX_DEPTH_M,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics)
    pcd.transform([[1, 0, 0, 0],        # flip for upright display
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    return pcd
 
 
def view_point_cloud(pcd):
    """Blocking Open3D window. Mouse: rotate; wheel: zoom; Shift+drag: pan."""
    shown, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    frame_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries(
        [shown, frame_axes],
        window_name="D435 Point Cloud (close window to resume)",
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
        elif key == ord("e"):
            emitter_on = not emitter_on
            depth_sensor.set_option(rs.option.emitter_enabled,
                                    1 if emitter_on else 0)
            print(f"IR projector {'ON' if emitter_on else 'OFF'}")
        elif key in (ord("q"), 27):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
 