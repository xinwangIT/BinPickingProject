import numpy as np
import cv2

# Replace these with your actual recorded points!
# Format: [Pixel_X, Pixel_Y]
cam_points = np.array([
    [460, 1057], 
    [431, 154], 
    [1647, 130], 
    [1664, 1067]
], dtype=np.float32)

# Format: [Robot_X, Robot_Y]
robot_points = np.array([
    [-138.35, -12.37], 
    [-332.95, -13.17], 
    [-327.11, 237.70], 
    [-145.65, 231.16]
], dtype=np.float32)

# Find the Homography (Perspective Transform)
# This handles rotation, scaling, and axis swapping perfectly
matrix, _ = cv2.findHomography(cam_points, robot_points)

def pixel_to_robot(u, v):
    # Convert single pixel point to robot coordinate
    point = np.array([[[u, v]]], dtype=np.float32)
    robot_coord = cv2.perspectiveTransform(point, matrix)
    return robot_coord[0][0][0], robot_coord[0][0][1]

# --- TEST IT ---
test_u, test_v = 1497.0, 357.5  # Center of image
x, y = pixel_to_robot(test_u, test_v)
print(f"Pixel ({test_u}, {test_v}) -> Robot ({x:.2f}, {y:.2f})")