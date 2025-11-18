import numpy as np
import pandas as pd

SLICE = True  # Set to True to generate a slice of the cylinder

# Cylinder parameters
height = 11.40 / 2  # mm
diameter = 6.60 / 2  # mm
radius = diameter / 2

# Number of points
n_circle = 2000   # around the side surface
n_top = 1000      # top disk
n_bottom = 1000   # bottom disk

# --- Generate side surface points ---
theta = np.random.uniform(0, 2 * np.pi, n_circle)
z = np.random.uniform(0, height, n_circle)
x = radius * np.cos(theta)
y = radius * np.sin(theta)
side_points = np.column_stack((x, z, y))  # Swap z and y

# --- Generate top disk points ---
r = radius * np.sqrt(np.random.uniform(0, 1, n_top))
theta = np.random.uniform(0, 2 * np.pi, n_top)
x = r * np.cos(theta)
y = r * np.sin(theta)
z = np.full(n_top, height)
top_points = np.column_stack((x, z, y))  # Swap z and y

# --- Generate bottom disk points ---
r = radius * np.sqrt(np.random.uniform(0, 1, n_bottom))
theta = np.random.uniform(0, 2 * np.pi, n_bottom)
x = r * np.cos(theta)
y = r * np.sin(theta)
z = np.zeros(n_bottom)
bottom_points = np.column_stack((x, z, y))  # Swap z and y

# Combine
points = np.vstack((side_points, top_points, bottom_points))

if SLICE:
    # Filter points to keep only a vertical slice of 1/3 of the cylinder
    slice_angle = 2 * np.pi / 3  # 1/3 of the full circle
    slice_mask = np.arctan2(points[:, 2], points[:, 0]) % (2 * np.pi) < slice_angle
    points = points[slice_mask]

# Save as ASCII PCD
pcd_header = (
    "VERSION .7\n"
    "FIELDS x y z\n"
    "SIZE 4 4 4\n"
    "TYPE F F F\n"
    "COUNT 1 1 1\n"
    f"WIDTH {points.shape[0]}\n"
    "HEIGHT 1\n"
    "VIEWPOINT 0 0 0 1 0 0 0\n"
    f"POINTS {points.shape[0]}\n"
    "DATA ascii\n"
)

pcd_path = "data/stella_template.pcd" if not SLICE else "data/stella_template_slice.pcd"
with open(pcd_path, "w") as f:
    f.write(pcd_header)
    for p in points:
        f.write(f"{p[0]} {p[1]} {p[2]}\n")

pcd_path
