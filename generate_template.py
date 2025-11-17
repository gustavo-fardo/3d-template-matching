import numpy as np
import pandas as pd

# Cylinder parameters
height = 114.0  # mm
diameter = 66.0  # mm
radius = diameter / 2

# Number of points
n_circle = 2000   # around the side surface
n_top = 1000      # top disk
n_bottom = 1000   # bottom disk

# --- Generate side surface points ---
theta = np.random.uniform(0, 2*np.pi, n_circle)
z = np.random.uniform(0, height, n_circle)
x = radius * np.cos(theta)
y = radius * np.sin(theta)
side_points = np.column_stack((x, y, z))

# --- Generate top disk points ---
r = radius * np.sqrt(np.random.uniform(0, 1, n_top))
theta = np.random.uniform(0, 2*np.pi, n_top)
x = r * np.cos(theta)
y = r * np.sin(theta)
z = np.full(n_top, height)
top_points = np.column_stack((x, y, z))

# --- Generate bottom disk points ---
r = radius * np.sqrt(np.random.uniform(0, 1, n_bottom))
theta = np.random.uniform(0, 2*np.pi, n_bottom)
x = r * np.cos(theta)
y = r * np.sin(theta)
z = np.zeros(n_bottom)
bottom_points = np.column_stack((x, y, z))

# Combine
points = np.vstack((side_points, top_points, bottom_points))

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

pcd_path = "stella_template.pcd"
with open(pcd_path, "w") as f:
    f.write(pcd_header)
    for p in points:
        f.write(f"{p[0]} {p[1]} {p[2]}\n")

pcd_path
