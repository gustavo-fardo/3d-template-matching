import numpy as np
import open3d as o3d

def generate_cylinder_pcd(height=0.114, diameter=0.066,
                          n_circle=2000, n_top=1000, n_bottom=1000,
                          mode='full'):
    """
    Generates a cylinder point cloud, with matching center for both
    full and sliced versions.

    mode:
        'full' → entire cylinder
        'face' → 1/3 vertical slice (simulated scan)
    """

    print(f"Generating Cylinder: H={height}m, D={diameter}m, Mode={mode}")

    radius = diameter / 2

    # ============================================================
    # 1) Generate RAW full cylinder (NOT centered yet)
    # ============================================================
    # ---- Side surface ----
    theta = np.random.uniform(0, 2*np.pi, n_circle)
    z = np.random.uniform(0, height, n_circle)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    side = np.column_stack((x, y, z))

    # ---- Top disk ----
    r = radius * np.sqrt(np.random.uniform(0, 1, n_top))
    theta = np.random.uniform(0, 2*np.pi, n_top)
    x = r * np.cos(theta); y = r * np.sin(theta)
    z = np.full(n_top, height)
    top = np.column_stack((x, y, z))

    # ---- Bottom disk ----
    r = radius * np.sqrt(np.random.uniform(0, 1, n_bottom))
    theta = np.random.uniform(0, 2*np.pi, n_bottom)
    x = r * np.cos(theta); y = r * np.sin(theta)
    z = np.zeros(n_bottom)
    bottom = np.column_stack((x, y, z))

    raw_full = np.vstack((side, top, bottom))

    # ============================================================
    # 2) Apply SAME center shift to both full & slice
    # ============================================================
    center_shift = raw_full.mean(axis=0)

    # Center full cylinder
    full_centered = raw_full - center_shift

    # If slice mode → filter BEFORE centering, then shift identically
    if mode == 'face':
        slice_angle = 2*np.pi/3  # keep 1/3 of cylinder
        mask = np.arctan2(raw_full[:, 1], raw_full[:, 0]) % (2*np.pi) < slice_angle
        raw_slice = raw_full[mask]
        points = raw_slice - center_shift
        print(f" > Slice reduced to {points.shape[0]} points")
    else:
        points = full_centered

    # ============================================================
    # 3) Convert to Open3D + compute normals
    # ============================================================
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()

    return pcd

# -----------------------------
# CONFIGURATION (mm to meters)
# -----------------------------
HEIGHT   = 0.140     # mm → m (half height)
DIAMETER = 0.053     # mm → m (half diameter)

# -----------------------------
# OUTPUT FILES
# -----------------------------
FILENAME_FULL = "data/Stella_template.pcd"
FILENAME_FACE = "data/Stella_template_slice.pcd"

# -----------------------------
# GENERATE FULL CYLINDER
# -----------------------------
pcd_full = generate_cylinder_pcd(HEIGHT, DIAMETER, mode='full')
pcd_full.paint_uniform_color([0, 1, 0])  # Green
o3d.io.write_point_cloud(FILENAME_FULL, pcd_full)
print(f"Saved {FILENAME_FULL}")

# -----------------------------
# GENERATE SCANNED SLICE
# -----------------------------
pcd_face = generate_cylinder_pcd(HEIGHT, DIAMETER, mode='face')
pcd_face.paint_uniform_color([1, 0, 0])  # Red
o3d.io.write_point_cloud(FILENAME_FACE, pcd_face)
print(f"Saved {FILENAME_FACE}")

# -----------------------------
# VISUALIZE BOTH (Side by Side)
# -----------------------------
pcd_face.translate([0.15, 0, 0])  # shift slice to the right
o3d.visualization.draw_geometries(
    [pcd_full, pcd_face],
    window_name="Left: Full Cylinder | Right: Slice Scan"
)