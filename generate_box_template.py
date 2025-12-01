import open3d as o3d
import numpy as np

def generate_box_pcd(height=0.20, width=0.07, depth=0.07,
                     n_points=5000, mode='full'):
    """
    Generates a point cloud box with identical center alignment
    for both full box and sliced version.

    Units: meters
    mode:
        'full' → full 3D solid box sampled
        'face' → only one front slice (scan-like)
    """

    print(f"Generating Box: {height}m x {width}m x {depth}m | Mode={mode}")

    # ==============================================================
    # 1. Generate RAW (not centered) FULL box first
    # ==============================================================
    mesh_box = o3d.geometry.TriangleMesh.create_box(
        width=width, height=height, depth=depth
    )
    mesh_box.compute_vertex_normals()

    # Sample full box first (we will slice this one if needed)
    pcd_full_raw = mesh_box.sample_points_poisson_disk(number_of_points=n_points)

    # Convert to numpy to compute center
    raw_full_pts = np.asarray(pcd_full_raw.points)
    center_shift = raw_full_pts.mean(axis=0)

    # ==============================================================
    # 2. If slice mode → filter BEFORE centering, then shift SAME way
    # ==============================================================
    if mode == 'face':
        pts = raw_full_pts  # raw uncentered points
        
        # Keep only points within front 20% of depth (like original code)
        z_max = pts[:, 2].max()
        mask = pts[:, 2] < (0.2 * z_max)
        sliced = pts[mask]

        # Apply same center shift as full box
        sliced_centered = sliced - center_shift
        print(f" > Slice reduced to {sliced_centered.shape[0]} points")

        pcd_slice = o3d.geometry.PointCloud()
        pcd_slice.points = o3d.utility.Vector3dVector(sliced_centered)
        pcd_slice.estimate_normals()
        return pcd_slice

    if mode == 'quarter':
        pts = raw_full_pts

        # Bounding box limits (uncentered)
        x_min, y_min, z_min = pts.min(axis=0)
        x_max, y_max, z_max = pts.max(axis=0)

        # Keep half width (X) and half depth (Z), but full height (Y)
        mask = (
            (pts[:, 0] < x_min + 0.5 * (x_max - x_min)) &
            (pts[:, 2] <= z_min + 0.5 * (z_max - z_min)) 
        )
        
        sliced = pts[mask]

        # Apply same center shift so the slice matches full template
        sliced_centered = sliced - center_shift

        print(f" > Quarter vertical slice: {sliced_centered.shape[0]} points")

        pcd_qtr = o3d.geometry.PointCloud()
        pcd_qtr.points = o3d.utility.Vector3dVector(sliced_centered)
        pcd_qtr.estimate_normals()
        return pcd_qtr


    # ============================================================

    # ==============================================================
    # 3. If full mode → center normally
    # ==============================================================
    centered = raw_full_pts - center_shift
    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(centered)
    pcd_full.estimate_normals()

    return pcd_full

# --- CONFIGURATION ---
HEIGHT = 0.20  # 20 cm
WIDTH  = 0.07  # 7 cm
LENGTH = 0.07  # 7 cm (Depth)

# OUTPUT FILES
FILENAME_FULL = "data/DellValleMaca_template.pcd"
FILENAME_FACE = "data/DellValleMaca_template_slice.pcd"
FILENAME_QUARTER = "data/DellValleMacaRotated_template_slice.pcd"

# --- GENERATE FULL TEMPLATE ---
pcd_full = generate_box_pcd(HEIGHT, WIDTH, LENGTH, n_points=5000, mode='full')
pcd_full.paint_uniform_color([0, 1, 0]) # Green
o3d.io.write_point_cloud(FILENAME_FULL, pcd_full)
print(f"Saved {FILENAME_FULL}")

# --- GENERATE SINGLE FACE (Simulating a Scan) ---
pcd_face = generate_box_pcd(HEIGHT, WIDTH, LENGTH, n_points=2000, mode='face')
pcd_face.paint_uniform_color([1, 0, 0]) # Red
o3d.io.write_point_cloud(FILENAME_FACE, pcd_face)
print(f"Saved {FILENAME_FACE}")

# --- GENERATE SINGLE FACE (Simulating a Scan) ---
pcd_quarter = generate_box_pcd(HEIGHT, WIDTH, LENGTH, n_points=2000, mode='quarter')
pcd_quarter.paint_uniform_color([0, 0, 1]) # Blue
o3d.io.write_point_cloud(FILENAME_QUARTER, pcd_quarter)
print(f"Saved {FILENAME_QUARTER}")

# --- VISUALIZE BOTH ---
# We translate one so they don't overlap in the viewer
pcd_face.translate([0.15, 0, 0]) 
pcd_quarter.translate([0.3, 0, 0]) 

o3d.visualization.draw_geometries(
    [pcd_full, pcd_face, pcd_quarter], 
    window_name="Full Box | Single Face | Corner Quarter"
)