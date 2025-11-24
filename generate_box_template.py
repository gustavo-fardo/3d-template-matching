import open3d as o3d
import numpy as np

def generate_box_pcd(height=0.20, width=0.07, depth=0.07, n_points=5000, mode='full'):
    """
    Generates a point cloud box.
    Units: Meters (0.20 = 20cm)
    Mode: 'full' (Solid 3D box) or 'face' (Just the front panel)
    """
    print(f"Generating Box: {height}m x {width}m x {depth}m")

    # 1. Create a Solid Mesh Box (Open3D primitive)
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
    
    # 2. Compute Normals (needed for consistent point sampling)
    mesh_box.compute_vertex_normals()
    
    # 3. Sample Points (Turn Mesh into Point Cloud)
    # Poisson Disk Sampling gives a nice, even distribution of points
    pcd = mesh_box.sample_points_poisson_disk(number_of_points=n_points)
    
    # 4. (Optional) Filter for a Single Face
    if mode == 'face':
        # Get points as numpy array
        pts = np.asarray(pcd.points)
        
        # Logic: Keep only points that are in the front 20% of the depth
        # The box is created from (0,0,0) to (w, h, d).
        # Let's keep points where depth (z) is less than 20% of the max depth
        z_max = pts[:, 2].max()
        
        # Keep points within the front 20% of the depth
        mask = pts[:, 2] < (0.2 * z_max)
        
        pcd = pcd.select_by_index(np.where(mask)[0])
        print(f" > Mode 'face' selected. Points reduced to: {len(pcd.points)}")

    # 5. Center the cloud (Good practice for templates)
    pts = np.asarray(pcd.points)
    pts -= pts.mean(axis=0)
    pcd.points = o3d.utility.Vector3dVector(pts)
    
    # Estimate normals for the new PCD (crucial for registration later)
    pcd.estimate_normals()
    
    return pcd

# --- CONFIGURATION ---
HEIGHT = 20/2  # 20 cm
WIDTH  = 7/2  # 7 cm
LENGTH = 7/2  # 7 cm (Depth)

# OUTPUT FILES
FILENAME_FULL = "data/dellvallemaca_template.pcd"
FILENAME_FACE = "data/dellvallemaca_template_slice.pcd"

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

# --- VISUALIZE BOTH ---
# We translate one so they don't overlap in the viewer
pcd_face.translate([0.15, 0, 0]) 

o3d.visualization.draw_geometries(
    [pcd_full, pcd_face], 
    window_name="Left: Full Box | Right: Single Face"
)