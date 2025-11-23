import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy

def preprocess_pcd(pcd, voxel_size):
    """Downsample + estimate normals + compute FPFH."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100)
    )
    return pcd_down, fpfh

def center_cloud(pcd):
    """Translate cloud so its centroid is at the origin."""
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    pts -= centroid
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def pca_align(pcd):
    """Rotate cloud using PCA so major axes align with XYZ."""
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid

    H = pts_centered.T @ pts_centered
    eigvals, eigvecs = np.linalg.eigh(H)

    # sort eigenvectors by eigenvalue (largest first)
    R = eigvecs[:, ::-1]

    pts_rot = pts_centered @ R
    aligned = o3d.geometry.PointCloud()
    aligned.points = o3d.utility.Vector3dVector(pts_rot)

    return aligned, R

# --- 2. PAPER METHODOLOGY: STEP 1 - RESAMPLING & THRESHOLDS ---
def get_hard_constraints(template, voxel_size):
    """
    Implements Section 3.1: 'Target point cloud resampling to set similarity thresholds'
    We mesh the template, resample it to 50% points, and register it back to itself.
    The resulting Fitness and RMSE become our 'Hard Constraints' for acceptance.
    """
    print("--- Step 1: Generating Hard Constraints (Self-Resampling) ---")
    
    # A. Create a Mesh (Approximating the Tetra-mesh/Delaunay step in the paper)
    # Using Ball Pivoting or Poisson for surface reconstruction
    template.estimate_normals()
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(template, depth=8)
    
    # B. Resample (Poisson Disk Sampling) [cite: 1311]
    # The paper suggests using 50% of the original point count
    n_points = int(len(template.points) * 0.5) 
    resampled_template = mesh.sample_points_poisson_disk(n_points)
    
    # C. Self-Registration (Template vs Resampled Template)
    # We use Point-to-Plane ICP as the 'Gold Standard' comparison here
    threshold = voxel_size * 1.5
    trans_init = np.eye(4) # Already aligned
    
    reg_p2l = o3d.pipelines.registration.registration_icp(
        resampled_template, template, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    T = reg_p2l.transformation
    template_transformed = template.transform(T)
    template_col = template_transformed.paint_uniform_color([1, 0, 0])  # red
    template_orig_col = template.paint_uniform_color([0, 1, 0])          # green
    o3d.visualization.draw_geometries([template_col, template_orig_col])

    # D. Extract Thresholds
    # The paper says this establishes the "object specificity threshold" [cite: 1495]
    min_fitness = reg_p2l.fitness
    max_rmse = reg_p2l.inlier_rmse
    
    # Apply a small safety margin (e.g., 10% tolerance)
    final_fitness_thresh = min_fitness * 0.90
    final_rmse_thresh = max_rmse * 1.10
    
    print(f"  > Self-Fitness: {min_fitness:.4f} -> Threshold: {final_fitness_thresh:.4f}")
    print(f"  > Self-RMSE:    {max_rmse:.4f}  -> Threshold: {final_rmse_thresh:.4f}")
    
    return final_fitness_thresh, final_rmse_thresh

def get_matrix(R=np.eye(3), T=np.zeros(3)):
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = T
    return mat

# --- MAIN EXECUTION ---

IMAGE = "data/Stella_xtion_1.png"
TEMPLATE = "data/stella_template_slice.pcd"
VOXEL_SIZE = 0.3   
RANSAC_DIST = VOXEL_SIZE * 1.5
ICP_DIST = VOXEL_SIZE * 3.0  # Very loose for the first ICP pass

##############################
# Load depth image ###########
##############################

depth_image = o3d.io.read_image(IMAGE)

intrinsic = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
)

pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, intrinsic)

o3d.visualization.draw_geometries([pcd])

##############################
# Load template ##############
##############################

template = o3d.io.read_point_cloud(TEMPLATE)   # reference model
centroid_template = np.asarray(template.points).mean(axis=0)
o3d.visualization.draw_geometries([template, pcd])

##############################
# Calculate min thresh #######
##############################
MIN_FITNESS, MAX_RMSE = get_hard_constraints(template, VOXEL_SIZE)

##############################
# Crop point cloud ###########
##############################

# Load YOLO bounding box
txt_file = IMAGE.rsplit(".", 1)[0] + ".txt"
cls_id, xc, yc, w, h = map(float, open(txt_file).read().split())

# --- Get image width/height ---
img_np = np.asarray(depth_image)
H, W = img_np.shape[:2]     # <-- Works on every Open3D version

# Convert YOLO → pixel coords
xmin = int((xc - w/2) * W)
xmax = int((xc + w/2) * W)
ymin = int((yc - h/2) * H)
ymax = int((yc + h/2) * H)

# --- Project point cloud to image ---
points = np.asarray(pcd.points)

X, Y, Z = points[:,0], points[:,1], points[:,2]

fx, fy = intrinsic.intrinsic_matrix[0,0], intrinsic.intrinsic_matrix[1,1]
cx, cy = intrinsic.intrinsic_matrix[0,2], intrinsic.intrinsic_matrix[1,2]

u = (X * fx / Z) + cx
v = (Y * fy / Z) + cy

# Mask
mask = (Z > 0) & \
       (u >= xmin) & (u <= xmax) & \
       (v >= ymin) & (v <= ymax)

cropped_points = points[mask]

cropped = o3d.geometry.PointCloud()
cropped.points = o3d.utility.Vector3dVector(cropped_points)

o3d.visualization.draw_geometries([cropped])
pcd = cropped

##############################
# Find clusters ##############
##############################

labels = np.array(
    pcd.cluster_dbscan(eps=2.00, min_points=500, print_progress=True)
)

print("Cluster labels:", labels)
print("Number of clusters:", len(set(labels)) - (1 if -1 in labels else 0))

# --- Color each cluster ---
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = [0, 0, 0, 1]  # noise points colored black

pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])  # keep RGB only

o3d.visualization.draw_geometries([pcd])

for cluster_label in range(max_label + 1):
    cluster_points = np.asarray(pcd.points)[labels == cluster_label]

    # Create a new point cloud for the selected cluster
    cluster = o3d.geometry.PointCloud()
    cluster.points = o3d.utility.Vector3dVector(cluster_points)
    centroid_cluster = cluster_points.mean(axis=0)

    template_working = copy.deepcopy(template)
    cluster_working  = copy.deepcopy(cluster)

    ##############################
    # Preprocess #################
    ##############################

    template_working = center_cloud(template_working)
    cluster_working  = center_cloud(cluster_working)

    template_aligned, R_t = pca_align(template_working)
    cluster_aligned,  R_c = pca_align(cluster_working)

    template_aligned.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
    )
    cluster_aligned.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
    )

    # Preprocess the template point cloud
    template_down, template_fpfh = preprocess_pcd(template_aligned, VOXEL_SIZE)
    cluster_down, cluster_fpfh   = preprocess_pcd(cluster_aligned, VOXEL_SIZE)

    ##############################
    # 1sr Register RANSAC ########
    ##############################

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        template_down, cluster_down,
        template_fpfh, cluster_fpfh,
        mutual_filter=True,
        max_correspondence_distance=RANSAC_DIST,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(RANSAC_DIST),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=400000,
            confidence=0.999
        )
    )

    cluster_col  = cluster.paint_uniform_color([0, 1, 0])               # green

    print(result_ransac)
    T = result_ransac.transformation
    template_transformed = template.transform(T)
    template_col = template_transformed.paint_uniform_color([1, 0, 0])  # red

    o3d.visualization.draw_geometries([template_col, cluster_col])

    #############################
    # 2nd Register ICP ##########
    #############################

    result_icp = o3d.pipelines.registration.registration_icp(
        template_aligned, cluster_aligned,
        max_correspondence_distance=ICP_DIST,
        init=result_ransac.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    print(result_icp)

    ############################
    # Visualize result #########
    ############################

    T = result_icp.transformation
    template_transformed = template.transform(T)
    template_col = template_transformed.paint_uniform_color([1, 0, 0])  # red

    o3d.visualization.draw_geometries([template_col, cluster_col])
    print("Cluster size:", len(cluster.points))
    print("Template bounds:", template.get_max_bound() - template.get_min_bound())
    print("Cluster bounds:", cluster.get_max_bound() - cluster.get_min_bound())

    if result_icp.fitness >= MIN_FITNESS and result_icp.inlier_rmse <= MAX_RMSE:
        print(f"Cluster {cluster_label}: MATCH FOUND! (Fit: {result_icp.fitness:.3f})")
    else:
        print(f"REJECTED (Fit: {result_icp.fitness:.3f} < {MIN_FITNESS:.3f})")

    ############################
    # Calculate transform ######
    ############################

    # Matrix 1: Move Original Template to Center
    # T = -centroid
    Mat_Temp_Center = get_matrix(T=-centroid_template)
    
    # Matrix 2: Rotate Template to PCA Aligned
    # Note: Your PCA code does "points @ R". In Open3D matrix math (column vectors),
    # this corresponds to a rotation matrix of R.T
    Mat_Temp_PCA = get_matrix(R=R_t.T)
    
    # Matrix 3: The Registration Result (Aligned Temp -> Aligned Cluster)
    Mat_Registration = result_icp.transformation
    
    # Matrix 4: Un-Rotate Cluster (Inverse PCA)
    # The inverse of "points @ R_c" is "points @ R_c.T". 
    # In Open3D matrix math, the rotation matrix is (R_c.T).T = R_c
    Mat_Clus_UnPCA = get_matrix(R=R_c)
    
    # Matrix 5: Un-Center Cluster (Move back to real world)
    # T = +centroid
    Mat_Clus_UnCenter = get_matrix(T=centroid_cluster)
    
    # CHAIN THEM TOGETHER (Order is: Last Step @ ... @ First Step)
    Final_Transform = Mat_Clus_UnCenter @ Mat_Clus_UnPCA @ Mat_Registration @ Mat_Temp_PCA @ Mat_Temp_Center
    
    print("Final Real-World Transform:")
    print(Final_Transform)

    verification_template = copy.deepcopy(template)
    verification_template.paint_uniform_color([1, 0, 0]) # Red
    verification_template.transform(Final_Transform)
    
    # Visualizing against the ORIGINAL cropped scene (not centered, not rotated)
    scene_check = copy.deepcopy(pcd)
    scene_check.paint_uniform_color([0, 1, 0]) # Green
    
    o3d.visualization.draw_geometries([scene_check, verification_template], 
                                      window_name="Verification: Real World Alignment")

