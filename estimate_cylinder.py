from dataclasses import dataclass
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy

@dataclass
class Params:
    # --- GLOBAL SCALE ---
    # The most important parameter. Roughly 5-10% of your object size.
    voxel_size: float = 0.15
    
    # --- PREPROCESSING ---
    # Multipliers relative to voxel_size
    normal_radius_mult: float = 2.0  # Search radius for normal estimation
    fpfh_radius_mult: float   = 5.0  # Search radius for feature description
    
    # --- CLUSTERING (DBSCAN) ---
    # How close points must be to belong to the same cluster (in scene units)
    dbscan_eps: float        = 2.0
    dbscan_min_points: int   = 500
    
    # --- RANSAC (Coarse Registration) ---
    ransac_n: int            = 3        # Points to pick per iteration (3 or 4)
    ransac_iter: int         = 4000000  # Number of attempts
    ransac_conf: float       = 0.999    # Confidence level
    # Multiplier for max distance in RANSAC (e.g., 1.5 * voxel_size)
    ransac_dist_mult: float  = 2.0      
    # Edge length check (0.0 to 1.0). Higher = stricter geometric check.
    # Set to 0.0 to disable if matching partial views to full models.
    edge_length_thresh: float = 0.0     

    # --- ICP (Fine Registration) ---
    # Multiplier for max distance in ICP (e.g., 3.0 * voxel_size)
    icp_dist_mult: float     = 4.0      
    
    # --- VERIFICATION (Paper Logic) ---
    # Safety margins for acceptance.
    # 0.90 means we accept 90% of the "perfect" self-match fitness.
    fitness_margin: float    = 0.60     
    rmse_margin: float       = 1.90

# Initialize the config
CFG = Params()

# --- HELPER FUNCTIONS ---

def preprocess_pcd(pcd, voxel_size):
    """Downsample + estimate normals + compute FPFH."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Radius parameters pulled from CFG
    radius_normal = voxel_size * CFG.normal_radius_mult
    radius_feature = voxel_size * CFG.fpfh_radius_mult

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, fpfh

def center_cloud(pcd):
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    pts -= centroid
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def pca_align(pcd):
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid
    H = pts_centered.T @ pts_centered
    eigvals, eigvecs = np.linalg.eigh(H)
    R = eigvecs[:, ::-1]
    pts_rot = pts_centered @ R
    aligned = o3d.geometry.PointCloud()
    aligned.points = o3d.utility.Vector3dVector(pts_rot)
    return aligned, R

def get_matrix(R=np.eye(3), T=np.zeros(3)):
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = T
    return mat

def get_hard_constraints(template, voxel_size):
    template.estimate_normals()
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(template, depth=8)
    
    n_points = int(len(template.points) * 0.5) 
    resampled_template = mesh.sample_points_poisson_disk(n_points)
    
    # Use config for threshold
    threshold = voxel_size * CFG.ransac_dist_mult 
    trans_init = np.eye(4)
    
    reg_p2l = o3d.pipelines.registration.registration_icp(
        resampled_template, template, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    min_fitness = reg_p2l.fitness
    max_rmse = reg_p2l.inlier_rmse
    
    # Use config for margins
    final_fitness_thresh = min_fitness * CFG.fitness_margin
    final_rmse_thresh = max_rmse * CFG.rmse_margin
    
    print(f"  > Self-Fitness: {min_fitness:.4f} -> Threshold: {final_fitness_thresh:.4f}")
    print(f"  > Self-RMSE:    {max_rmse:.4f}  -> Threshold: {final_rmse_thresh:.4f}")
    
    return final_fitness_thresh, final_rmse_thresh

# --- MAIN EXECUTION ---

print("--- 3D Object Pose Estimation Pipeline ---")
print("Using Config:", CFG)

IMAGE = "data/Stella_xtion_1.png"
TEMPLATE_SLICE = "data/stella_template_slice.pcd"
TEMPLATE = "data/stella_template.pcd"

# Derived Parameters (Calculated from Config)
RANSAC_DIST = CFG.voxel_size * CFG.ransac_dist_mult
ICP_DIST    = CFG.voxel_size * CFG.icp_dist_mult

# 1. Load Data
depth_image = o3d.io.read_image(IMAGE)
intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, intrinsic)

template = o3d.io.read_point_cloud(TEMPLATE_SLICE)
centroid_template_original = np.asarray(template.points).mean(axis=0)

# 2. Get Constraints
print("--- Step 1: Generating Hard Constraints (Self-Resampling) ---")
MIN_FITNESS, MAX_RMSE = get_hard_constraints(template, CFG.voxel_size)

# 3. Crop (Keeping your YOLO logic as is)
txt_file = IMAGE.rsplit(".", 1)[0] + ".txt"
cls_id, xc, yc, w, h = map(float, open(txt_file).read().split())
img_np = np.asarray(depth_image)
H, W = img_np.shape[:2]
xmin, xmax = int((xc - w/2) * W), int((xc + w/2) * W)
ymin, ymax = int((yc - h/2) * H), int((yc + h/2) * H)

points = np.asarray(pcd.points)
X, Y, Z = points[:,0], points[:,1], points[:,2]
fx, fy, cx, cy = intrinsic.intrinsic_matrix[0,0], intrinsic.intrinsic_matrix[1,1], intrinsic.intrinsic_matrix[0,2], intrinsic.intrinsic_matrix[1,2]
u = (X * fx / Z) + cx
v = (Y * fy / Z) + cy
mask = (Z > 0) & (u >= xmin) & (u <= xmax) & (v >= ymin) & (v <= ymax)
cropped = o3d.geometry.PointCloud()
cropped.points = o3d.utility.Vector3dVector(points[mask])

# 4. Clustering
labels = np.array(cropped.cluster_dbscan(eps=CFG.dbscan_eps, min_points=CFG.dbscan_min_points, print_progress=True))
max_label = labels.max()

print("--- Step 2: Clustering Scene Point Cloud ---")
print(f"> Found {max_label + 1} clusters")

# 5. Loop
for cluster_label in range(max_label + 1):
    print("-> Cluster ", cluster_label)
    cluster_points = np.asarray(cropped.points)[labels == cluster_label]
    cluster = o3d.geometry.PointCloud()
    cluster.points = o3d.utility.Vector3dVector(cluster_points)
    
    # Save original centroid for final transform
    centroid_cluster = cluster_points.mean(axis=0)

    template_working = copy.deepcopy(template)
    cluster_working  = copy.deepcopy(cluster)

    # Center & PCA
    template_working = center_cloud(template_working)
    cluster_working  = center_cloud(cluster_working)

    template_aligned, R_t = pca_align(template_working)
    cluster_aligned,  R_c = pca_align(cluster_working)
    
    template_aligned.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=CFG.voxel_size, max_nn=30))
    cluster_aligned.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=CFG.voxel_size, max_nn=30))

    # Preprocess
    template_down, template_fpfh = preprocess_pcd(template_aligned, CFG.voxel_size)
    cluster_down, cluster_fpfh   = preprocess_pcd(cluster_aligned, CFG.voxel_size)

    # RANSAC
    checkers = [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(RANSAC_DIST)]
    
    # Only add edge checker if enabled in config
    if CFG.edge_length_thresh > 0:
        checkers.append(o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(CFG.edge_length_thresh))

    print("--- Step 3: RANSAC Registration ---")
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        template_down, cluster_down, template_fpfh, cluster_fpfh,
        mutual_filter=True,
        max_correspondence_distance=RANSAC_DIST,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=CFG.ransac_n,
        checkers=checkers,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(CFG.ransac_iter, CFG.ransac_conf)
    )

    print(f"> RANSAC Fitness: {result_ransac.fitness:.3f} | Inlier RMSE: {result_ransac.inlier_rmse:.3f}")

    print("--- Step 4: ICP Refinement ---")
    # ICP
    result_icp = o3d.pipelines.registration.registration_icp(
        template_aligned, cluster_aligned,
        max_correspondence_distance=ICP_DIST,
        init=result_ransac.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    print(f"> ICP Fitness: {result_icp.fitness:.3f} | Inlier RMSE: {result_icp.inlier_rmse:.3f}")

    # Verification
    accepted_fitness = (result_icp.fitness >= MIN_FITNESS)
    accepted_rmse    = (result_icp.inlier_rmse <= MAX_RMSE)
    if not accepted_fitness:
        print("  > REJECTED: Fitness below threshold.")
        print(f"    - Fitness: {result_icp.fitness:.4f} (Min: {MIN_FITNESS:.4f})")
    if not accepted_rmse:
        print("  > REJECTED: RMSE above threshold.")
        print(f"    - RMSE: {result_icp.inlier_rmse:.4f} (Max: {MAX_RMSE:.4f})")
    if accepted_fitness and accepted_rmse:
        print("  > ACCEPTED: Meets all criteria.")

        # Calculate Final Transform
        Mat_Temp_Center = get_matrix(T=-centroid_template_original)
        Mat_Temp_PCA = get_matrix(R=R_t.T)
        Mat_Registration = result_icp.transformation
        Mat_Clus_UnPCA = get_matrix(R=R_c)
        Mat_Clus_UnCenter = get_matrix(T=centroid_cluster)
        
        Final_Transform = Mat_Clus_UnCenter @ Mat_Clus_UnPCA @ Mat_Registration @ Mat_Temp_PCA @ Mat_Temp_Center
        print("  > Final Transformation Matrix:")
        print(Final_Transform)

        # Visualize Match
        verification_template = o3d.io.read_point_cloud(TEMPLATE)
        verification_template.paint_uniform_color([1, 0, 0])
        verification_template.transform(Final_Transform)
        
        scene_check = copy.deepcopy(pcd)
        scene_check.paint_uniform_color([0, 1, 0])
        
        o3d.visualization.draw_geometries([scene_check, verification_template], window_name="Result")