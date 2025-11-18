import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

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

IMAGE = "data/Stella_xtion_1.png"
TEMPLATE = "data/stella_template_slice.pcd"
voxel_size = 0.01   # 1 cm
depth_image = o3d.io.read_image(IMAGE)

intrinsic = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
)

pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, intrinsic)

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


template = o3d.io.read_point_cloud(TEMPLATE)   # reference model
# Extract the points corresponding to a specific cluster label
cluster_label = 0  # Specify the cluster label you want to extract
cluster_points = np.asarray(pcd.points)[labels == cluster_label]

# Create a new point cloud for the selected cluster
cluster = o3d.geometry.PointCloud()
cluster.points = o3d.utility.Vector3dVector(cluster_points)

o3d.visualization.draw_geometries([template, cluster])

template = center_cloud(template)
cluster  = center_cloud(cluster)

template_aligned, R_t = pca_align(template)
cluster_aligned,  R_c = pca_align(cluster)

template.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
)
cluster.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
)

# Preprocess the template point cloud
template_down, template_fpfh = preprocess_pcd(template, voxel_size)
cluster_down, cluster_fpfh   = preprocess_pcd(cluster, voxel_size)

distance_threshold = voxel_size * 1.5

result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    template_down, cluster_down,
    template_fpfh, cluster_fpfh,
    mutual_filter=True,
    max_correspondence_distance=distance_threshold,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n=4,
    checkers=[
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9)
    ],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
        max_iteration=400000,
        confidence=0.999
    )
)

print(result_ransac)

result_icp = o3d.pipelines.registration.registration_icp(
    template, cluster,
    max_correspondence_distance=voxel_size * 1.0,
    init=result_ransac.transformation,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

print(result_icp)
T = result_icp.transformation

template_transformed = template.transform(T)

template_col = template_transformed.paint_uniform_color([1, 0, 0])  # red
cluster_col  = cluster.paint_uniform_color([0, 1, 0])               # green

o3d.visualization.draw_geometries([template_col, cluster_col])
print("Cluster size:", len(cluster.points))
print("Template bounds:", template.get_max_bound() - template.get_min_bound())
print("Cluster bounds:", cluster.get_max_bound() - cluster.get_min_bound())

