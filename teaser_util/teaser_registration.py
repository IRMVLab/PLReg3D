import open3d as o3d
import teaserpp_python
import numpy as np
import copy
from teaser_util.helpers import *

def make_od3(points, voxel_size = None, color = [0.0, 0.0, 1.0]):
    A_pcd_raw = o3d.geometry.PointCloud()
    if points.shape[0] < points.shape[1]:
        points = points.T
    A_pcd_raw.points = o3d.utility.Vector3dVector(points)
    A_pcd_raw.paint_uniform_color(color)  # show A_pcd in blue
    if voxel_size is not None:
        A_pcd_raw = A_pcd_raw.voxel_down_sample(voxel_size=voxel_size)
    return A_pcd_raw

def draw_correspondences(A_xyz, B_xyz, corrs_A, corrs_B):

    A_corr = A_xyz[:, corrs_A]  # np array of size 3 by num_corrs
    B_corr = B_xyz[:, corrs_B]  # np array of size 3 by num_corrs

    num_corrs = A_corr.shape[1]

    # visualize the point clouds together with feature correspondences
    points = np.concatenate((A_corr.T, B_corr.T), axis=0)
    lines = []
    for i in range(num_corrs):
        lines.append([i, i + num_corrs])
    colors = [[0, 1, 0] for i in range(len(lines))]  # lines are shown in green
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    A_pcd = make_od3(A_xyz, color = [0.0, 0.0, 1.0])
    B_pcd = make_od3(B_xyz, color = [1.0, 0.0, 0.0])

    o3d.visualization.draw_geometries([A_pcd, B_pcd, line_set])

def teaser_feat(A_raw, B_raw, A_feats, B_feats, voxel_size, VISUALIZE = False):
    #  A_raw [N, 3] A_feats [N, F]

    A_pcd = make_od3(A_raw, color = [0.0, 0.0, 1.0])
    B_pcd = make_od3(B_raw, color = [1.0, 0.0, 0.0])
    if VISUALIZE:
        o3d.visualization.draw_geometries([A_pcd, B_pcd])

    A_xyz = pcd2xyz(A_pcd)  # np array of size 3 by N
    B_xyz = pcd2xyz(B_pcd)  # np array of size 3 by M

    # establish correspondences by nearest neighbour search in feature space
    # corrs_A: np_array [N] corrs_B: np_array [N]
    corrs_A, corrs_B = find_correspondences(
        A_feats, B_feats, mutual_filter=True)
    A_corr = A_xyz[:, corrs_A]  # np array of size 3 by num_corrs
    B_corr = B_xyz[:, corrs_B]  # np array of size 3 by num_corrs

    # 画匹配关系
    if VISUALIZE:
        draw_correspondences(A_xyz, B_xyz, corrs_A, corrs_B)

    # robust global registration using TEASER++
    NOISE_BOUND = voxel_size
    teaser_solver = get_teaser_solver(NOISE_BOUND)
    teaser_solver.solve(A_corr, B_corr)
    solution = teaser_solver.getSolution()
    R_teaser = solution.rotation
    t_teaser = solution.translation
    T_teaser = Rt2T(R_teaser, t_teaser)

    # Visualize the registration results
    A_pcd_T_teaser = copy.deepcopy(A_pcd).transform(T_teaser)

    if VISUALIZE:
        o3d.visualization.draw_geometries([A_pcd_T_teaser, B_pcd])

    return T_teaser