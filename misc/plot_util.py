import open3d as o3d

def plot_reg(pcd0, pcd1, T):

  pcd0.paint_uniform_color([1, 0.706, 0])
  pcd1.paint_uniform_color([0, 0.651, 0.929])

  o3d.visualization.draw_geometries([pcd0, pcd1])

  pcd0.transform(T)

  o3d.visualization.draw_geometries([pcd0, pcd1])