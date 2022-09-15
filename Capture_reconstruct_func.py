import os.path

import open3d as o3d
import numpy as np
import pyrealsense2 as rs


def set_bounds_in_first_quadrant(mesh):
    """translate mesh to first quadrant of coordinate system"""
    min_bound = mesh.get_min_bound()
    mesh.translate((-(min_bound[0]), -(min_bound[1]), -(min_bound[2])))
    return mesh


def zoom_image(image):
    return o3d.geometry.AxisAlignedBoundingBox(image.get_min_bound(),
                                               image.get_max_bound())


def reconstrct_aplha_shapes(pcd, alpha):
    """
    Reconstruction algorithm using Alpha shapes method.
    Ref::  Edelsbrunner, Herbert, David Kirkpatrick, and Raimund Seidel.
    "On the shape of a set of points in the plane." IEEE Transactions on information theory 29.4 (1983): 551-559.
    """

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    mesh = set_bounds_in_first_quadrant(mesh)
    return mesh


def reconstrct_poisson_surface(pcd, depth, width, scale, linear_fit, n_threads):
    """
    Reconstruction method using Poisson surface method.
    Ref:: Kazhdan and M. Bolitho and H. Hoppe: Poisson surface reconstruction, Eurographics, 2006.
    """

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth,
                                                                                width=width,
                                                                                scale=scale,
                                                                                linear_fit=linear_fit,
                                                                                n_threads=n_threads)
    mesh.compute_vertex_normals()
    # vertices_to_remove = densities < np.quantile(densities, 0.001)
    # mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh = set_bounds_in_first_quadrant(mesh)
    return mesh


def reconstruct_ball_pivoting(pcd, factor):
    """
    Reconstruction method using Ball pivoting method.
    Ref:: Bernardini and J. Mittleman and HRushmeier and C. Silva and G. Taubin:
    The ball-pivoting algorithm for surface reconstruction, IEEE transactions on
    visualization and computer graphics, 5(4), 349-359, 1999
    """
    distances = pcd.compute_nearest_neighbor_distance()
    ro = (1.25 * np.mean(distances)) / 2  # https://cs184team.github.io/cs184-final/writeup.html
    radii = [ro, factor * ro, ro * factor * 2]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    mesh.compute_vertex_normals()
    mesh = set_bounds_in_first_quadrant(mesh)
    return radii, mesh


def get_scene_pcd_from_camera():
    """
    This function captures the three-dimensional point cloud from the intel realsense camera.
    and save the point cloud in a global variable. This variable value is updated every time this
    function is called.
    """

    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Streaming start
    pipeline = rs.pipeline()
    pipeline.start(config)

    # Align objects generated
    align_to = rs.stream.color
    align = rs.align(align_to)

    # frame wait (Color Ando Depth)
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    color_image = o3d.geometry.Image(np.asanyarray(color_frame.get_data()))

    depth_image = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    # Rotating
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Normal calculation
    pcd.estimate_normals()

    pcd = set_bounds_in_first_quadrant(pcd)
    pipeline.stop()
    return pcd


def remove_statistical_outlier(pcd, nb_neighbors, std_ratio):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                             std_ratio=std_ratio)
    cl.estimate_normals()
    return cl


def remove_radius_outlier(pcd, nb_points, radius):
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    cl.estimate_normals()
    return cl


def down_sample_voxel_size(pcd, ds_voxel_size):
    pcd.voxel_down_sample(voxel_size=ds_voxel_size)
    pcd.estimate_normals()

    return pcd


def down_sample_uniform(pcd, every_k_point):
    pcd.uniform_down_sample(every_k_points=every_k_point)
    pcd.estimate_normals()
    return pcd


def crop_func(pcd):
    print("Demo for manual geometry cropping")
    print(
        "1) Press 'Y' twice to align geometry with negative direction of y-axis"
    )
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry")
    print("5) Press 'S' to save the selected geometry")
    print("6) Press 'F' to switch to freeview mode")
    o3d.visualization.draw_geometries_with_editing([pcd])



def crop_function2(app, pcd):
    print('crop_function2')
    #vis = o3d.visualization.VisualizerWithEditing()
    vis = o3d.visualization.O3DVisualizer('test')

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"

    print('before add_geometry')
    vis.add_geometry('test', pcd, mat)

    print('before camera reset')
    vis.reset_camera_to_default()
    bounds = pcd.get_axis_aligned_bounding_box()
    extent = bounds.get_extent()
    print('before setup camera')
    vis.setup_camera(60, bounds.get_center(),
                         bounds.get_center() + [0, 0, -3], [0, -1, 0])
    print('New vis is prepped')
    app.add_window(vis)
    print('added window')

    #print(vis.get_picked_points())
    return None
