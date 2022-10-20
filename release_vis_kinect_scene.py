import open3d as o3d
import cv2
import numpy as np
import json
from utils import *
import os
import argparse




def vis_kinect_scene():
    # read rgb/depth frame
    img_master_path = '{}/kinect_color/{}/master/frame_{}.jpg'.format(args.release_data_root, args.recording_name, args.vis_frame_id)
    depth_master_path = '{}/kinect_depth/{}/master/frame_{}.png'.format(args.release_data_root, args.recording_name, args.vis_frame_id)

    img_master = cv2.imread(img_master_path).astype(np.float32)[:, :, ::-1] / 255.0  # [1080, 1920, 3]
    depth_im_master = cv2.imread(depth_master_path, flags=-1).astype(float)  # [576, 640]
    depth_im_master = depth_im_master / 8.
    depth_im_master = depth_im_master * 0.001  # mm->m
    depth_im_master[depth_im_master >= 6] = 0

    # read extrinsic between master kinect and 3d scene mesh
    scene_calib_path = os.path.join(args.release_data_root, 'calibrations', args.recording_name, 'cal_trans', 'kinect12_to_world', '{}.json'.format(args.scene_name))
    with open(scene_calib_path) as calib_file:
        trans_master2scene = json.load(calib_file)['trans']
        trans_master2scene = np.asarray(trans_master2scene)

    # read master kinect cam params
    color_calib_path = os.path.join(args.release_data_root, 'kinect_cam_params/kinect_master/Color.json')
    depth_calib_path = os.path.join(args.release_data_root, 'kinect_cam_params/kinect_master/IR.json')
    with open(color_calib_path) as calib_file:
        color_cam_master = json.load(calib_file)
    with open(depth_calib_path) as calib_file:
        depth_cam_master = json.load(calib_file)

    # get colored point clouds from master kinect depth
    default_color_master = [1.00, 0.75, 0.80]
    points_depth_coord_master = unproject_depth_image(depth_im_master, depth_cam_master).reshape(-1, 3)  # point cloud from depth map in depth cam coord [576*640, 3]
    colors_master = np.tile(default_color_master, [points_depth_coord_master.shape[0], 1])  # [576*640, 3], each row: [1.0, 0.75, 0.8]
    # project points to color cam coord
    points_color_coord_master = points_coord_trans(points_depth_coord_master, np.asarray(depth_cam_master['ext_depth2color']))   # point cloud from depth map in color cam coord [576*640, 3]

    # full 3D points --> 2D coordinates in color image
    valid_idx_master, uvs_master = get_valid_idx(points_color_coord_master, color_cam_master)
    # get valid points and colors
    points_color_coord_master = points_color_coord_master[valid_idx_master]
    points_depth_coord_master = points_depth_coord_master[valid_idx_master]
    colors_master[valid_idx_master == True, :3] = img_master[uvs_master[:, 1], uvs_master[:, 0]]
    colors_master = colors_master[valid_idx_master]

    # read/vis 3d scene mesh
    scene_mesh_path = os.path.join(args.release_data_root, 'scene_mesh', args.scene_name, '{}.obj'.format(args.scene_name))
    scene_mesh = o3d.io.read_triangle_mesh(scene_mesh_path, enable_post_processing=True, print_progress=True)
    scene_mesh.compute_vertex_normals()
    scene_verts = np.asarray(scene_mesh.vertices)
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_verts)
    print('[INFO] visualize 3d scene mesh...')
    o3d.visualization.draw_geometries([scene_mesh])

    # visualize master kinect point clouds and 3d scene together, in the coordinate of master kinect color camera
    pcd_master = o3d.geometry.PointCloud()
    pcd_master.points = o3d.utility.Vector3dVector(points_color_coord_master)
    pcd_master.colors = o3d.utility.Vector3dVector(colors_master)
    print('[INFO] visualize master kinect point clouds...')
    o3d.visualization.draw_geometries([pcd_master])

    pcd_master.transform(trans_master2scene)
    print('[INFO] visualize master kinect point clouds and 3d scene together...')
    o3d.visualization.draw_geometries([pcd_master, scene_mesh])




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--release_data_root', type=str, default='/mnt/ssd/egobody_release', help='path to egobody dataset')
    parser.add_argument('--recording_name', type=str, default='recording_20220225_S27_S26_01')
    parser.add_argument('--scene_name', type=str, default='cnb_dlab_0225')
    parser.add_argument('--vis_frame_id', type=str, default='02500')

    args = parser.parse_args()
    vis_kinect_scene()
