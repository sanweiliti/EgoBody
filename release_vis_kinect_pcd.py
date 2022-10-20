import numpy as np
import open3d as o3d
import os
import cv2
import json
from utils import *
import argparse


def vis_kinect_pcd():
    if args.recording_name.split('_')[1][0:4] == '2022':
        num_cam = 5
    else:
        num_cam = 3

    color_calib_path = os.path.join(args.release_data_root, 'kinect_cam_params/kinect_master/Color.json')
    depth_calib_path = os.path.join(args.release_data_root, 'kinect_cam_params/kinect_master/IR.json')
    with open(color_calib_path) as calib_file:
        kinect_color_cam_master = json.load(calib_file)
    with open(depth_calib_path) as calib_file:
        kinect_depth_cam_master = json.load(calib_file)

    color_calib_path = os.path.join(args.release_data_root, 'kinect_cam_params/kinect_sub_1/Color.json')
    depth_calib_path = os.path.join(args.release_data_root, 'kinect_cam_params/kinect_sub_1/IR.json')
    with open(color_calib_path) as calib_file:
        kinect_color_cam_sub_1 = json.load(calib_file)
    with open(depth_calib_path) as calib_file:
        kinect_depth_cam_sub_1 = json.load(calib_file)

    color_calib_path = os.path.join(args.release_data_root, 'kinect_cam_params/kinect_sub_2/Color.json')
    depth_calib_path = os.path.join(args.release_data_root, 'kinect_cam_params/kinect_sub_2/IR.json')
    with open(color_calib_path) as calib_file:
        kinect_color_cam_sub_2 = json.load(calib_file)
    with open(depth_calib_path) as calib_file:
        kinect_depth_cam_sub_2 = json.load(calib_file)

    color_calib_path = os.path.join(args.release_data_root, 'kinect_cam_params/kinect_sub_3/Color.json')
    depth_calib_path = os.path.join(args.release_data_root, 'kinect_cam_params/kinect_sub_3/IR.json')
    with open(color_calib_path) as calib_file:
        kinect_color_cam_sub_3 = json.load(calib_file)
    with open(depth_calib_path) as calib_file:
        kinect_depth_cam_sub_3 = json.load(calib_file)

    color_calib_path = os.path.join(args.release_data_root, 'kinect_cam_params/kinect_sub_4/Color.json')
    depth_calib_path = os.path.join(args.release_data_root, 'kinect_cam_params/kinect_sub_4/IR.json')
    with open(color_calib_path) as calib_file:
        kinect_color_cam_sub_4 = json.load(calib_file)
    with open(depth_calib_path) as calib_file:
        kinect_depth_cam_sub_4 = json.load(calib_file)

    # read trans matrix: sub to master kinect
    calib_trans_dir = os.path.join(args.release_data_root, 'calibrations', args.recording_name, 'cal_trans')
    trans_sub1tomaster_path = os.path.join(calib_trans_dir, 'kinect_11to12_color.json')
    trans_sub2tomaster_path = os.path.join(calib_trans_dir, 'kinect_13to12_color.json')
    if num_cam == 5:
        trans_sub3tomaster_path = os.path.join(calib_trans_dir, 'kinect_14to12_color.json')
        trans_sub4tomaster_path = os.path.join(calib_trans_dir, 'kinect_15to12_color.json')
    # load calibration from sub kinect to master kinect (between color cameras)
    with open(os.path.join(trans_sub1tomaster_path), 'r') as f:
        trans_sub1tomaster = np.asarray(json.load(f)['trans'])
    with open(os.path.join(trans_sub2tomaster_path), 'r') as f:
        trans_sub2tomaster = np.asarray(json.load(f)['trans'])
    if num_cam == 5:
        with open(os.path.join(trans_sub3tomaster_path), 'r') as f:
            trans_sub3tomaster = np.asarray(json.load(f)['trans'])
        with open(os.path.join(trans_sub4tomaster_path), 'r') as f:
            trans_sub4tomaster = np.asarray(json.load(f)['trans'])


    kinect_img_master_path = '{}/kinect_color/{}/master/frame_{}.jpg'.format(args.release_data_root, args.recording_name, args.vis_frame_id)
    kinect_depth_master_path = '{}/kinect_depth/{}/master/frame_{}.png'.format(args.release_data_root, args.recording_name, args.vis_frame_id)

    kinect_img_master = cv2.imread(kinect_img_master_path)
    kinect_depth_im_master = cv2.imread(kinect_depth_master_path, flags=-1).astype(float)  # [576, 640]
    kinect_depth_im_master = kinect_depth_im_master / 8.
    kinect_depth_im_master = kinect_depth_im_master * 0.001  # mm->m
    # kinect_depth_im_master[kinect_depth_im_master >= 3.86] = 0

    kinect_img_sub_1_path = '{}/kinect_color/{}/sub_1/frame_{}.jpg'.format(args.release_data_root, args.recording_name, args.vis_frame_id)
    kinect_depth_sub_1_path = '{}/kinect_depth/{}/sub_1/frame_{}.png'.format(args.release_data_root, args.recording_name, args.vis_frame_id)
    kinect_img_sub_1 = cv2.imread(kinect_img_sub_1_path)
    kinect_depth_im_sub_1 = cv2.imread(kinect_depth_sub_1_path, flags=-1).astype(float)  # [576, 640]
    kinect_depth_im_sub_1 = kinect_depth_im_sub_1 / 8.
    kinect_depth_im_sub_1 = kinect_depth_im_sub_1 * 0.001  # mm->m
    # kinect_depth_im_sub_1[kinect_depth_im_sub_1 >= 3.86] = 0

    kinect_img_sub_2_path = '{}/kinect_color/{}/sub_2/frame_{}.jpg'.format(args.release_data_root, args.recording_name, args.vis_frame_id)
    kinect_depth_sub_2_path = '{}/kinect_depth/{}/sub_2/frame_{}.png'.format(args.release_data_root, args.recording_name, args.vis_frame_id)
    kinect_img_sub_2 = cv2.imread(kinect_img_sub_2_path)
    kinect_depth_im_sub_2 = cv2.imread(kinect_depth_sub_2_path, flags=-1).astype(float)  # [576, 640]
    kinect_depth_im_sub_2 = kinect_depth_im_sub_2 / 8.
    kinect_depth_im_sub_2 = kinect_depth_im_sub_2 * 0.001  # mm->m
    # kinect_depth_im_sub_2[kinect_depth_im_sub_2 >= 3.86] = 0

    if num_cam == 5:
        kinect_img_sub_3_path = '{}/kinect_color/{}/sub_3/frame_{}.jpg'.format(args.release_data_root, args.recording_name, args.vis_frame_id)
        kinect_depth_sub_3_path = '{}/kinect_depth/{}/sub_3/frame_{}.png'.format(args.release_data_root, args.recording_name, args.vis_frame_id)
        kinect_img_sub_3 = cv2.imread(kinect_img_sub_3_path)
        kinect_depth_im_sub_3 = cv2.imread(kinect_depth_sub_3_path, flags=-1).astype(float)  # [576, 640]
        kinect_depth_im_sub_3 = kinect_depth_im_sub_3 / 8.
        kinect_depth_im_sub_3 = kinect_depth_im_sub_3 * 0.001  # mm->m
        # kinect_depth_im_sub_3[kinect_depth_im_sub_3 >= 3.86] = 0

        kinect_img_sub_4_path = '{}/kinect_color/{}/sub_4/frame_{}.jpg'.format(args.release_data_root, args.recording_name, args.vis_frame_id)
        kinect_depth_sub_4_path = '{}/kinect_depth/{}/sub_4/frame_{}.png'.format(args.release_data_root, args.recording_name, args.vis_frame_id)
        kinect_img_sub_4 = cv2.imread(kinect_img_sub_4_path)
        kinect_depth_im_sub_4 = cv2.imread(kinect_depth_sub_4_path, flags=-1).astype(float)  # [576, 640]
        kinect_depth_im_sub_4 = kinect_depth_im_sub_4 / 8.
        kinect_depth_im_sub_4 = kinect_depth_im_sub_4 * 0.001  # mm->m
        # kinect_depth_im_sub_4[kinect_depth_im_sub_4 >= 3.86] = 0


    default_color = [1.00, 0.75, 0.80]
    # kinect master pcd
    kinect_img_master = kinect_img_master.astype(np.float32)[:, :, ::-1] / 255.0  # [1080, 1920, 3]
    kinect_points_depth_coord_master = unproject_depth_image(kinect_depth_im_master, kinect_depth_cam_master).reshape(-1, 3)  # point cloud from depth map in depth cam coord [576*640, 3]
    kinect_colors_master = np.tile(default_color, [kinect_points_depth_coord_master.shape[0], 1])  # [576*640, 3], each row: [1.0, 0.75, 0.8]
    # point cloud from depth map in color cam coord [576*640, 3]
    kinect_points_color_coord_master = points_coord_trans(kinect_points_depth_coord_master, np.asarray(kinect_depth_cam_master['ext_depth2color']))
    # full 3D points --> 2D coordinates in color image
    kinect_valid_idx_master, kinect_uvs_master = get_valid_idx(kinect_points_color_coord_master, kinect_color_cam_master)
    # get valid points and colors
    kinect_points_color_coord_master = kinect_points_color_coord_master[kinect_valid_idx_master]
    kinect_colors_master[kinect_valid_idx_master == True, :3] = kinect_img_master[kinect_uvs_master[:, 1], kinect_uvs_master[:, 0]]
    kinect_colors_master = kinect_colors_master[kinect_valid_idx_master]

    # kinect sub1 pcd
    kinect_img_sub_1 = kinect_img_sub_1.astype(np.float32)[:, :, ::-1] / 255.0  # [1080, 1920, 3]
    kinect_points_depth_coord_sub_1 = unproject_depth_image(kinect_depth_im_sub_1, kinect_depth_cam_sub_1).reshape(-1, 3)
    kinect_colors_sub_1 = np.tile(default_color, [kinect_points_depth_coord_sub_1.shape[0], 1])
    kinect_points_color_coord_sub_1 = points_coord_trans(kinect_points_depth_coord_sub_1, np.asarray(kinect_depth_cam_sub_1['ext_depth2color']))
    kinect_valid_idx_sub_1, kinect_uvs_sub_1 = get_valid_idx(kinect_points_color_coord_sub_1, kinect_color_cam_sub_1)
    kinect_points_color_coord_sub_1 = kinect_points_color_coord_sub_1[kinect_valid_idx_sub_1]
    kinect_colors_sub_1[kinect_valid_idx_sub_1 == True, :3] = kinect_img_sub_1[kinect_uvs_sub_1[:, 1], kinect_uvs_sub_1[:, 0]]
    kinect_colors_sub_1 = kinect_colors_sub_1[kinect_valid_idx_sub_1]

    # kinect sub2 pcd
    kinect_img_sub_2 = kinect_img_sub_2.astype(np.float32)[:, :, ::-1] / 255.0  # [1080, 1920, 3]
    kinect_points_depth_coord_sub_2 = unproject_depth_image(kinect_depth_im_sub_2, kinect_depth_cam_sub_2).reshape(-1, 3)
    kinect_colors_sub_2 = np.tile(default_color, [kinect_points_depth_coord_sub_2.shape[0], 1])
    kinect_points_color_coord_sub_2 = points_coord_trans(kinect_points_depth_coord_sub_2, np.asarray(kinect_depth_cam_sub_2['ext_depth2color']))
    kinect_valid_idx_sub_2, kinect_uvs_sub_2 = get_valid_idx(kinect_points_color_coord_sub_2, kinect_color_cam_sub_2)
    kinect_points_color_coord_sub_2 = kinect_points_color_coord_sub_2[kinect_valid_idx_sub_2]
    kinect_colors_sub_2[kinect_valid_idx_sub_2 == True, :3] = kinect_img_sub_2[kinect_uvs_sub_2[:, 1], kinect_uvs_sub_2[:, 0]]
    kinect_colors_sub_2 = kinect_colors_sub_2[kinect_valid_idx_sub_2]

    if num_cam == 5:
        # kinect sube pcd
        kinect_img_sub_3 = kinect_img_sub_3.astype(np.float32)[:, :, ::-1] / 255.0  # [1080, 1920, 3]
        kinect_points_depth_coord_sub_3 = unproject_depth_image(kinect_depth_im_sub_3, kinect_depth_cam_sub_3).reshape(-1, 3)
        kinect_colors_sub_3 = np.tile(default_color, [kinect_points_depth_coord_sub_3.shape[0], 1])
        kinect_points_color_coord_sub_3 = points_coord_trans(kinect_points_depth_coord_sub_3, np.asarray(kinect_depth_cam_sub_3['ext_depth2color']))
        kinect_valid_idx_sub_3, kinect_uvs_sub_3 = get_valid_idx(kinect_points_color_coord_sub_3, kinect_color_cam_sub_3)
        kinect_points_color_coord_sub_3 = kinect_points_color_coord_sub_3[kinect_valid_idx_sub_3]
        kinect_colors_sub_3[kinect_valid_idx_sub_3 == True, :3] = kinect_img_sub_3[kinect_uvs_sub_3[:, 1], kinect_uvs_sub_3[:, 0]]
        kinect_colors_sub_3 = kinect_colors_sub_3[kinect_valid_idx_sub_3]

        # kinect sub4 pcd
        kinect_img_sub_4 = kinect_img_sub_4.astype(np.float32)[:, :, ::-1] / 255.0  # [1080, 1920, 3]
        kinect_points_depth_coord_sub_4 = unproject_depth_image(kinect_depth_im_sub_4, kinect_depth_cam_sub_4).reshape(-1, 3)
        kinect_colors_sub_4 = np.tile(default_color, [kinect_points_depth_coord_sub_4.shape[0], 1])
        kinect_points_color_coord_sub_4 = points_coord_trans(kinect_points_depth_coord_sub_4, np.asarray(kinect_depth_cam_sub_4['ext_depth2color']))
        kinect_valid_idx_sub_4, kinect_uvs_sub_4 = get_valid_idx(kinect_points_color_coord_sub_4, kinect_color_cam_sub_4)
        kinect_points_color_coord_sub_4 = kinect_points_color_coord_sub_4[kinect_valid_idx_sub_4]
        kinect_colors_sub_4[kinect_valid_idx_sub_4 == True, :3] = kinect_img_sub_4[kinect_uvs_sub_4[:, 1], kinect_uvs_sub_4[:, 0]]
        kinect_colors_sub_4 = kinect_colors_sub_4[kinect_valid_idx_sub_4]


    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

    kinect_pcd_master = o3d.geometry.PointCloud()
    kinect_pcd_master.points = o3d.utility.Vector3dVector(kinect_points_color_coord_master)
    kinect_pcd_master.colors = o3d.utility.Vector3dVector(kinect_colors_master)
    print('[INFO] visualize master kinect pcd ... (in master kinect color coord)')
    o3d.visualization.draw_geometries([kinect_pcd_master])

    kinect_pcd_sub_1 = o3d.geometry.PointCloud()
    kinect_pcd_sub_1.points = o3d.utility.Vector3dVector(kinect_points_color_coord_sub_1)
    kinect_pcd_sub_1.colors = o3d.utility.Vector3dVector(kinect_colors_sub_1)
    kinect_pcd_sub_1.transform(trans_sub1tomaster)
    print('[INFO] visualize sub1 kinect pcd ... (in master kinect color coord)')
    o3d.visualization.draw_geometries([kinect_pcd_sub_1])

    kinect_pcd_sub_2 = o3d.geometry.PointCloud()
    kinect_pcd_sub_2.points = o3d.utility.Vector3dVector(kinect_points_color_coord_sub_2)
    kinect_pcd_sub_2.colors = o3d.utility.Vector3dVector(kinect_colors_sub_2)
    kinect_pcd_sub_2.transform(trans_sub2tomaster)
    print('[INFO] visualize sub2 kinect pcd... (in master kinect color coord)')
    o3d.visualization.draw_geometries([kinect_pcd_sub_2])

    if num_cam == 5:
        kinect_pcd_sub_3 = o3d.geometry.PointCloud()
        kinect_pcd_sub_3.points = o3d.utility.Vector3dVector(kinect_points_color_coord_sub_3)
        kinect_pcd_sub_3.colors = o3d.utility.Vector3dVector(kinect_colors_sub_3)
        kinect_pcd_sub_3.transform(trans_sub3tomaster)
        print('[INFO] visualize sub3 kinect pcd... (in master kinect color coord)')
        o3d.visualization.draw_geometries([kinect_pcd_sub_3])

        kinect_pcd_sub_4 = o3d.geometry.PointCloud()
        kinect_pcd_sub_4.points = o3d.utility.Vector3dVector(kinect_points_color_coord_sub_4)
        kinect_pcd_sub_4.colors = o3d.utility.Vector3dVector(kinect_colors_sub_4)
        kinect_pcd_sub_4.transform(trans_sub4tomaster)
        print('[INFO] visualize sub4 kinect pcd... (in master kinect color coord)')
        o3d.visualization.draw_geometries([kinect_pcd_sub_4])

    print('[INFO] visualize kinect pcd from all views... (in master kinect color coord)')
    if num_cam == 5:
        o3d.visualization.draw_geometries([kinect_pcd_master, kinect_pcd_sub_1, kinect_pcd_sub_2, kinect_pcd_sub_3, kinect_pcd_sub_4, mesh_frame])
    else:
        o3d.visualization.draw_geometries([kinect_pcd_master, kinect_pcd_sub_1, kinect_pcd_sub_2, mesh_frame])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--release_data_root', type=str, default='/mnt/ssd/egobody_release', help='path to egobody dataset')
    parser.add_argument('--recording_name', type=str, default='recording_20220218_S02_S23_02')
    parser.add_argument('--vis_frame_id', type=str, default='03000')

    args = parser.parse_args()
    vis_kinect_pcd()