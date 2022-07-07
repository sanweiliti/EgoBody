import os
import sys

# rootPath = '../'
# sys.path.append(rootPath)

import os.path as osp
import cv2
import numpy as np
import json
import trimesh
import argparse
# os.environ["PYOPENGL_PLATFORM"] = "egl"
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import pyrender
import PIL.Image as pil_img
import pickle
import smplx
import torch
import glob
import ast
# import open3d as o3d
from tqdm import tqdm
from os.path import basename
from PIL import ImageDraw
import pandas as pd

from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model2data(model_type):
    return smpl_to_openpose(model_type=model_type, use_hands=True, use_face=True, use_face_contour=False, openpose_format='coco25')

def main(args):
    print('[INFO] recording_name:', args.recording_name)

    calib_trans_dir = os.path.join(args.release_data_root, 'calibrations', args.recording_name)  # extrinsics
    fpv_recording_dir = glob.glob(os.path.join(args.release_data_root, 'egocentric_color', args.recording_name, '202*'))[0]

    if args.model_type == 'smplx':
        fitting_root_interactee = osp.join(args.release_data_root, 'smplx_interactee', args.recording_name)
    elif args.model_type == 'smpl':
        fitting_root_interactee = osp.join(args.release_data_root, 'smpl_interactee', args.recording_name)
    else:
        print('body model type error!')
        exit()

    ################################## read recording info
    df = pd.read_csv(os.path.join(args.release_data_root, 'data_info_release.csv'))
    recording_name_list = list(df['recording_name'])
    start_frame_list = list(df['start_frame'])
    end_frame_list = list(df['end_frame'])
    body_idx_fpv_list = list(df['body_idx_fpv'])

    body_idx_fpv_dict = dict(zip(recording_name_list, body_idx_fpv_list))
    start_frame_dict = dict(zip(recording_name_list, start_frame_list))
    end_frame_dict = dict(zip(recording_name_list, end_frame_list))
    # get body idx, gender for the interactee
    interactee_idx = int(body_idx_fpv_dict[args.recording_name].split(' ')[0])
    interactee_gender = body_idx_fpv_dict[args.recording_name].split(' ')[1]

    ################################## create saving folders
    if args.rendering_mode == 'both' or args.rendering_mode == 'body':
        rendering_dir = os.path.join(args.save_root, 'renderings_ego_img')
        if not osp.exists(rendering_dir):
            os.mkdir(rendering_dir)
        rendering_dir = os.path.join(rendering_dir, args.recording_name)
        if not osp.exists(rendering_dir):
            os.mkdir(rendering_dir)

    if args.rendering_mode == 'both' or args.rendering_mode == '3d':
        body_scene_rendering_dir = os.path.join(args.save_root, 'renderings_ego_3d')
        if not osp.exists(body_scene_rendering_dir):
            os.mkdir(body_scene_rendering_dir)
        body_scene_rendering_dir = os.path.join(body_scene_rendering_dir, args.recording_name)
        if not osp.exists(body_scene_rendering_dir):
            os.mkdir(body_scene_rendering_dir)

        scene_dir = os.path.join(os.path.join(args.release_data_root, 'scene_mesh'), args.scene_name)
        cam2world_dir = os.path.join(calib_trans_dir, 'cal_trans/kinect12_to_world')  # transformation from master kinect RGB camera to scene mesh
        with open(os.path.join(cam2world_dir, args.scene_name + '.json'), 'r') as f:
            trans_scene_to_main = np.array(json.load(f)['trans'])
        trans_scene_to_main = np.linalg.inv(trans_scene_to_main)

    ################################## read hololens world <-> kinect master RGB cam extrinsics
    holo2kinect_dir = os.path.join(calib_trans_dir, 'cal_trans', 'holo_to_kinect12.json')
    with open(holo2kinect_dir, 'r') as f:
        trans_holo2kinect = np.array(json.load(f)['trans'])
    trans_kinect2holo = np.linalg.inv(trans_holo2kinect)


    ################################## read hololens egocentric data
    ######## holo_frame_id_dict: key: frame_id, value: pv frame img path
    holo_pv_path_list = glob.glob(os.path.join(fpv_recording_dir, 'PV', '*_frame_*.jpg'))
    holo_pv_path_list = sorted(holo_pv_path_list)
    holo_frame_id_list = [basename(x).split('.')[0].split('_', 1)[1] for x in holo_pv_path_list]
    holo_frame_id_dict = dict(zip(holo_frame_id_list, holo_pv_path_list))
    # holo_timestamp_dict: key: timestamp, value: frame id
    holo_timestamp_list = [basename(x).split('_')[0] for x in holo_pv_path_list]
    holo_timestamp_dict = dict(zip(holo_timestamp_list, holo_frame_id_list))

    ######## read valid frame info and openpose 2d joints
    valid_frame_npz = osp.join(fpv_recording_dir, 'valid_frame.npz')
    kp_npz = osp.join(fpv_recording_dir, 'keypoints.npz')
    valid_frames = np.load(valid_frame_npz)
    holo_2djoints_info = np.load(kp_npz)
    assert len(valid_frames['valid']) == len(valid_frames['imgname'])
    # read info in valid_frame.npz
    holo_frame_id_all = [basename(x).split('.')[0].split('_', 1)[1] for x in valid_frames['imgname']]
    holo_valid_dict = dict(zip(holo_frame_id_all, valid_frames['valid']))  # 'frame_01888': True
    # read info in keypoints.npz
    holo_frame_id_valid = [basename(x).split('.')[0].split('_', 1)[1] for x in holo_2djoints_info['imgname']]  # list of all valid frame names (e.x., 'frame_01888')
    holo_keypoint_dict = dict(zip(holo_frame_id_valid, holo_2djoints_info['keypoints']))

    ######## read hololens camera info
    # for each sequence: unique cx, cy, w, h
    # for each frame: different fx, fy, pv2world_transform
    pv_info_path = glob.glob(os.path.join(fpv_recording_dir, '*_pv.txt'))[0]
    with open(pv_info_path) as f:
        lines = f.readlines()
    holo_cx, holo_cy, holo_w, holo_h = ast.literal_eval(lines[0])  # hololens pv camera infomation

    holo_fx_dict = {}
    holo_fy_dict = {}
    holo_pv2world_trans_dict = {}
    for i, frame in enumerate(lines[1:]):
        frame = frame.split((','))
        cur_timestamp = frame[0]  # string
        cur_fx = float(frame[1])
        cur_fy = float(frame[2])
        cur_pv2world_transform = np.array(frame[3:20]).astype(float).reshape((4, 4))

        if cur_timestamp in holo_timestamp_dict.keys():
            cur_frame_id = holo_timestamp_dict[cur_timestamp]
            holo_fx_dict[cur_frame_id] = cur_fx
            holo_fy_dict[cur_frame_id] = cur_fy
            holo_pv2world_trans_dict[cur_frame_id] = cur_pv2world_transform


    ######## read gaze data
    # load head, hand, eye data
    if args.plot_gaze:
        gaze_dir = glob.glob(os.path.join(args.release_data_root, 'egocentric_gaze', args.recording_name, '202*'))[0]
        holo_gaze_file_path = glob.glob(os.path.join(gaze_dir, '*_head_hand_eye.csv'))[0]
        holo_gaze_point3d_dict = {}
        holo_gaze_point2d_dict = {}
        (timestamps, _, gaze_data, gaze_available) = load_head_hand_eye_data(holo_gaze_file_path)
        for pv_timestamp in holo_timestamp_dict.keys():
            gaze_ts = match_timestamp(int(pv_timestamp), timestamps)

            if gaze_available[gaze_ts]:
                point, origin_homog, direction_homog, dist = get_eye_gaze_point(gaze_data[gaze_ts])  # 3D point in holo world coord
                # if pv_timestamp in holo_timestamp_dict.keys():
                cur_frame_id = holo_timestamp_dict[pv_timestamp]
                holo_gaze_point3d_dict[cur_frame_id] = point
                # project to 2D
                K = np.array([[holo_fx_dict[cur_frame_id], 0, holo_cx],
                              [0, holo_fy_dict[cur_frame_id], holo_cy],
                              [0, 0, 1]])
                try:
                    Rt = np.linalg.inv(holo_pv2world_trans_dict[cur_frame_id])
                except np.linalg.LinAlgError:
                    print('No pv2world transform')
                    continue
                rvec, _ = cv2.Rodrigues(Rt[:3, :3])
                tvec = Rt[:3, 3]
                xy, _ = cv2.projectPoints(point.reshape((1, 3)), rvec, tvec, K, None)
                ixy = (int(xy[0][0][0]), int(xy[0][0][1]))
                ixy = (1920 - ixy[0], ixy[1])
                holo_gaze_point2d_dict[cur_frame_id] = ixy  # 2D coord for gaze in pv img


    ################################## create smplx/smpl body model
    joint_mapper = JointMapper(get_model2data(model_type=args.model_type))
    if args.model_type == 'smplx':
        body_model = smplx.create(os.path.join(args.model_folder, 'smplx_model'), model_type='smplx',
                                  joint_mapper=joint_mapper, gender=interactee_gender, ext='npz', num_pca_comps=args.num_pca_comps,
                                  create_global_orient=True, create_body_pose=True, create_betas=True, create_transl=True,
                                  create_left_hand_pose=True, create_right_hand_pose=True,
                                  create_expression=True, create_jaw_pose=True, create_leye_pose=True, create_reye_pose=True, ).to(device)
    elif args.model_type == 'smpl':
        body_model = smplx.create(args.model_folder, model_type='smpl', gender=interactee_gender).to(device)


    ################################## start rendering
    H, W = 1080, 1920
    camera_center = np.array([holo_cx, holo_cy])
    camera_pose = np.eye(4)
    camera_pose = np.array([1.0, 1.0, 1.0, 1.0]).reshape(-1, 1) * camera_pose
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)


    base_color = (1.0, 193/255, 193/255, 1.0)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=base_color
        )

    for i_frame in tqdm(range(start_frame_dict[args.recording_name], end_frame_dict[args.recording_name]+1)[args.start::args.step]):
        holo_frame_id = 'frame_{}'.format("%05d" % i_frame)

        if not osp.exists(osp.join(fitting_root_interactee, 'body_idx_{}'.format(interactee_idx), 'results', holo_frame_id, '000.pkl')):
            print('interactee fitting {} do not exist!'.format(holo_frame_id))
            continue

        with open(osp.join(fitting_root_interactee, 'body_idx_{}'.format(interactee_idx), 'results', holo_frame_id, '000.pkl'), 'rb') as f:
            param = pickle.load(f)
        torch_param = {}
        if args.model_type == 'smpl':
            torch_param['transl'] = torch.tensor(param['transl']).to(device)
            torch_param['global_orient'] = torch.tensor(param['global_orient']).to(device)
            torch_param['betas'] = torch.tensor(param['betas']).to(device)
            torch_param['body_pose'] = torch.tensor(param['body_pose']).to(device)
        elif args.model_type == 'smplx':
            for key in param.keys():
                if key in ['pose_embedding', 'camera_rotation', 'camera_translation', 'gender']:
                    continue
                else:
                    torch_param[key] = torch.tensor(param[key]).to(device)
        output = body_model(return_verts=True, **torch_param)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        joints = output.joints.detach().cpu().numpy().squeeze()  # in openpose topology
        body = trimesh.Trimesh(vertices, body_model.faces, process=False)

        if holo_frame_id not in holo_frame_id_dict.keys():  # the frame is dropped in hololens recording
            pass
        else:
            fpv_img_path = os.path.join(fpv_recording_dir, 'PV', holo_frame_id_dict[holo_frame_id])
            cur_fx = holo_fx_dict[holo_frame_id]
            cur_fy = holo_fy_dict[holo_frame_id]
            cur_pv2world_transform = holo_pv2world_trans_dict[holo_frame_id]
            cur_world2pv_transform = np.linalg.inv(cur_pv2world_transform)

            camera = pyrender.camera.IntrinsicsCamera(
                fx=cur_fx, fy=cur_fy,
                cx=camera_center[0], cy=camera_center[1])

            body.apply_transform(trans_kinect2holo)  # master kinect RGB coord --> hololens world coord
            body.apply_transform(cur_world2pv_transform)  # hololens world coord --> current frame hololens pv(RGB) coordinate
            body_mesh = pyrender.Mesh.from_trimesh(body, material=material)
            # if save_meshes:
            #     body.export(osp.join(meshes_dir,img_name, '000.ply'))

            if args.rendering_mode == 'body' or args.rendering_mode == 'both':
                img = cv2.imread(fpv_img_path)[:, :, ::-1]
                scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                       ambient_light=(0.3, 0.3, 0.3))
                scene.add(camera, pose=camera_pose)
                scene.add(light, pose=camera_pose)
                scene.add(body_mesh, 'mesh')
                r = pyrender.OffscreenRenderer(viewport_width=W,
                                               viewport_height=H,
                                               point_size=1.0)
                color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)

                color = color.astype(np.float32) / 255.0
                alpha = 1.0  # set transparency in [0.0, 1.0]
                color[:, :, -1] = color[:, :, -1] * alpha
                color = pil_img.fromarray((color * 255).astype(np.uint8))
                output_img = pil_img.fromarray((img).astype(np.uint8))
                output_img.paste(color, (0, 0), color)

                ############ get gaze heatmap
                if args.plot_gaze and holo_frame_id in holo_gaze_point2d_dict.keys():
                    gaze_heatmap = draw_gaze_heatmap_2d(H, W, holo_gaze_point2d_dict, holo_frame_id)
                    # cv2.imshow('xxx', gaze_heatmap)
                    # cv2.waitKey(-1)
                    gaze_heatmap = pil_img.fromarray(gaze_heatmap)
                    output_img.paste(gaze_heatmap, (0, 0), gaze_heatmap)

                ############# project 3d joints to pv img
                # if it is a valid hololens frame (visible 2d joints >=6),
                if (holo_frame_id in holo_valid_dict.keys()) and holo_valid_dict[holo_frame_id] and args.plot_2d_joints:
                    joints = points_coord_trans(joints, trans_kinect2holo)  # gt 3d joints in hololens world coord
                    joints = points_coord_trans(joints, cur_world2pv_transform)  # gt 3d joints in current hololens PV(RGB) camera coord
                    add_trans = np.array([[1.0, 0, 0, 0],
                                          [0, -1, 0, 0],
                                          [0, 0, -1, 0],
                                          [0, 0, 0, 1]])  # different y/z axis definition in opencv/opengl convention
                    joints = points_coord_trans(joints, add_trans)  # gt 3d joints in current hololens PV(RGB) camera coord, [n_joints, 3]

                    camera_center_holo = torch.tensor([holo_cx, holo_cy]).view(-1, 2)
                    camera_holo_kp = create_camera(camera_type='persp_holo',
                                                   focal_length_x=torch.tensor([cur_fx]).to(device).unsqueeze(0),
                                                   focal_length_y=torch.tensor([cur_fy]).to(device).unsqueeze(0),
                                                   center=camera_center_holo,
                                                   batch_size=1).to(device=device)

                    joints = torch.from_numpy(joints).float().to(device).unsqueeze(0)  # [1, n_joints, 3]
                    gt_joints_2d = camera_holo_kp(joints)  # project 2d joints on holo images of gt body [1, n_joints, 2]
                    gt_joints_2d = gt_joints_2d.squeeze().detach().cpu().numpy()  # [n_joints, 2]
                    keypoints_holo = holo_keypoint_dict[holo_frame_id]  # [25, 3] openpose detections

                    draw = ImageDraw.Draw(output_img)
                    for k in range(25):
                        draw.ellipse((gt_joints_2d[k][0] - 4, gt_joints_2d[k][1] - 4,
                                      gt_joints_2d[k][0] + 4, gt_joints_2d[k][1] + 4), fill=(0, 255, 0, 0))
                    for k in range(25):
                        draw.ellipse((keypoints_holo[k][0] - 4, keypoints_holo[k][1] - 4,
                                      keypoints_holo[k][0] + 4, keypoints_holo[k][1] + 4), fill=(255, 0, 0, 0))

                output_img.convert('RGB')
                output_img = output_img.resize((int(W / args.scale), int(H / args.scale)))
                output_img.save(os.path.join(rendering_dir, 'holo_' + holo_frame_id + '_output.jpg'))


            if args.rendering_mode == '3d' or args.rendering_mode == 'both':
                static_scene = trimesh.load(osp.join(scene_dir, args.scene_name + '.obj'), enable_post_processing=True, print_progress=True)
                static_scene.apply_transform(trans_scene_to_main)  # 3d scene coord --> master kinect RGB cam coord
                static_scene.apply_transform(trans_kinect2holo)  # master kinect RGB cam coord --> hololens world coord
                static_scene.apply_transform(cur_world2pv_transform)  # hololens world coord --> current frame hololens pv(RGB) coordinate
                static_scene_mesh = pyrender.Mesh.from_trimesh(static_scene)

                scene = pyrender.Scene()
                scene.add(camera, pose=camera_pose)
                scene.add(light, pose=camera_pose)
                scene.add(static_scene_mesh, 'scene_mesh')
                body_mesh = pyrender.Mesh.from_trimesh(body, material=material)
                scene.add(body_mesh, 'body_mesh')
                r = pyrender.OffscreenRenderer(viewport_width=W,
                                               viewport_height=H)
                color, _ = r.render(scene)
                color = pil_img.fromarray(color)

                ############ get gaze heatmap
                if args.plot_gaze and holo_frame_id in holo_gaze_point2d_dict.keys():
                    gaze_heatmap = draw_gaze_heatmap_2d(H, W, holo_gaze_point2d_dict, holo_frame_id)
                    gaze_heatmap = pil_img.fromarray(gaze_heatmap)
                    color.paste(gaze_heatmap, (0, 0), gaze_heatmap)

                color = color.resize((int(W / args.scale), int(H / args.scale)))
                color.save(os.path.join(body_scene_rendering_dir, 'holo_' + holo_frame_id + '_output.jpg'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--release_data_root', type=str, default='/mnt/ssd/egobody_release', help='path to egobody dataset')
    parser.add_argument('--save_root', type=str, default='./', help='path to save renderings')
    parser.add_argument('--recording_name', type=str, default='recording_20220318_S33_S34_01')
    parser.add_argument('--scene_name', type=str, default='cnb_dlab_0225')

    parser.add_argument('--model_type', type=str, default='smplx', choices=['smplx', 'smpl'])

    parser.add_argument('--plot_2d_joints', default='False', type=lambda x: x.lower() in ['true', '1'],
                        help='draw gt/openpose 2d joints on rendered images or not, always disabled when rendering in 3d scenes')
    parser.add_argument('--plot_gaze', default='False', type=lambda x: x.lower() in ['true', '1'], help='draw 2d gaze or not')

    parser.add_argument('--scale', type=int, default=2, help='the scale to downsample output rendering images')
    parser.add_argument('--start', type=int, default=0, help='id of the starting frame')
    parser.add_argument('--step', type=int, default=1, help='downsample framerate')
    parser.add_argument('--model_folder', default='/mnt/hdd/PROX/body_models', type=str, help='path to smpl/smplx models')
    parser.add_argument('--num_pca_comps', type=int, default=12)
    parser.add_argument('--rendering_mode', default='body', type=str, choices=['body', '3d', 'both'],
                        help='body: render gt body on egocentric images; 3d: render gt body in 3d scenes')

    args = parser.parse_args()
    main(args)
