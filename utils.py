import numpy as np
import cv2
import ast
import torch
import torch.nn as nn
from smplx.lbs import transform_mat


def row(A):
    return A.reshape((1, -1))

def col(A):
    return A.reshape((-1, 1))

def unproject_depth_image(depth_image, cam):
    us = np.arange(depth_image.size) % depth_image.shape[1]  # (w*h,)  [0,1,2,...,640, ..., 0,1,2,...,w]
    vs = np.arange(depth_image.size) // depth_image.shape[1]  # (w*h,)  [0,0,...,0, ..., 576,576,...,h]
    ds = depth_image.ravel()  # (w*h,) return flatten depth_image (still the same memory, not a copy)
    uvd = np.array(np.vstack((us.ravel(), vs.ravel(), ds.ravel())).T)  # [w*h, 3]
    # undistort depth map
    xy_undistorted_camspace = cv2.undistortPoints(np.asarray(uvd[:, :2].reshape((1, -1, 2)).copy()),
                                                  np.asarray(cam['camera_mtx']), np.asarray(cam['k']))
    # unproject to 3d points in depth cam coord
    xyz_camera_space = np.hstack((xy_undistorted_camspace.squeeze(), col(uvd[:, 2])))  # [w*h, 3]
    xyz_camera_space[:, :2] *= col(xyz_camera_space[:, 2])  # scale x,y by z, --> 3d coordinates in depth camera coordinate
    return xyz_camera_space   # [w*h, 3]


def points_coord_trans(xyz_source_coord, trans_mtx):
    # trans_mtx: sourceCoord_2_targetCoord, same as trans in open3d pcd.transform(trans)
    xyz_target_coord = xyz_source_coord.dot(trans_mtx[:3, :3].transpose())  # [N, 3]
    xyz_target_coord = xyz_target_coord + row(trans_mtx[:3, 3])
    return xyz_target_coord


def projectPoints(v, cam):
    v = v.reshape((-1, 3)).copy()
    return cv2.projectPoints(v, np.asarray([[0.0,0.0,0.0]]), np.asarray([0.0,0.0,0.0]), np.asarray(cam['camera_mtx']),
                             np.asarray(cam['k']))[0].squeeze()


def get_valid_idx(points_color_coord, color_cam, TH=1e-2):
    # 3D points --> 2D coordinates in color image
    uvs = projectPoints(points_color_coord, color_cam)  # [n_depth_points, 2]
    uvs = np.round(uvs).astype(int)
    valid_x = np.logical_and(uvs[:, 1] >= 0, uvs[:, 1] < 1080)  # [n_depth_points], true/false
    valid_y = np.logical_and(uvs[:, 0] >= 0, uvs[:, 0] < 1920)
    valid_idx = np.logical_and(valid_x, valid_y)  # [n_depth_points], true/false
    valid_idx = np.logical_and(valid_idx, points_color_coord[:, 2] > TH)
    uvs = uvs[valid_idx == True]  # valid 2d coords in color img of 3d depth points
    return valid_idx, uvs


def load_pv_data(csv_path):
    # load camera params, RGB frame timestamps of hololens data
    with open(csv_path) as f:
        lines = f.readlines()

    # The first line contains info about the intrinsics.
    # The following lines (one per frame) contain timestamp, focal length and transform PVtoWorld
    n_frames = len(lines) - 1
    frame_timestamps = np.zeros(n_frames, dtype=np.longlong)
    focal_lengths = np.zeros((n_frames, 2))
    pv2world_transforms = np.zeros((n_frames, 4, 4))

    intrinsics_ox, intrinsics_oy, \
        intrinsics_width, intrinsics_height = ast.literal_eval(lines[0])

    for i_frame, frame in enumerate(lines[1:]):
        # Row format is timestamp, focal length (2), transform PVtoWorld (4x4)
        frame = frame.split(',')
        frame_timestamps[i_frame] = int(frame[0])
        focal_lengths[i_frame, 0] = float(frame[1])
        focal_lengths[i_frame, 1] = float(frame[2])
        pv2world_transforms[i_frame] = np.array(frame[3:20]).astype(float).reshape((4, 4))

    return (frame_timestamps, focal_lengths, pv2world_transforms,
            intrinsics_ox, intrinsics_oy, intrinsics_width, intrinsics_height)


def load_head_hand_eye_data(csv_path):
    joint_count = 26

    # load head and eye tracking of hololens data
    data = np.loadtxt(csv_path, delimiter=',')

    n_frames = len(data)
    timestamps = np.zeros(n_frames)
    head_transs = np.zeros((n_frames, 3))

    left_hand_transs = np.zeros((n_frames, joint_count, 3))
    left_hand_transs_available = np.ones(n_frames, dtype=bool)
    right_hand_transs = np.zeros((n_frames, joint_count, 3))
    right_hand_transs_available = np.ones(n_frames, dtype=bool)

    # origin (vector, homog) + direction (vector, homog) + distance (scalar)
    gaze_data = np.zeros((n_frames, 9))
    gaze_available = np.ones(n_frames, dtype=bool)

    for i_frame, frame in enumerate(data):
        timestamps[i_frame] = frame[0]
        # head
        head_transs[i_frame, :] = frame[1:17].reshape((4, 4))[:3, 3]

        # left hand
        left_hand_transs_available[i_frame] = (frame[17] == 1)
        left_start_id = 18
        for i_j in range(joint_count):
            j_start_id = left_start_id + 16 * i_j
            j_trans = frame[j_start_id:j_start_id + 16].reshape((4, 4))[:3, 3]
            left_hand_transs[i_frame, i_j, :] = j_trans
        # right hand
        right_hand_transs_available[i_frame] = (frame[left_start_id + joint_count * 4 * 4] == 1)
        right_start_id = left_start_id + joint_count * 4 * 4 + 1
        for i_j in range(joint_count):
            j_start_id = right_start_id + 16 * i_j
            j_trans = frame[j_start_id:j_start_id + 16].reshape((4, 4))[:3, 3]
            right_hand_transs[i_frame, i_j, :] = j_trans

        # assert(j_start_id + 16 == 851)
        gaze_available[i_frame] = (frame[851] == 1)
        gaze_data[i_frame, :4] = frame[852:856]
        gaze_data[i_frame, 4:8] = frame[856:860]
        gaze_data[i_frame, 8] = frame[860]

    return (timestamps, head_transs, left_hand_transs, left_hand_transs_available,
            right_hand_transs, right_hand_transs_available, gaze_data, gaze_available)
    # return (timestamps, head_transs, gaze_data, gaze_available)



def get_eye_gaze_point(gaze_data):
    origin_homog = gaze_data[:4]
    direction_homog = gaze_data[4:8]
    direction_homog = direction_homog / np.linalg.norm(direction_homog)
    # if no distance was recorded, set 1m by default
    dist = gaze_data[8] if gaze_data[8] > 0.0 else 1.0
    point = origin_homog + direction_homog * dist
    return point[:3], origin_homog, direction_homog, dist


def match_timestamp(target, all_timestamps):
    return np.argmin([abs(x - target) for x in all_timestamps])


def draw_gaze_heatmap_2d(H=1080, W=1920, holo_gaze_point2d_dict=None, holo_frame_id=None):
    gaze_heatmap = np.zeros([H, W])
    # color: (1080, 1920, 3)
    us = np.arange(H * W) % W
    vs = np.arange(H * W) // W
    gaze_u = int(holo_gaze_point2d_dict[holo_frame_id][0])
    gaze_v = int(holo_gaze_point2d_dict[holo_frame_id][1])
    gaze_visible = False
    if gaze_u < 1920 and gaze_u > 0 and gaze_v < 1080 and gaze_u > 0:
        gaze_visible = True
        d = (us - gaze_u) ** 2 + (vs - gaze_v) ** 2
        d = d ** 0.5
        d[d > 150] = 150
        # assert np.min(d) == 0
        d = d / np.max(d)  # in [0,1]
        d = 1 - d
        gaze_heatmap = d.reshape([H, W])

    gaze_heatmap = np.uint8(255 * gaze_heatmap)
    gaze_heatmap = cv2.applyColorMap(gaze_heatmap, cv2.COLORMAP_JET)
    # turn into red headmap
    gaze_heatmap[:, :, -1] = 255
    gaze_heatmap[:, :, 0] = 0
    gaze_heatmap[:, :, 1] = 0

    gaze_heatmap = gaze_heatmap[:, :, ::-1]
    gaze_heatmap = cv2.cvtColor(gaze_heatmap, cv2.COLOR_RGB2RGBA)
    if gaze_visible:
        gaze_heatmap[:, :, -1] = d.reshape([H, W]) * 255  # set alpha by distance from gaze
    else:
        gaze_heatmap[:, :, -1] = 0
    gaze_heatmap[:, :, -1] = gaze_heatmap[:, :, -1] * 0.7  # numpy array
    return gaze_heatmap

# cite from https://github.com/mohamedhassanmus/prox/tree/master/prox
# topology transformation between smpl/smpx/smplh and openpose joints
def smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                     use_face_contour=False, openpose_format='coco25'):
    ''' Returns the indices of the permutation that maps OpenPose to SMPL

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'

    '''
    if openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            # ex: body_mapping[0]=55: smplx joint 55 = openpose joint 0
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)   # len of 25
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)  # 21 joints for each hand
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)   # len of 51
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 +
                                         17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    else:
        raise ValueError('Unknown joint format: {}'.format(openpose_format))


class JointMapper(nn.Module):
    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer('joint_maps',
                                 torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)


def create_camera(camera_type='persp', **kwargs):
    # if camera_type.lower() == 'persp':
    #     return PerspectiveCamera(**kwargs)
    if camera_type.lower() == 'persp_holo':
        return PerspectiveCamera_holo(**kwargs)
    else:
        raise ValueError('Uknown camera type: {}'.format(camera_type))

class PerspectiveCamera_holo(nn.Module):

    FOCAL_LENGTH = 5000

    def __init__(self, rotation=None, translation=None,
                 focal_length_x=None, focal_length_y=None,
                 batch_size=1,
                 center=None, dtype=torch.float32, **kwargs):
        super(PerspectiveCamera_holo, self).__init__()
        self.batch_size = batch_size
        self.dtype = dtype
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix
        self.register_buffer('zero', torch.zeros([batch_size], dtype=dtype))

        self.register_buffer('focal_length_x', focal_length_x)    # Adds a persistent buffer to the module
        self.register_buffer('focal_length_y', focal_length_y)

        if center is None:
            center = torch.zeros([batch_size, 2], dtype=dtype)
        self.register_buffer('center', center)     # [bs, 2]

        if rotation is None:
            rotation = torch.eye(
                3, dtype=dtype).unsqueeze(dim=0).repeat(batch_size, 1, 1)    # [bs, 3, 3]

        rotation = nn.Parameter(rotation, requires_grad=True)
        self.register_parameter('rotation', rotation)  # Adds a parameter to the module, shape [1,3,3],  [[1,0,0],[0,1,0],[0,0,1]]

        if translation is None:
            translation = torch.zeros([batch_size, 3], dtype=dtype)   # [bs, 3]

        translation = nn.Parameter(translation,
                                   requires_grad=True)
        self.register_parameter('translation', translation)  # all 0

    def forward(self, points):
        device = points.device  # [bs, 118, 3]

        with torch.no_grad():
            camera_mat = torch.zeros([self.batch_size, 2, 2],
                                     dtype=self.dtype, device=points.device)
            camera_mat[:, 0, 0] = self.focal_length_x  # todo, self.focal_length_x: [bs]
            camera_mat[:, 1, 1] = self.focal_length_y  # [bs, 2, 2], each batch: [[f_x, 0], [0, f_y]]

        camera_transform = transform_mat(self.rotation,
                                         self.translation.unsqueeze(dim=-1))   # [bs, 4, 4], each batch: I
        homog_coord = torch.ones(list(points.shape)[:-1] + [1],
                                 dtype=points.dtype,
                                 device=device)     # [bs, 118, 1]
        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points, homog_coord], dim=-1)   # [1, 118, 4]

        projected_points = torch.einsum('bki,bji->bjk',
                                        [camera_transform, points_h])   # [1, 118, 4]

        img_points = torch.div(projected_points[:, :, :2],
                               projected_points[:, :, 2].unsqueeze(dim=-1))  # [1, 118, 2]
        img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) + self.center.unsqueeze(dim=1)
        return img_points   # [1, 118, 2]

