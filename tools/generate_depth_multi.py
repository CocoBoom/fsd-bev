from nuscenes import NuScenes
import json
import os
import os.path as osp
import sys
import copy

import numpy as np
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box, reduce, PointCloud
from nuscenes.utils.data_io import load_bin_file, panoptic_to_lidarseg
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.geometry_utils import points_in_box
from multiprocessing import  Process


NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }

static_attribute_tokens = ("c3246a1e22a14fcb878aa61e69ae3329", "58aa28b1c2a54dc88e169808c07331e3", 
        "5a655f9751944309a277276b8f473452", "03aa62109bf043afafdea7d875dd4f43", "4d8821270b4a47e3a8a300cbec48188e")
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
class_idx = dict()
for i, name in enumerate(class_names):
    class_idx[name] = i
filter_lidarseg_classes = tuple(NameMapping.keys())
camera_names=('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT')
# print(valid_classes)

nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes/', verbose=False)
filter_lidarseg_labels = []
for class_name in filter_lidarseg_classes:
    filter_lidarseg_labels.append(nusc.lidarseg_name2idx_mapping[class_name])

load_sweep = 1
total_sample_num = len(nusc.sample)
process_num = 8
sample_per_process = int(np.ceil(total_sample_num / process_num))

def generate(nusc, start_idx, end_idx):
    for i in tqdm(range(start_idx, end_idx)):
        sample = nusc.sample[i]
        sample_data = sample['data']
        ref_pc_token = sample_data['LIDAR_TOP']
        ref_pc_dict = nusc.get('sample_data', ref_pc_token)
        ref_pc = LidarPointCloud.from_file(osp.join(nusc.dataroot, ref_pc_dict['filename']))
        ref_pose_rec = nusc.get('ego_pose', ref_pc_dict['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_pc_dict['calibrated_sensor_token'])
        car_from_ref = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=False)
        global_from_car = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=False)
        ref_pc.transform(np.dot(global_from_car, car_from_ref))
        lidarseg_labels_filename = osp.join(nusc.dataroot, nusc.get('lidarseg', ref_pc_token)['filename'])
        ref_points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).astype(np.int16)
        assert len(ref_points_label) == ref_pc.nbr_points(), "lidarseg size not equal to lidar points"
        for i in range(len(ref_points_label)):
            if ref_points_label[i] in filter_lidarseg_labels:
                ref_points_label[i] = class_idx[NameMapping[nusc.lidarseg_idx2name_mapping[ref_points_label[i]]]]
            else:
                ref_points_label[i] = -1

        # get static boxes
        boxes = nusc.get_boxes(ref_pc_token)
        static_boxes = []
        for box in boxes:
            anno = nusc.get('sample_annotation',box.token)
            attribute_tokens = anno['attribute_tokens']
            if len(attribute_tokens) == 0 or attribute_tokens[0] in static_attribute_tokens: 
                if anno['num_lidar_pts'] < 1000:
                    static_boxes.append(box)

        # get nearby points in static boxes
        if load_sweep and len(static_boxes) > 0:
            next_sample = sample
            prev_sample = sample
            sweep_tokens = []
            sweep_timestamps = []
            for i in range(load_sweep):
                if prev_sample['prev']:
                    sweep_tokens.append(prev_sample['prev'])
                    prev_sample = nusc.get('sample', prev_sample['prev'])
                if next_sample['next']:
                    sweep_tokens.append(next_sample['next'])
                    next_sample = nusc.get('sample', next_sample['next'])

            sweep_pc = LidarPointCloud(np.zeros([4,0], dtype=np.float32))
            sweep_points_label = np.zeros(0, dtype=np.uint8)
            for sweep_token in sweep_tokens:
                # load sweep pc
                current_sample = nusc.get('sample', sweep_token)
                current_pc_token = current_sample['data']['LIDAR_TOP']
                current_pc_dict = nusc.get('sample_data', current_pc_token)
                current_pc_path = osp.join(nusc.dataroot, current_pc_dict['filename'])
                current_pc = LidarPointCloud.from_file(current_pc_path)
                lidarseg_labels_filename = osp.join(nusc.dataroot, nusc.get('lidarseg', current_pc_token)['filename'])
                current_points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)
                assert len(current_points_label) == current_pc.nbr_points(), "lidarseg size not equal to lidar points"
                filtered_points_idx=[]

                for i in range(len(current_points_label)):
                    if current_points_label[i] in filter_lidarseg_labels:
                        current_points_label[i] = class_idx[NameMapping[nusc.lidarseg_idx2name_mapping[current_points_label[i]]]]
                        filtered_points_idx.append(i)
                current_pc.points = current_pc.points[:, filtered_points_idx]
                current_points_label = current_points_label[filtered_points_idx]                            

                current_pose_rec = nusc.get('ego_pose', current_pc_dict['ego_pose_token'])
                current_cs_rec = nusc.get('calibrated_sensor', current_pc_dict['calibrated_sensor_token'])
                global_from_car = transform_matrix(current_pose_rec['translation'],
                                                   Quaternion(current_pose_rec['rotation']), inverse=False)
                car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                    inverse=False)
                trans_matrix = reduce(np.dot, [global_from_car, car_from_current])
                current_pc.transform(trans_matrix)
                sweep_pc.points = np.hstack((sweep_pc.points, current_pc.points))
                sweep_points_label = np.hstack((sweep_points_label, current_points_label))

            static_points_idx = []
            for box in static_boxes:
                points_idx = np.where(points_in_box(box, sweep_pc.points[:3, :]))[0]
                static_points_idx += list(points_idx)
            sweep_pc.points = sweep_pc.points[:,static_points_idx]
            sweep_points_label = sweep_points_label[static_points_idx]

        all_pc = copy.deepcopy(ref_pc)
        all_pc.points = np.hstack((all_pc.points, sweep_pc.points))
        all_points_label = np.hstack((ref_points_label, sweep_points_label))
        ref_or_sweep = np.hstack((np.zeros(len(ref_points_label)), np.ones(len(sweep_points_label))))

        for camera_name in camera_names:
            camera_token = sample_data[camera_name]
            cam = nusc.get('sample_data', camera_token)
            cam_filename = cam['filename']

            pc_tmp = copy.deepcopy(all_pc)
            poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
            pc_tmp.translate(-np.array(poserecord['translation']))
            pc_tmp.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)
            cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
            pc_tmp.translate(-np.array(cs_record['translation']))
            pc_tmp.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

            depths = pc_tmp.points[2, :]
            
            points = view_points(pc_tmp.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > 1.0)
            mask = np.logical_and(mask, points[0, :] > 1)
            mask = np.logical_and(mask, points[0, :] < 1600 - 1)
            mask = np.logical_and(mask, points[1, :] > 1)
            mask = np.logical_and(mask, points[1, :] < 900 - 1)
            points = points[:, mask]
            points_label_tmp = all_points_label[mask]
            ref_or_sweep_tmp = ref_or_sweep[mask] #.astype(np.int16)
            # points = np.round(points).astype(np.int16)
            # depths = np.round(depths[mask] * 100).astype(np.int16)
            depths = depths[mask]
            points_depth_label = np.concatenate([points[:2], depths[np.newaxis,:], 
                                                points_label_tmp[np.newaxis,:],
                                                ref_or_sweep_tmp[np.newaxis,:]], axis=0)
            depth_save_path = osp.join(nusc.dataroot, 'samples_depth_label_multi' + cam_filename[7:-3] + 'npy')
            if not osp.exists(osp.dirname(depth_save_path)):
                os.makedirs(osp.dirname(depth_save_path))
            np.save(depth_save_path, points_depth_label)

if __name__=='__main__':
    process_list = []
    for i in range(process_num):
        start_idx = i * sample_per_process
        end_idx = start_idx + sample_per_process
        if end_idx > total_sample_num: end_idx = total_sample_num
        p=Process(target=generate, args=(nusc, start_idx, end_idx))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()
