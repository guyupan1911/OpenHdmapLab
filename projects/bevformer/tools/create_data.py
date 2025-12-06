#!/usr/bin/env python
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
# Create pkl files for NuScenesTemporalDataset from NuScenes dataset root
# using NuScenes API
# ---------------------------------------------
import argparse
import os
import pickle
import sys
from typing import Dict, Any, Optional, Tuple

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.data_classes import Quaternion

category2label = {
    'car': 0,
    'truck': 1,
    'trailer': 2,
    'bus': 3,
    'construction_vehicle': 4,
    'pedestrian': 5,
    'motorcycle': 6,
    'bicycle': 7,
    'traffic_cone': 8,
    'barrier': 9
}

def get_sensor2top(
    lidar2ego_translation: list,
    lidar2ego_rotation: list,
    lidar_ego2global_translation: list,
    lidar_ego2global_rotation: list,
    cam2ego_translation: list,
    cam2ego_rotation: list,
    cam_ego2global_translation: list,
    cam_ego2global_rotation: list) -> Tuple[np.ndarray, np.ndarray]:
    """Compute transformation matrix from LiDAR to camera.
    
    Since LiDAR and camera may have different ego_pose (different timestamps),
    we need to transform through global coordinate system:
    lidar -> lidar_ego -> global -> cam_ego -> cam
    
    Args:
        lidar2ego_translation (list): Translation from LiDAR to LiDAR's Ego [x, y, z].
        lidar2ego_rotation (list): Rotation quaternion from LiDAR to LiDAR's Ego [w, x, y, z].
        lidar_ego2global_translation (list): Translation from LiDAR's Ego to Global [x, y, z].
        lidar_ego2global_rotation (list): Rotation quaternion from LiDAR's Ego to Global [w, x, y, z].
        cam2ego_translation (list): Translation from Camera to Camera's Ego [x, y, z].
        cam2ego_rotation (list): Rotation quaternion from Camera to Camera's Ego [w, x, y, z].
        cam_ego2global_translation (list): Translation from Camera's Ego to Global [x, y, z].
        cam_ego2global_rotation (list): Rotation quaternion from Camera's Ego to Global [w, x, y, z].
        
    Returns:
        tuple: (lidar2cam_rotation (3x3), lidar2cam_translation (3,))
            Transformation from LiDAR to Camera coordinate system.
    """
    # Convert quaternions to rotation matrices
    lidar2ego_quat = Quaternion(lidar2ego_rotation)
    lidar2ego_rot = lidar2ego_quat.rotation_matrix  # 3x3
    
    lidar_ego2global_quat = Quaternion(lidar_ego2global_rotation)
    lidar_ego2global_rot = lidar_ego2global_quat.rotation_matrix  # 3x3
    
    cam2ego_quat = Quaternion(cam2ego_rotation)
    cam2ego_rot = cam2ego_quat.rotation_matrix  # 3x3
    
    cam_ego2global_quat = Quaternion(cam_ego2global_rotation)
    cam_ego2global_rot = cam_ego2global_quat.rotation_matrix  # 3x3
    
    # Convert to numpy arrays
    lidar2ego_trans = np.array(lidar2ego_translation)
    lidar_ego2global_trans = np.array(lidar_ego2global_translation)
    cam2ego_trans = np.array(cam2ego_translation)
    cam_ego2global_trans = np.array(cam_ego2global_translation)
    
    # Transform chain: lidar -> lidar_ego -> global -> cam_ego -> cam
    # Step 1: lidar -> lidar_ego -> global
    lidar2global_rot = lidar_ego2global_rot @ lidar2ego_rot
    lidar2global_trans = lidar_ego2global_rot @ lidar2ego_trans + lidar_ego2global_trans
    
    # Step 2: global -> cam_ego (inverse of cam_ego2global)
    global2cam_ego_rot = cam_ego2global_rot.T
    global2cam_ego_trans = -global2cam_ego_rot @ cam_ego2global_trans
    
    # Step 3: cam_ego -> cam (inverse of cam2ego)
    cam_ego2cam_rot = cam2ego_rot.T
    cam_ego2cam_trans = -cam_ego2cam_rot @ cam2ego_trans
    
    # Combine: lidar -> global -> cam_ego -> cam
    # Rotation: R_lidar2cam = R_cam_ego2cam @ R_global2cam_ego @ R_lidar2global
    lidar2cam_rot = cam_ego2cam_rot @ global2cam_ego_rot @ lidar2global_rot
    # Translation: t_lidar2cam = R_cam_ego2cam @ (R_global2cam_ego @ t_lidar2global + t_global2cam_ego) + t_cam_ego2cam
    lidar2cam_trans = cam_ego2cam_rot @ (global2cam_ego_rot @ lidar2global_trans + global2cam_ego_trans) + cam_ego2cam_trans
    
    return lidar2cam_rot, lidar2cam_trans

def get_sample_data_info(
    nusc: NuScenes,
    sample_token: str,
    data_root: str) -> Dict[str, Any]:
    """Get data info for a single sample using NuScenes API.
    
    Args:
        nusc (NuScenes): NuScenes instance.
        sample_token (str): Sample token.
        data_root (str): Root directory of the dataset.
        
    Returns:
        dict: Data info dictionary.
    """
    sample = nusc.get('sample', sample_token)
    scene_token = sample['scene_token']
    
    # Get lidar sample data
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_sd = nusc.get('sample_data', lidar_token)
    lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    
    lidar_filename = lidar_sd['filename']
    if '/' in lidar_filename:
        lidar_filename = lidar_filename.split('/')[-1]
    lidar_points = {}
    lidar_points['lidar_path'] = lidar_filename

    info = {
        'token': sample_token,
        'scene_token': scene_token,
        'timestamp': sample['timestamp'],
        'lidar_points': lidar_points,
        'lidar2ego_translation': lidar_cs['translation'],
        'lidar2ego_rotation': lidar_cs['rotation'],
        'ego2global_translation': lidar_pose['translation'],
        'ego2global_rotation': lidar_pose['rotation'],
    }
    
    # Camera info - mmdet3d requires 'images' field
    images = {}
    for channel in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
        if channel in sample['data']:
            cam_token = sample['data'][channel]
            cam_sd = nusc.get('sample_data', cam_token)
            cam_cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
            cam_pose = nusc.get('ego_pose', cam_sd['ego_pose_token'])

            # Compute LiDAR to camera transformation matrix
            # Note: lidar_pose and cam_pose may be different (different timestamps)
            # So we need to transform through global coordinate system
            lidar2cam_rot, lidar2cam_trans = get_sensor2top(
                lidar2ego_translation=lidar_cs['translation'],
                lidar2ego_rotation=lidar_cs['rotation'],
                lidar_ego2global_translation=lidar_pose['translation'],
                lidar_ego2global_rotation=lidar_pose['rotation'],
                cam2ego_translation=cam_cs['translation'],
                cam2ego_rotation=cam_cs['rotation'],
                cam_ego2global_translation=cam_pose['translation'],
                cam_ego2global_rotation=cam_pose['rotation']
            )
            
            # Build lidar2cam matrix (4x4 homogeneous transformation matrix)
            lidar2cam = np.eye(4, dtype=np.float32)
            lidar2cam[:3, :3] = lidar2cam_rot
            lidar2cam[:3, 3] = lidar2cam_trans

            cam_filename = cam_sd['filename']
            if '/' in cam_filename:
                cam_filename = cam_filename.split('/')[-1]

            images[channel] = {
                'img_path': cam_filename,
                'sample_data_token': cam_sd['token'],
                'sensor2ego_translation': cam_cs['translation'],
                'sensor2ego_rotation': cam_cs['rotation'],
                'ego2global_translation': cam_pose['translation'],
                'ego2global_rotation': cam_pose['rotation'],
                'cam2img': cam_cs['camera_intrinsic'],
                'lidar2cam': lidar2cam,  # 4x4 transformation matrix
            }
    info['images'] = images
    
    
    
    # Sweeps info (historical frames)
    sweeps = []
    current_sd = lidar_sd
    while current_sd['prev'] != '':
        prev_token = current_sd['prev']
        prev_sd = nusc.get('sample_data', prev_token)
        prev_cs = nusc.get('calibrated_sensor', prev_sd['calibrated_sensor_token'])
        prev_pose = nusc.get('ego_pose', prev_sd['ego_pose_token'])
        
        prev_filename = prev_sd['filename']
        if '/' in prev_filename:
            prev_filename = prev_filename.split('/')[-1]

        sweeps.append({
            'lidar_path': prev_filename,
            'sample_data_token': prev_sd['token'],
            'sensor2ego_translation': prev_cs['translation'],
            'sensor2ego_rotation': prev_cs['rotation'],
            'ego2global_translation': prev_pose['translation'],
            'ego2global_rotation': prev_pose['rotation'],
            'timestamp': prev_sd['timestamp'],
        })
        current_sd = prev_sd
    info['sweeps'] = sweeps
    
    info['frame_idx'] = 0  # Will be set based on scene position
    
    # Annotations - mmdet3d parse_ann_info expects 'instances' format
    ann_tokens = sample['anns']
    instances = []

    for i, ann_token in enumerate(ann_tokens):
        try:
            ann = nusc.get('sample_annotation', ann_token)
            
            category_name = ann.get('category_name', None)
            if category_name is None:
                continue      

            box = nusc.get_box(ann_token)
            
            # Remove prefix (e.g., 'vehicle.car' -> 'car', 'human.pedestrian.adult' -> 'pedestrian')
            if '.' in category_name:
                # Take the last part after the last dot
                parts = category_name.split('.')
                # For most cases, take the last part
                # But for 'human.pedestrian.adult', we want 'pedestrian'
                if len(parts) >= 2 and parts[0] == 'human' and parts[1] == 'pedestrian':
                    category_name = 'pedestrian'
                elif len(parts) >= 2 and parts[0] == 'vehicle':
                    category_name = parts[-1]  # 'vehicle.car' -> 'car'
                else:
                    category_name = parts[-1]  # Fallback: take last part
            
            # Get velocity, handle None case
            try:
                velocity = nusc.box_velocity(ann_token)
                if velocity is not None:
                    velocity_2d = velocity[:2].tolist()
                else:
                    velocity_2d = [0.0, 0.0]
            except Exception:
                velocity_2d = [0.0, 0.0]
            
            # Format: [x, y, z, w, l, h, yaw]
            bbox_3d = box.center.tolist() + box.wlh.tolist() + [box.orientation.yaw_pitch_roll[0]]
            
            instance = {
                'bbox_3d': bbox_3d,
                'bbox_label_3d': category2label.get(category_name, -1),  # Will be mapped to label index in parse_ann_info
                'velocity': velocity_2d,  # [vx, vy]
                'num_lidar_pts': ann.get('num_lidar_pts', 0),
                'num_radar_pts': ann.get('num_radar_pts', 0),
                'bbox_3d_isvalid': ann.get('visibility_token', '') != '',
            }
            instances.append(instance)
        except Exception as e:
            # Print error for debugging
            print(f'  Error processing annotation {i} (token: {ann_token}): {e}')
            import traceback
            traceback.print_exc()
            continue

    info['instances'] = instances
    
    return info


def create_temporal_pkl(
    data_root: str,
    output_pkl: Optional[str] = None,
    version: str = 'v1.0-trainval',
    split: str = 'train') -> bool:
    """Create pkl file for NuScenesTemporalDataset from NuScenes dataset root.
    
    Args:
        data_root (str): Root directory of NuScenes dataset.
        output_pkl (str, optional): Path to output pkl file. If None, will be 
            auto-generated as 'nuscenes_infos_temporal_{split}.pkl' in data_root.
        version (str): NuScenes dataset version. Default: 'v1.0-trainval'.
        split (str): Dataset split name (train/val/test). Default: 'train'.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    # Auto-generate output_pkl if not provided
    if output_pkl is None:
        output_pkl = os.path.join(data_root, f'nuscenes_infos_temporal_{split}.pkl')
    # Check data root
    if not os.path.exists(data_root):
        print(f"Error: Data root not found: {data_root}")
        return False
    
    # Initialize NuScenes
    print(f"Initializing NuScenes dataset (version: {version})...")
    try:
        nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
    except Exception as e:
        print(f"Error initializing NuScenes: {e}")
        return False
    
    # Get split scenes
    print(f"Getting {split} split scenes...")
    # Get scene names for the split based on version
    if version == 'v1.0-trainval':
        if split == 'train':
            split_scene_names = splits.train
        elif split == 'val':
            split_scene_names = splits.val
        elif split == 'test':
            split_scene_names = []
        else:
            print(f"Error: Unknown split: {split}")
            return False
    elif version == 'v1.0-test':
        if split == 'test':
            split_scene_names = splits.test
        else:
            print(f"Error: v1.0-test only has test split")
            return False
    elif version == 'v1.0-mini':
        if split == 'train':
            split_scene_names = splits.mini_train
        elif split == 'val':
            split_scene_names = splits.mini_val
        elif split == 'test':
            split_scene_names = []
        else:
            print(f"Error: Unknown split: {split}")
            return False
    else:
        # For other versions, try to use train/val splits
        if split == 'train':
            split_scene_names = splits.train
        elif split == 'val':
            split_scene_names = splits.val
        elif split == 'test':
            split_scene_names = splits.test if hasattr(splits, 'test') else []
        else:
            print(f"Error: Unknown split: {split}")
            return False
    
    # Get available scene names from dataset
    available_scene_names = [scene['name'] for scene in nusc.scene]
    
    # Filter to only scenes that exist in the dataset
    split_scene_names = [name for name in split_scene_names if name in available_scene_names]
    
    # Get scene tokens for the split
    scene_tokens = []
    for scene in nusc.scene:
        if scene['name'] in split_scene_names:
            scene_tokens.append(scene['token'])
    
    print(f"Found {len(scene_tokens)} scenes in {split} split")
    
    # Collect all sample tokens from scenes
    sample_tokens = []
    for scene_token in scene_tokens:
        scene = nusc.get('scene', scene_token)
        first_sample_token = scene['first_sample_token']
        sample_token = first_sample_token
        frame_idx = 0
        while sample_token != '':
            sample_tokens.append((sample_token, frame_idx))
            sample = nusc.get('sample', sample_token)
            sample_token = sample['next']
            frame_idx += 1
    
    print(f"Found {len(sample_tokens)} samples in {split} split")
    
    # Process each sample
    print("Processing samples...")
    data_list = []
    for i, (sample_token, frame_idx) in enumerate(sample_tokens):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(sample_tokens)} samples...")
        
        try:
            info = get_sample_data_info(nusc, sample_token, data_root)
            info['frame_idx'] = frame_idx
            data_list.append(info)
        except Exception as e:
            print(f"Warning: Failed to process sample {i} (token: {sample_token}): {e}")
            continue
    
    print(f"Successfully processed {len(data_list)} samples")
    
    # Create metainfo
    metainfo = {
        'classes': [
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
        ],
        'version': version,
        'split': split,
    }
    
    # Create output dict
    output_data = {
        'data_list': data_list,
        'metainfo': metainfo
    }
    
    # Save output file
    print(f"Saving output pkl file: {output_pkl}")
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_pkl)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_pkl, 'wb') as f:
            pickle.dump(output_data, f)
        print(f"✓ Successfully created: {output_pkl}")
        print(f"  - Total samples: {len(data_list)}")
        print(f"  - Metainfo: {metainfo}")
        return True
    except Exception as e:
        print(f"Error saving output file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Create pkl file for NuScenesTemporalDataset from NuScenes dataset root')
    parser.add_argument(
        'data_root',
        type=str,
        help='Root directory of NuScenes dataset')
    parser.add_argument(
        '--output-pkl',
        type=str,
        default=None,
        help='Path to output pkl file. If not provided, will be auto-generated '
             'as "nuscenes_infos_temporal_{split}.pkl" in data_root. '
             'Only used when --split is specified.')
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0-trainval',
        help='NuScenes dataset version (default: v1.0-trainval)')
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        choices=['train', 'val', 'test'],
        help='Dataset split name. If not specified, will generate both train and val splits.')
    args = parser.parse_args()
    
    # If split is specified, generate only that split
    if args.split is not None:
        success = create_temporal_pkl(
            args.data_root,
            output_pkl=args.output_pkl,
            version=args.version,
            split=args.split)
        return 0 if success else 1
    else:
        # Default: generate both train and val splits
        print("No split specified, generating both train and val splits by default...")
        success_train = create_temporal_pkl(
            args.data_root,
            output_pkl=None,
            version=args.version,
            split='train')
        
        success_val = create_temporal_pkl(
            args.data_root,
            output_pkl=None,
            version=args.version,
            split='val')
        
        return 0 if (success_train and success_val) else 1


if __name__ == '__main__':
    sys.exit(main())
