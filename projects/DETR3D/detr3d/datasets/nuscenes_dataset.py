import copy
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import os.path as osp

import torch
from torch.utils.data import Dataset

from mmengine.dataset.base_dataset import Compose
from mmengine.fileio import load
from mmdet3d.structures import get_box_type


class NuScenesDataset(Dataset):

    METAINFO = {
        'classes':
        ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'),
        'version':
        'v1.0-trainval',
        'palette': [
            (255, 158, 0),  # Orange
            (255, 99, 71),  # Tomato
            (255, 140, 0),  # Darkorange
            (255, 127, 80),  # Coral
            (233, 150, 70),  # Darksalmon
            (220, 20, 60),  # Crimson
            (255, 61, 99),  # Red
            (0, 0, 230),  # Blue
            (47, 79, 79),  # Darkslategrey
            (112, 128, 144),  # Slategrey
        ]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metainfo = None,
                 data_prefix: dict = dict(pts='velodyne', img=''),
                 pipeline=None,
                 box_type_3d: str = 'LiDAR',
                 load_type: str = 'frame_based',
                 modality: dict = dict(
                    use_camera=False,
                    use_lidar=True
                 ),
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 load_eval_anns: bool = True,
                 with_velocity: bool = True,
                 use_valid_flag: bool = False,
                 **kwargs) -> None:
        self.use_valid_flag = use_valid_flag
        self.with_velocity = with_velocity
        self.load_type = load_type
        self.filter_empty_gt = filter_empty_gt
        self.load_eval_anns = load_eval_anns
        self.modality = modality

        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

        if metainfo is not None and 'classes' in metainfo:
            self.label_mapping = {
                i: -1
                for i in range(len(self.METAINFO['classes']))
            }
            self.label_mapping[-1] = -1
            for label_idx, name in enumerate(metainfo['classes']):
                ori_label = self.METAINFO['classes'].index(name)
                self.label_mapping[ori_label] = label_idx
            
            self.num_ins_per_cat = [0] * len(metainfo['classes'])
        else:
            self.label_mapping = {
                i: i
                for i in range(len(self.METAINFO['classes']))
            }
            self.label_mapping[-1] = -1

            self.num_ins_per_cat = [0] * len(self.METAINFO['classes'])
        
        self.ann_file = ann_file
        self._metainfo = copy.deepcopy(metainfo)
        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.test_mode = test_mode
        self.data_list: List[dict] = []
        
        # join path
        self._join_prefix()
        self.pipeline = Compose(pipeline)
        
        self.data_list = self.load_dataset()

    def _join_prefix(self):
        self.ann_file = osp.join(self.data_root, self.ann_file)
        for data_key, prefix in self.data_prefix.items():
            self.data_prefix[data_key] = osp.join(self.data_root, prefix)
    
    def load_dataset(self) -> List[dict]:
        annotations = load(self.ann_file)
        metainfo = annotations['metainfo']
        raw_data_list = annotations['data_list']

        data_list = []
        for raw_data_info in raw_data_list:
            data_info = self.parse_data_info(raw_data_info)
            data_list.append(data_info)
        
        return data_list
    
    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        # print(f'raw_data_info: {info.keys()}')
        if self.load_type == 'mv_image_based':
            pass
        else:
            if self.modality['use_lidar']:
                info['lidar_points']['lidar_path'] = \
                    osp.join(
                        self.data_prefix.get('pts', ''),
                        info['lidar_points']['lidar_path'])

                info['num_pts_feats'] = info['lidar_points']['num_pts_feats']
                info['lidar_path'] = info['lidar_points']['lidar_path']
                if 'lidar_sweeps' in info:
                    for sweep in info['lidar_sweeps']:
                        file_suffix = sweep['lidar_points']['lidar_path'].split(
                            os.sep)[-1]
                        if 'samples' in sweep['lidar_points']['lidar_path']:
                            sweep['lidar_points']['lidar_path'] = osp.join(
                                self.data_prefix['pts'], file_suffix)
                        else:
                            sweep['lidar_points']['lidar_path'] = osp.join(
                                self.data_prefix['sweeps'], file_suffix)

            if self.modality['use_camera']:
                for cam_id, img_info in info['images'].items():
                    if 'img_path' in img_info:
                        if cam_id in self.data_prefix:
                            cam_prefix = self.data_prefix[cam_id]
                        else:
                            cam_prefix = self.data_prefix.get('img', '')
                        img_info['img_path'] = osp.join(cam_prefix, img_info['img_path'])
            
            if self.test_mode and self.load_eval_anns:
                info['eval_ann_info'] = self.parse_ann_info(info)
    
    def parse_ann_info(self, info: dict) -> dict:
        name_mapping = {
            'bbox_label_3d': 'gt_labels_3d',
            'bbox_label': 'gt_bboxes_labels',
            'bbox': 'gt_bboxes',
            'bbox_3d': 'gt_bboxes_3d',
            'depth': 'depths',
            'center_2d': 'centers_2d',
            'attr_label': 'attr_labels',
            'velocity': 'velocities',
        }

        instances = info['instances']
        if len(instances) == 0:
            return None
        else:
            keys = list(instances[0].keys())
            ann_info = dict()
            for ann_name in keys:
                temp_anns = [item[ann_name] for item in instances]
                # map the original dataset label to training label
                if 'label' in ann_name and ann_name != 'attr_label':
                    temp_anns = [
                        self.label_mapping[item] for item in temp_anns
                    ]
                if ann_name in name_mapping:
                    mapped_ann_name = name_mapping[ann_name]
                else:
                    mapped_ann_name = ann_name

                if 'label' in ann_name:
                    temp_anns = np.array(temp_anns).astype(np.int64)
                elif ann_name in name_mapping:
                    temp_anns = np.array(temp_anns).astype(np.float32)
                else:
                    temp_anns = np.array(temp_anns)

                ann_info[mapped_ann_name] = temp_anns
            ann_info['instances'] = info['instances']

            for label in ann_info['gt_labels_3d']:
                if label != -1:
                    self.num_ins_per_cat[label] += 1
        
            