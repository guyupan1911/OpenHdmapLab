import copy
import os.path as osp

import random
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

import torch
from torch.utils.data import Dataset


from mmengine.fileio import load
from mmengine.structures import BaseDataElement
# from mmengine.dataset.base_dataset import Compose
# from mmdet3d.structures.bbox_3d import get_box_type
# from mmdet3d.registry import DATASETS

from .transforms import LoadMultiViewImageFromFiles, NormalizeMultiviewImage, \
                        PadMultiViewImage, MultiScaleFlipAug3D

# @DATASETS.register_module()
class CustomNuScenesDataset(Dataset):
    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 use_valid_flag=False,
                 queue_length=4,
                 **kwargs):
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        self.with_velocity = with_velocity
        print(f'test mode: {test_mode}')

        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.queue_length = queue_length
        # self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

        self.CLASSES = self.get_classes(classes)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(osp.join(self.data_root, self.ann_file))
    
        # if pipeline is not None:
        #     self.pipeline = Compose(pipeline)
        #     print(self.pipeline)
        self.pipeline = []
        self.pipeline.append(LoadMultiViewImageFromFiles())
        self.pipeline.append(NormalizeMultiviewImage())
        self.pipeline.append(PadMultiViewImage())
        self.pipeline.append(MultiScaleFlipAug3D())


    def load_annotations(self, ann_file):
        data = load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos


    def get_data_info(self, index):
        info = self.data_infos[index]

        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts= []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                
                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsics=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))
        
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
        
        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict


    def get_ann_info(self, index):
        info = self.data_infos[index]
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []

        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        
        # gt_bboxes_3d = LiDARInstance3DBoxes(
        #     gt_bboxes_3d,
        #     box_dim=gt_bboxes_3d.shape[-1],
        #     origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)


        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results


    def pre_pipeline(self, results):
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        # results['box_type_3d'] = self.box_type_3d
        # results['box_mode_3d'] = self.box_mode_3d       


    def union2one(self, queue):
        imgs_list = [each['img'][0] for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'][0]
            # print(f'metas_map[i]: {metas_map[i][0].keys()}')
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        queue[-1]['img'] = torch.stack(imgs_list)
        queue[-1]['img_metas'] = metas_map
        queue = queue[-1]
        return queue


    def prepare_train_data(self, index):
        queue = []
        index_list = list(range(index-self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            for func in self.pipeline:
                input_dict = func(input_dict)
            # print(f'example: {example.keys()}')
            # if self.filter_empty_gt and (example is None or ~(example['gt_labels_3d']._data != -1).any()):
            #     return None
            queue.append(input_dict)
        # print(f'queue length: {len(queue)}')
        return self.union2one(queue)


    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        for func in self.pipeline:
            input_dict = func(input_dict)
        # return example
        input_dict['img'] = torch.stack(input_dict['img'])
        return input_dict


    @classmethod
    def get_classes(cls, classes=None):
        if classes is None:
            return cls.CLASSES
        
        if isinstance(classes, str):
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} if classes.')
        
        return class_names
    

    def __len__(self):
        return len(self.data_infos)


    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data