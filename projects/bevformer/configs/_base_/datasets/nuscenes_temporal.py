from projects.bevformer.datasets import NuScenesTemporalDataset

from projects.bevformer.datasets.transforms import (LoadMultiFrameData, MultiFrameWrapper,
                                                    PackMultiFrame3DDetInputs)

data_root = 'data/nuscenes'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# NOTE: class order must match the original BEVFormer checkpoints.
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
metainfo = dict(classes=class_names)

input_modality = dict(use_lidar=True, use_camera=True)
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    sweeps='sweeps/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT')

backend_args = None

test_transforms = []
train_transforms = [dict(type='mmdet3d.PhotoMetricDistortion3D')]

train_pipeline = [
    dict(type=LoadMultiFrameData,
         transforms = [
            dict(
                type='mmdet3d.LoadMultiViewImageFromFiles',
                to_float32=True,
                num_views=6,
                backend_args=backend_args),
            dict(
                type='mmdet3d.LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                backend_args=backend_args),
            # dict(
            #     type='mmdet3d.LoadPointsFromMultiSweeps',
            #     sweeps_num=10,
            #     backend_args=backend_args),
        ]),
    dict(
        type='mmdet3d.LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    # dict(type=MultiFrameWrapper, transforms=train_transforms), TODO: need check
    dict(type='mmdet3d.ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='mmdet3d.ObjectNameFilter', classes=class_names),
    dict(type=PackMultiFrame3DDetInputs, keys=('img', 'points', 'gt_bboxes_3d', 'gt_labels_3d')),
    # return
    # inputs['img'] : (temporal_queue, 6, 3, h, w)
    # inputs['points'] : list of points from (-3, -2, -1, 0)
    # data_samples['temporal_metainfos'] : dict of metainfos for each frame, key is frame_idx
]

dataset = dict(
    type=NuScenesTemporalDataset,
    data_root=data_root,
    frames=(-3,-2,-1,0),
    ann_file='nuscenes-mini_infos_train.pkl',
    pipeline=train_pipeline,
    load_type='frame_based',
    metainfo=metainfo,
    modality=input_modality,
    test_mode=False,
    data_prefix=data_prefix,
    # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
    # and box_type_3d='Depth' in sunrgbd and scannet dataset.
    box_type_3d='LiDAR',
    backend_args=backend_args,
    # Performance optimization: serialize data to cache parsed results
    # This avoids re-parsing pkl file on each epoch, significantly speeds up loading
    serialize_data=True)