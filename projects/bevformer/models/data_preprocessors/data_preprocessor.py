from typing import Dict, List, Union

import torch
from torch import Tensor
from mmdet3d.models import Det3DDataPreprocessor
from mmhdmap.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList


@MODELS.register_module()
class TemporalDet3DDataPreprocessor(Det3DDataPreprocessor):

    def simple_process(self, data: dict, training: bool = False) -> dict:
        if 'img' in data['inputs']:
            batch_imgs_5d = data['inputs']['img']
            assert isinstance(batch_imgs_5d, list) and len(batch_imgs_5d) > 0
            assert batch_imgs_5d[0].dim() == 5, \
                f'Except img dim==5, but got {batch_imgs_5d[0].shape}'

            batch_data_samples = data['data_samples']
            assert isinstance(batch_data_samples, list) and \
                len(batch_data_samples) == len(batch_imgs_5d)

            T, num_cams, C, H, W = batch_imgs_5d[0].shape

            flattened_imgs: List[Tensor] = []
            for imgs in batch_imgs_5d:
                assert imgs.shape[:2] == (T, num_cams), \
                    f'All samples should have same T and num_cams, ' \
                    f'but got {imgs.shape[:2]} vs {T, num_cams}'

                flattened_imgs.append(imgs.view(T * num_cams, C, H, W))

            data['inputs']['img'] = flattened_imgs

        out = super().simple_process(data, training)

        if 'imgs' in out['inputs']:
            imgs = out['inputs']['imgs']
            B, TN, C, Hp, Wp = imgs.shape
            assert TN % num_cams == 0
            T = TN // num_cams
            imgs = imgs.view(B, T, num_cams, C, Hp, Wp)
            out['inputs']['imgs'] = imgs

            # Make metainfo shapes consistent with padded tensor (Hp, Wp).
            # BEVFormer uses img_shape for projection normalization.
            if 'data_samples' in out and isinstance(out['data_samples'], list):
                per_view_shape = (int(Hp), int(Wp), 3)
                for ds in out['data_samples']:
                    ds.set_metainfo({
                        'img_shape': [per_view_shape] * num_cams,
                        'pad_shape': [per_view_shape] * num_cams,
                        'batch_input_shape': (int(Hp), int(Wp)),
                    })



        return out