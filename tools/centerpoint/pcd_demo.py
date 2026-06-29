# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import os.path as osp
import json
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from mmengine.config import Config
from mmengine.fileio import dump
from mmengine.logging import print_log

from mmdet3d.apis import LidarDet3DInferencer
from mmdet3d.apis.inferencers.base_3d_inferencer import Base3DInferencer
from mmdet3d.structures import Box3DMode

from convert_plus_pcd_to_bin import (make_fifth_dim,
                                     parse_binary_compressed_payload,
                                     read_pcd_header)


CLASS_COLORS = (
    (28, 26, 228),
    (184, 126, 55),
    (74, 175, 77),
    (163, 78, 152),
    (0, 127, 255),
    (40, 86, 166),
    (191, 129, 247),
    (153, 153, 153),
    (207, 190, 23),
    (34, 189, 188),
)


def _box_type_name(data_sample):
    box_mode = getattr(data_sample, 'box_mode_3d', None)
    if box_mode == Box3DMode.LIDAR:
        return 'LiDAR'
    if box_mode == Box3DMode.CAM:
        return 'Camera'
    if box_mode == Box3DMode.DEPTH:
        return 'Depth'

    box_type = getattr(data_sample, 'box_type_3d', None)
    box_type_name = getattr(box_type, '__name__', str(box_type))
    if 'LiDAR' in box_type_name:
        return 'LiDAR'
    if 'Camera' in box_type_name or 'Cam' in box_type_name:
        return 'Camera'
    if 'Depth' in box_type_name:
        return 'Depth'
    return None


def _pred2dict_compatible(self, data_sample, pred_out_dir=''):
    result = {}
    if 'pred_instances_3d' in data_sample:
        pred_instances_3d = data_sample.pred_instances_3d.numpy()
        result = {
            'labels_3d': pred_instances_3d.labels_3d.tolist(),
            'scores_3d': pred_instances_3d.scores_3d.tolist(),
            'bboxes_3d': pred_instances_3d.bboxes_3d.tensor.cpu().tolist()
        }

    if 'pred_pts_seg' in data_sample:
        pred_pts_seg = data_sample.pred_pts_seg.numpy()
        result['pts_semantic_mask'] = \
            pred_pts_seg.pts_semantic_mask.tolist()

    box_type_name = _box_type_name(data_sample)
    if box_type_name is not None:
        result['box_type_3d'] = box_type_name

    if pred_out_dir != '':
        if 'lidar_path' in data_sample:
            lidar_path = osp.basename(data_sample.lidar_path)
            lidar_path = osp.splitext(lidar_path)[0]
            out_json_path = osp.join(pred_out_dir, 'preds',
                                     lidar_path + '.json')
        elif 'img_path' in data_sample:
            img_path = osp.basename(data_sample.img_path)
            img_path = osp.splitext(img_path)[0]
            out_json_path = osp.join(pred_out_dir, 'preds',
                                     img_path + '.json')
        else:
            output_idx = getattr(self, 'num_visualized_imgs',
                                 getattr(self, 'num_visualized_frames', 0))
            out_json_path = osp.join(
                pred_out_dir, 'preds',
                f'{str(output_idx).zfill(8)}.json')
        dump(result, out_json_path)

    return result


Base3DInferencer.pred2dict = _pred2dict_compatible


def _walk_cfg(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_cfg(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_cfg(child)


def _parse_point_cloud_range(value):
    if value is None:
        return None
    parts = [float(v) for v in value.split(',')]
    if len(parts) != 6:
        raise ValueError(
            '--point-cloud-range must be "min_x,min_y,min_z,max_x,max_y,max_z"')
    if parts[0] >= parts[3] or parts[1] >= parts[4] or parts[2] >= parts[5]:
        raise ValueError('--point-cloud-range min values must be less than max values')
    return parts


def _override_point_cloud_range(node, point_cloud_range):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == 'point_cloud_range':
                node[key] = list(point_cloud_range)
            elif key in ('post_center_range', 'post_center_limit_range'):
                node[key] = list(point_cloud_range)
            elif key == 'pc_range' and isinstance(value, (list, tuple)):
                if len(value) == 2:
                    node[key] = list(point_cloud_range[:2])
                elif len(value) == 3:
                    node[key] = list(point_cloud_range[:3])
                elif len(value) == 6:
                    node[key] = list(point_cloud_range)
                else:
                    _override_point_cloud_range(value, point_cloud_range)
            else:
                _override_point_cloud_range(value, point_cloud_range)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _override_point_cloud_range(value, point_cloud_range)


def _first_key_value(node, key):
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = _first_key_value(value, key)
            if found is not None:
                return found
    elif isinstance(node, (list, tuple)):
        for value in node:
            found = _first_key_value(value, key)
            if found is not None:
                return found
    return None


def _first_voxel_size_3d(node):
    if isinstance(node, dict):
        if 'voxel_size' in node:
            voxel_size = node['voxel_size']
            if isinstance(voxel_size, (list, tuple)) and len(voxel_size) >= 3:
                return voxel_size
        for value in node.values():
            found = _first_voxel_size_3d(value)
            if found is not None:
                return found
    elif isinstance(node, (list, tuple)):
        for value in node:
            found = _first_voxel_size_3d(value)
            if found is not None:
                return found
    return None


def _override_sparse_shape(cfg, point_cloud_range):
    model = cfg._cfg_dict.get('model', {})
    voxel_size = _first_voxel_size_3d(model)
    if voxel_size is None:
        return
    voxel_size = [float(v) for v in voxel_size]
    if len(voxel_size) < 3:
        return

    x_bins = int(round((point_cloud_range[3] - point_cloud_range[0]) /
                       voxel_size[0]))
    y_bins = int(round((point_cloud_range[4] - point_cloud_range[1]) /
                       voxel_size[1]))
    z_bins = int(round((point_cloud_range[5] - point_cloud_range[2]) /
                       voxel_size[2]))
    sparse_shape = [z_bins + 1, y_bins, x_bins]

    middle_encoder = _first_key_value(model, 'pts_middle_encoder')
    if isinstance(middle_encoder, dict) and 'sparse_shape' in middle_encoder:
        middle_encoder['sparse_shape'] = sparse_shape
        print_log(
            f'override pts_middle_encoder.sparse_shape to {sparse_shape}',
            logger='current')


def _prepare_model_config(model, point_cloud_range):
    if point_cloud_range is None:
        return model, None

    cfg = Config.fromfile(model)
    cfg.point_cloud_range = list(point_cloud_range)
    _override_point_cloud_range(cfg._cfg_dict, point_cloud_range)
    _override_sparse_shape(cfg, point_cloud_range)

    tmp_dir = tempfile.TemporaryDirectory(prefix='centerpoint_cfg_')
    out_path = Path(tmp_dir.name) / Path(model).name
    cfg.dump(str(out_path))
    print_log(
        f'override point_cloud_range to {point_cloud_range}; temporary config: '
        f'{out_path}',
        logger='current')
    return str(out_path), tmp_dir


def _find_load_points_cfg(model):
    cfg = Config.fromfile(model)
    for node in _walk_cfg(cfg._cfg_dict):
        transform_type = node.get('type')
        if isinstance(transform_type, str) and \
                transform_type.endswith('LoadPointsFromFile'):
            return node
    return None


def _infer_load_dim(model):
    load_points_cfg = _find_load_points_cfg(model)
    if load_points_cfg is None:
        print_log(
            'LoadPointsFromFile is not found in config. Use load_dim=5 for '
            'converted Plus PCD.',
            logger='current',
            level=logging.WARNING)
        return 5
    return int(load_points_cfg.get('load_dim', 5))


def _make_pseudo_ring(arrays, num_rings):
    x = arrays['x'].astype(np.float32)
    y = arrays['y'].astype(np.float32)
    z = arrays['z'].astype(np.float32)
    horizontal_distance = np.sqrt(x * x + y * y)
    vertical_angle = np.arctan2(z, horizontal_distance)
    finite = np.isfinite(vertical_angle)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)

    min_angle, max_angle = np.percentile(vertical_angle[finite], [1, 99])
    if max_angle <= min_angle:
        return np.zeros_like(x, dtype=np.float32)
    normalized = (vertical_angle - min_angle) / (max_angle - min_angle)
    pseudo_ring = np.clip(np.rint(normalized * (num_rings - 1)), 0,
                          num_rings - 1)
    return pseudo_ring.astype(np.float32)


def _make_fifth_dim(arrays, mode, pseudo_ring_count):
    if mode == 'pseudo-ring':
        return _make_pseudo_ring(arrays, pseudo_ring_count)
    return make_fifth_dim(arrays, mode)


def _convert_plus_pcd_for_model(pcd_path, out_path, load_dim, clip_intensity,
                                fifth_dim, pseudo_ring_count):
    with open(pcd_path, 'rb') as f:
        header = read_pcd_header(f)
        raw = f.read()

    required = {'x', 'y', 'z', 'intensity'}
    if load_dim >= 5 and fifth_dim == 'ring':
        required.add('ring')
    missing = required - set(header['FIELDS'])
    if missing:
        raise KeyError(f'PCD missing fields: {sorted(missing)}')

    arrays, points = parse_binary_compressed_payload(raw, header)
    intensity = arrays['intensity'].astype(np.float32)
    if clip_intensity:
        intensity = np.clip(intensity, 0.0, 1.0)

    columns = [
        arrays['x'].astype(np.float32),
        arrays['y'].astype(np.float32),
        arrays['z'].astype(np.float32),
        intensity,
    ]
    if load_dim >= 5:
        columns.append(_make_fifth_dim(arrays, fifth_dim, pseudo_ring_count))
    while len(columns) < load_dim:
        columns.append(np.zeros_like(columns[0], dtype=np.float32))

    out = np.stack(columns[:load_dim], axis=1)
    finite = np.isfinite(out[:, :min(load_dim, 4)]).all(axis=1)
    out = out[finite].astype(np.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.tofile(out_path)
    print_log(
        f'converted Plus PCD {pcd_path} -> {out_path} '
        f'({points} -> {len(out)} points, load_dim={load_dim})',
        logger='current')


def _prepare_points_input(pcd_path, model, clip_intensity, fifth_dim,
                          pseudo_ring_count):
    path = Path(pcd_path)
    load_dim = _infer_load_dim(model)
    if path.suffix.lower() != '.pcd':
        return str(path), None, load_dim

    tmp_dir = tempfile.TemporaryDirectory(prefix='centerpoint_plus_pcd_')
    out_path = Path(tmp_dir.name) / f'{path.stem}.bin'
    _convert_plus_pcd_for_model(path, out_path, load_dim, clip_intensity,
                                fifth_dim, pseudo_ring_count)
    return str(out_path), tmp_dir, load_dim


def _box_corners_bev(box):
    x, y, _, dx, dy, _, yaw = box[:7]
    local = np.array([[dx / 2, dy / 2], [dx / 2, -dy / 2],
                      [-dx / 2, -dy / 2], [-dx / 2, dy / 2]],
                     dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]],
                   dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)


def _infer_bev_ranges(points, boxes):
    xy = points[:, :2] if len(points) else np.zeros((0, 2), dtype=np.float32)
    if len(boxes):
        xy = np.vstack([xy, boxes[:, :2]])
    if len(xy) == 0:
        return [-60.0, 60.0], [-60.0, 60.0]

    x_min, x_max = np.percentile(xy[:, 0], [1, 99])
    y_min, y_max = np.percentile(xy[:, 1], [1, 99])
    x_margin = max(5.0, 0.08 * float(x_max - x_min))
    y_margin = max(5.0, 0.08 * float(y_max - y_min))
    return [float(x_min - x_margin), float(x_max + x_margin)], [
        float(y_min - y_margin),
        float(y_max + y_margin),
    ]


def _make_bev_projector(xlim, ylim, width, height, pad=80):
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    if x_span <= 0 or y_span <= 0:
        raise ValueError('invalid BEV range')
    scale = min((width - 2 * pad) / y_span, (height - 2 * pad) / x_span)
    x_center = 0.5 * (xlim[0] + xlim[1])
    y_center = 0.5 * (ylim[0] + ylim[1])

    def project(xy):
        xy = np.asarray(xy, dtype=np.float32)
        x = xy[..., 0]
        y = xy[..., 1]
        u = width * 0.5 - (y - y_center) * scale
        v = height * 0.5 - (x - x_center) * scale
        return np.stack([u, v], axis=-1)

    return project


def _draw_bev_grid(image, project, xlim, ylim):
    grid_color = (32, 32, 32)
    text_color = (135, 135, 135)
    for x in range(int(np.ceil(xlim[0] / 10) * 10),
                   int(np.floor(xlim[1] / 10) * 10) + 1, 10):
        p0 = project([[x, ylim[0]]])[0].astype(int)
        p1 = project([[x, ylim[1]]])[0].astype(int)
        cv2.line(image, tuple(p0), tuple(p1), grid_color, 1, cv2.LINE_AA)
        cv2.putText(image, f'x={x}', tuple(p0 + [4, -4]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1,
                    cv2.LINE_AA)

    for y in range(int(np.ceil(ylim[0] / 10) * 10),
                   int(np.floor(ylim[1] / 10) * 10) + 1, 10):
        p0 = project([[xlim[0], y]])[0].astype(int)
        p1 = project([[xlim[1], y]])[0].astype(int)
        cv2.line(image, tuple(p0), tuple(p1), grid_color, 1, cv2.LINE_AA)
        cv2.putText(image, f'y={y}', tuple(p0 + [4, -4]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1,
                    cv2.LINE_AA)

    origin = project([[0.0, 0.0]])[0].astype(int)
    cv2.circle(image, tuple(origin), 6, (255, 255, 255), -1, cv2.LINE_AA)


def _intensity_colors(points):
    if points.shape[1] < 4:
        values = np.ones((len(points), ), dtype=np.float32)
    else:
        values = points[:, 3].astype(np.float32)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.full((len(points), 3), 180, dtype=np.uint8)

    lo, hi = np.percentile(values, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    intensity = points[:, 3] if points.shape[1] >= 4 else np.ones(len(points))
    norm = np.clip((intensity - lo) / (hi - lo), 0.0, 1.0)
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    gray = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray.reshape(-1, 1), cv2.COLORMAP_TURBO)[:, 0, :]


def _draw_intensity_points(image, points, project, max_points):
    finite = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    points = points[finite]
    if max_points > 0 and len(points) > max_points:
        stride = int(np.ceil(len(points) / max_points))
        points = points[::stride]
    if len(points) == 0:
        return

    pixels = np.rint(project(points[:, :2])).astype(np.int32)
    h, w = image.shape[:2]
    valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < w) &
             (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
    pixels = pixels[valid]
    colors = _intensity_colors(points)[valid]
    image[pixels[:, 1], pixels[:, 0]] = colors


def _draw_bev_box(image, box, label, score, project):
    color = CLASS_COLORS[int(label) % len(CLASS_COLORS)]
    corners = np.rint(project(_box_corners_bev(box))).astype(np.int32)
    cv2.polylines(image, [corners], isClosed=True, color=color, thickness=3)

    x, y, _, dx, _, _, yaw = box[:7]
    start = np.array([x, y], dtype=np.float32)
    heading = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
    end = start + heading * (dx * 0.55)
    heading_pixels = np.rint(project([start, end])).astype(np.int32)
    cv2.arrowedLine(image,
                    tuple(heading_pixels[0]),
                    tuple(heading_pixels[1]),
                    color,
                    2,
                    cv2.LINE_AA,
                    tipLength=0.25)
    text_org = tuple(np.rint(project([[x, y]])[0]).astype(np.int32))
    cv2.putText(image, f'{int(label)} {score:.2f}', text_org,
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _load_prediction(pred_path):
    with open(pred_path, 'r') as f:
        pred = json.load(f)
    boxes = np.asarray(pred.get('bboxes_3d', []), dtype=np.float32)
    scores = np.asarray(pred.get('scores_3d', []), dtype=np.float32)
    labels = np.asarray(pred.get('labels_3d', []), dtype=np.int64)
    if boxes.size == 0:
        boxes = np.zeros((0, 7), dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError(f'{pred_path} bboxes_3d must have shape Nx7 or Nx9')
    if len(boxes) != len(scores) or len(boxes) != len(labels):
        raise ValueError(f'{pred_path} prediction length mismatch')
    return boxes, scores, labels


def _find_prediction_path(out_dir, stem):
    preds_dir = Path(out_dir) / 'preds'
    expected = preds_dir / f'{stem}.json'
    if expected.exists():
        return expected
    candidates = sorted(preds_dir.glob('*.json'),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f'no prediction json found under {preds_dir}')


def _render_bev_visualization(vis_info, out_dir, pred_score_thr):
    pred_path = _find_prediction_path(out_dir, vis_info['stem'])
    raw = np.fromfile(vis_info['points_path'], dtype=np.float32)
    load_dim = vis_info['load_dim']
    if raw.size % load_dim != 0:
        raise ValueError(
            f'{vis_info["points_path"]} has {raw.size} floats, not '
            f'divisible by load_dim={load_dim}')
    points = raw.reshape(-1, load_dim)

    boxes, scores, labels = _load_prediction(pred_path)
    keep = scores >= pred_score_thr
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    width = 1800
    height = 1800
    image = np.zeros((height, width, 3), dtype=np.uint8)
    xlim, ylim = _infer_bev_ranges(points, boxes)
    project = _make_bev_projector(xlim, ylim, width, height)
    _draw_bev_grid(image, project, xlim, ylim)
    _draw_intensity_points(image, points, project, max_points=350000)
    for box, score, label in zip(boxes, scores, labels):
        _draw_bev_box(image, box, label, score, project)

    title = f'BEV intensity points + boxes >= {pred_score_thr:.2f}'
    cv2.putText(image, title, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(image, 'x forward, y left; point color = intensity',
                (24, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (185, 185, 185), 1,
                cv2.LINE_AA)

    out_path = Path(out_dir) / 'bev' / f'{vis_info["stem"]}_bev.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), image):
        raise RuntimeError(f'failed to write {out_path}')
    print_log(f'BEV visualization has been saved at {out_path}',
              logger='current')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file. Plus .pcd is converted '
                        'to the model LoadPointsFromFile load_dim.')
    parser.add_argument('model', help='Config file')
    parser.add_argument('weights', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='output',
        help='Output directory of prediction and visualization results.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show online visualization results')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=-1,
        help='The interval of show (s). Demo will be blocked in showing'
        'results, if wait_time is -1. Defaults to -1.')
    parser.add_argument(
        '--no-save-vis',
        action='store_true',
        help='Do not save detection visualization results')
    parser.add_argument(
        '--no-save-pred',
        action='store_true',
        help='Do not save detection prediction results')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--clip-intensity',
        action='store_true',
        help='Clip Plus PCD intensity to [0, 1] before inference.')
    parser.add_argument(
        '--fifth-dim',
        choices=('ring', 'feature', 'zeros', 'timestamp', 'pseudo-ring'),
        default='ring',
        help='Feature used as the 5th column when the model load_dim is at '
        'least 5.')
    parser.add_argument(
        '--pseudo-ring-count',
        type=int,
        default=32,
        help='Number of vertical-angle buckets used by '
        '`--fifth-dim pseudo-ring`.')
    parser.add_argument(
        '--point-cloud-range',
        default=None,
        help='Override config point_cloud_range as '
        '"min_x,min_y,min_z,max_x,max_y,max_z", for example '
        '"-80,-80,-5,120,80,5".')
    parser.add_argument(
        '--bev-vis',
        dest='bev_vis',
        action='store_true',
        default=True,
        help='Save a BEV image with intensity-colored points and predicted '
        'boxes. Enabled by default.')
    parser.add_argument(
        '--no-bev-vis',
        dest='bev_vis',
        action='store_false',
        help='Disable the extra BEV visualization image.')
    call_args = vars(parser.parse_args())

    pcd = call_args.pop('pcd')
    bev_vis = call_args.pop('bev_vis')

    if bev_vis and call_args['no_save_pred']:
        print_log(
            '`--no-save-pred` is ignored because BEV visualization needs the '
            'prediction json to draw boxes. Use `--no-bev-vis` to disable it.',
            logger='current',
            level=logging.WARNING)
        call_args['no_save_pred'] = False

    if not bev_vis and call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    init_kws = ['model', 'weights', 'device']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    point_cloud_range = _parse_point_cloud_range(
        call_args.pop('point_cloud_range'))
    model_path, model_tmp_dir = _prepare_model_config(init_args['model'],
                                                      point_cloud_range)
    init_args['model'] = model_path

    pseudo_ring_count = call_args.pop('pseudo_ring_count')
    if pseudo_ring_count <= 1:
        raise ValueError('--pseudo-ring-count must be greater than 1')

    points_path, tmp_dir, load_dim = _prepare_points_input(
        pcd, init_args['model'], call_args.pop('clip_intensity'),
        call_args.pop('fifth_dim'), pseudo_ring_count)
    call_args['inputs'] = dict(points=points_path)
    call_args['_tmp_dirs'] = [d for d in (model_tmp_dir, tmp_dir)
                              if d is not None]
    call_args['_bev_vis_info'] = None
    if bev_vis and call_args['out_dir'] != '':
        call_args['_bev_vis_info'] = {
            'points_path': points_path,
            'load_dim': load_dim,
            'stem': Path(pcd).stem,
        }

    # NOTE: If your operating environment does not have a display device,
    # (e.g. a remote server), you can save the predictions and visualize
    # them in local devices.
    if os.environ.get('DISPLAY') is None and call_args['show']:
        print_log(
            'Display device not found. `--show` is forced to False',
            logger='current',
            level=logging.WARNING)
        call_args['show'] = False

    return init_args, call_args


def main():
    init_args, call_args = parse_args()
    tmp_dirs = call_args.pop('_tmp_dirs', [])
    bev_vis_info = call_args.pop('_bev_vis_info', None)

    inferencer = LidarDet3DInferencer(**init_args)
    try:
        inferencer(**call_args)
        if bev_vis_info is not None:
            _render_bev_visualization(bev_vis_info, call_args['out_dir'],
                                      call_args['pred_score_thr'])
    finally:
        for tmp_dir in tmp_dirs:
            tmp_dir.cleanup()

    if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                           and call_args['no_save_pred']):
        print_log(
            f'results have been saved at {call_args["out_dir"]}',
            logger='current')


if __name__ == '__main__':
    main()
