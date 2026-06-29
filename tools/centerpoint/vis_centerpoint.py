#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""Visualize CenterPoint LiDAR predictions as a BEV PNG.

This script is intentionally display-free: it uses numpy + cv2 and does not
need Open3D, DISPLAY, or xvfb.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


NUSCENES_CLASSES = (
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
)

WAYMO_CLASSES = (
    'Car',
    'Pedestrian',
    'Cyclist',
)

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


def parse_range(value):
    if value is None:
        return None
    parts = [float(v) for v in value.split(',')]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('range must be "min,max"')
    if parts[0] >= parts[1]:
        raise argparse.ArgumentTypeError('range min must be smaller than max')
    return parts


def parse_args():
    parser = argparse.ArgumentParser(
        description='Draw point cloud and CenterPoint json predictions in BEV.')
    parser.add_argument('--pcd', required=True, help='Raw float32 point file.')
    parser.add_argument('--pred', required=True, help='Prediction json file.')
    parser.add_argument('--out', required=True, help='Output png path.')
    parser.add_argument(
        '--load-dim',
        type=int,
        default=0,
        help='Number of float dimensions in the point file. 0 auto-detects '
        'common 5D/6D formats.')
    parser.add_argument(
        '--classes',
        choices=('nuscenes', 'waymo'),
        default='nuscenes',
        help='Class name mapping used by prediction labels.')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='Minimum prediction score to draw.')
    parser.add_argument(
        '--xlim',
        type=parse_range,
        default=None,
        help='Optional BEV x range as "min,max", in meters.')
    parser.add_argument(
        '--ylim',
        type=parse_range,
        default=None,
        help='Optional BEV y range as "min,max", in meters.')
    parser.add_argument(
        '--zlim',
        type=parse_range,
        default=None,
        help='Optional front/side z range as "min,max", in meters.')
    parser.add_argument(
        '--view',
        choices=('bev', 'front', 'side', 'all'),
        default='all',
        help='View to render. "all" writes three images with suffixes.')
    parser.add_argument(
        '--max-points',
        type=int,
        default=250000,
        help='Maximum points to draw. 0 means draw all points.')
    parser.add_argument(
        '--point-size',
        type=float,
        default=1.5,
        help='Point radius in pixels.')
    parser.add_argument(
        '--width',
        type=int,
        default=1800,
        help='Output image width in pixels.')
    parser.add_argument(
        '--height',
        type=int,
        default=1800,
        help='Output image height in pixels.')
    parser.add_argument(
        '--title',
        default='CenterPoint BEV',
        help='Figure title.')
    return parser.parse_args()


def load_points(path, load_dim):
    points = np.fromfile(path, dtype=np.float32)
    if load_dim == 0:
        candidates = [dim for dim in (5, 6, 4) if points.size % dim == 0]
        if len(candidates) != 1:
            raise ValueError(
                f'{path} has {points.size} floats; cannot auto-detect '
                f'load_dim from candidates 5/6/4. Please pass --load-dim.')
        load_dim = candidates[0]
        print(f'auto-detected load_dim={load_dim}')
    if points.size % load_dim != 0:
        raise ValueError(
            f'{path} has {points.size} floats, not divisible by load_dim '
            f'{load_dim}')
    return points.reshape(-1, load_dim)


def load_predictions(path):
    with open(path, 'r') as f:
        pred = json.load(f)
    required_keys = {'bboxes_3d', 'scores_3d', 'labels_3d'}
    missing = required_keys - set(pred)
    if missing:
        raise KeyError(f'prediction json misses keys: {sorted(missing)}')
    boxes = np.asarray(pred['bboxes_3d'], dtype=np.float32)
    scores = np.asarray(pred['scores_3d'], dtype=np.float32)
    labels = np.asarray(pred['labels_3d'], dtype=np.int64)
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError('bboxes_3d must have shape Nx7 or Nx9')
    if len(boxes) != len(scores) or len(boxes) != len(labels):
        raise ValueError('bboxes_3d, scores_3d, labels_3d length mismatch')
    return boxes, scores, labels


def filter_points(points, xlim, ylim):
    mask = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    if xlim is not None:
        mask &= (points[:, 0] >= xlim[0]) & (points[:, 0] <= xlim[1])
    if ylim is not None:
        mask &= (points[:, 1] >= ylim[0]) & (points[:, 1] <= ylim[1])
    return points[mask]


def maybe_sample_points(points, max_points):
    if max_points <= 0 or len(points) <= max_points:
        return points
    stride = int(np.ceil(len(points) / max_points))
    return points[::stride]


def box_corners_bev(box):
    x, y, _, dx, dy, _, yaw = box[:7]
    local = np.array(
        [[dx / 2, dy / 2], [dx / 2, -dy / 2], [-dx / 2, -dy / 2],
         [-dx / 2, dy / 2]],
        dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]],
                   dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)


def box_corners_3d(box):
    x, y, z, dx, dy, dz, yaw = box[:7]
    local = np.array(
        [[dx / 2, dy / 2, -dz / 2], [dx / 2, -dy / 2, -dz / 2],
         [-dx / 2, -dy / 2, -dz / 2], [-dx / 2, dy / 2, -dz / 2],
         [dx / 2, dy / 2, dz / 2], [dx / 2, -dy / 2, dz / 2],
         [-dx / 2, -dy / 2, dz / 2], [-dx / 2, dy / 2, dz / 2]],
        dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot = np.array([[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0],
                    [0.0, 0.0, 1.0]],
                   dtype=np.float32)
    return local @ rot.T + np.array([x, y, z], dtype=np.float32)


def infer_ranges(points, boxes, xlim, ylim):
    if xlim is not None and ylim is not None:
        return xlim, ylim

    xy = points[:, :2] if len(points) else np.zeros((0, 2), dtype=np.float32)
    if len(boxes):
        box_xy = boxes[:, :2]
        xy = np.vstack([xy, box_xy])
    if len(xy) == 0:
        xy = np.array([[-60.0, -60.0], [60.0, 60.0]], dtype=np.float32)

    out_x = xlim
    out_y = ylim
    if out_x is None:
        min_x, max_x = np.percentile(xy[:, 0], [1, 99])
        margin_x = max(5.0, 0.08 * float(max_x - min_x))
        out_x = [float(min_x - margin_x), float(max_x + margin_x)]
    if out_y is None:
        min_y, max_y = np.percentile(xy[:, 1], [1, 99])
        margin_y = max(5.0, 0.08 * float(max_y - min_y))
        out_y = [float(min_y - margin_y), float(max_y + margin_y)]
    return out_x, out_y


def infer_range_1d(values, explicit_range, default_range, percentile=(1, 99)):
    if explicit_range is not None:
        return explicit_range
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return list(default_range)
    min_v, max_v = np.percentile(values, percentile)
    margin = max(2.0, 0.08 * float(max_v - min_v))
    return [float(min_v - margin), float(max_v + margin)]


def make_projector(horizontal_range,
                   vertical_range,
                   width,
                   height,
                   pad,
                   flip_horizontal=False):
    span_h = horizontal_range[1] - horizontal_range[0]
    span_v = vertical_range[1] - vertical_range[0]
    if span_h <= 0 or span_v <= 0:
        raise ValueError('invalid horizontal/vertical range')
    scale = min((width - 2 * pad) / span_h, (height - 2 * pad) / span_v)
    center_h = 0.5 * (horizontal_range[0] + horizontal_range[1])
    center_v = 0.5 * (vertical_range[0] + vertical_range[1])

    def project(xy):
        xy = np.asarray(xy, dtype=np.float32)
        horizontal = xy[..., 0]
        vertical = xy[..., 1]
        direction = -1.0 if flip_horizontal else 1.0
        u = width * 0.5 + direction * (horizontal - center_h) * scale
        v = height * 0.5 - (vertical - center_v) * scale
        return np.stack([u, v], axis=-1)

    return project, scale


def draw_points(image, points, project, point_radius, axes):
    if len(points) == 0:
        return
    pixels = np.rint(project(points[:, axes])).astype(np.int32)
    h, w = image.shape[:2]
    valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < w) &
             (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
    pixels = pixels[valid]
    if len(pixels) == 0:
        return

    if point_radius <= 1:
        image[pixels[:, 1], pixels[:, 0]] = (255, 255, 255)
    else:
        radius = max(1, int(round(point_radius)))
        for pixel in pixels:
            cv2.circle(image, tuple(pixel), radius, (255, 255, 255), -1,
                       cv2.LINE_AA)


def draw_grid(image, project, horizontal_range, vertical_range, horizontal_name,
              vertical_name):
    grid_color = (32, 32, 32)
    text_color = (135, 135, 135)
    h_step = 10 if horizontal_name != 'z' else 2
    v_step = 10 if vertical_name != 'z' else 2
    h_start = int(np.ceil(horizontal_range[0] / h_step) * h_step)
    h_end = int(np.floor(horizontal_range[1] / h_step) * h_step)
    v_start = int(np.ceil(vertical_range[0] / v_step) * v_step)
    v_end = int(np.floor(vertical_range[1] / v_step) * v_step)

    for h in range(h_start, h_end + 1, h_step):
        p0 = project([[h, vertical_range[0]]])[0].astype(int)
        p1 = project([[h, vertical_range[1]]])[0].astype(int)
        cv2.line(image, tuple(p0), tuple(p1), grid_color, 1, cv2.LINE_AA)
        cv2.putText(image, f'{horizontal_name}={h}', tuple(p0 + [4, -4]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1,
                    cv2.LINE_AA)

    for v in range(v_start, v_end + 1, v_step):
        p0 = project([[horizontal_range[0], v]])[0].astype(int)
        p1 = project([[horizontal_range[1], v]])[0].astype(int)
        cv2.line(image, tuple(p0), tuple(p1), grid_color, 1, cv2.LINE_AA)
        cv2.putText(image, f'{vertical_name}={v}', tuple(p0 + [4, -4]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1,
                    cv2.LINE_AA)

    origin = project([[0.0, 0.0]])[0].astype(int)
    cv2.circle(image, tuple(origin), 6, (255, 255, 255), -1, cv2.LINE_AA)


def class_names_for_args(args):
    if args.classes == 'waymo':
        return WAYMO_CLASSES
    return NUSCENES_CLASSES


def label_to_name(label, class_names):
    label = int(label)
    if 0 <= label < len(class_names):
        return class_names[label]
    return str(label)


def draw_box_cv2(image, box, label, score, project, class_names):
    color = CLASS_COLORS[int(label) % len(CLASS_COLORS)]
    corners_xy = box_corners_bev(box)
    corners = np.rint(project(corners_xy[:, [1, 0]])).astype(np.int32)
    cv2.polylines(image, [corners], isClosed=True, color=color, thickness=3)

    x, y, _, dx, _, _, yaw = box[:7]
    heading = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
    start = np.array([x, y], dtype=np.float32)
    end = start + heading * (dx * 0.55)
    heading_pixels = np.rint(project([[start[1], start[0]],
                                      [end[1], end[0]]])).astype(np.int32)
    cv2.arrowedLine(image,
                    tuple(heading_pixels[0]),
                    tuple(heading_pixels[1]),
                    color,
                    2,
                    cv2.LINE_AA,
                    tipLength=0.25)

    class_name = label_to_name(label, class_names)
    text = f'{class_name} {score:.2f}'
    text_org = tuple(np.rint(project([[y, x]])[0]).astype(np.int32))
    cv2.putText(image, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                cv2.LINE_AA)


def draw_projected_box_cv2(image, box, label, score, project, axes,
                           class_names):
    color = CLASS_COLORS[int(label) % len(CLASS_COLORS)]
    corners = box_corners_3d(box)[:, axes]
    min_h, min_v = corners.min(axis=0)
    max_h, max_v = corners.max(axis=0)
    rect = np.array([[min_h, min_v], [max_h, min_v], [max_h, max_v],
                     [min_h, max_v]],
                    dtype=np.float32)
    pixels = np.rint(project(rect)).astype(np.int32)
    cv2.polylines(image, [pixels], isClosed=True, color=color, thickness=3)

    center = project([[box[axes[0]], box[axes[1]]]])[0].astype(np.int32)
    class_name = label_to_name(label, class_names)
    cv2.putText(image, f'{class_name} {score:.2f}', tuple(center),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def view_output_path(out_path, view, multiple_views):
    out_path = Path(out_path)
    if not multiple_views:
        return out_path
    return out_path.with_name(f'{out_path.stem}_{view}{out_path.suffix}')


def render_view(args, points, boxes, scores, labels, view, out_path):
    xlim, ylim = infer_ranges(points, boxes, args.xlim, args.ylim)
    z_values = points[:, 2] if points.shape[1] >= 3 else np.array([])
    if len(boxes):
        corners_z = np.concatenate([box_corners_3d(box)[:, 2] for box in boxes])
        z_values = np.concatenate([z_values, corners_z])
    zlim = infer_range_1d(z_values, args.zlim, (-5.0, 5.0), percentile=(0, 99))

    if view == 'bev':
        axes = (1, 0)
        horizontal_range = ylim
        vertical_range = xlim
        horizontal_name = 'y'
        vertical_name = 'x'
        flip_horizontal = True
        box_drawer = draw_box_cv2
        subtitle = 'BEV: x forward, y left'
    elif view == 'front':
        axes = (1, 2)
        horizontal_range = ylim
        vertical_range = zlim
        horizontal_name = 'y'
        vertical_name = 'z'
        flip_horizontal = True
        box_drawer = draw_projected_box_cv2
        subtitle = 'Front: y left, z up'
    elif view == 'side':
        axes = (0, 2)
        horizontal_range = xlim
        vertical_range = zlim
        horizontal_name = 'x'
        vertical_name = 'z'
        flip_horizontal = False
        box_drawer = draw_projected_box_cv2
        subtitle = 'Side: x forward, z up'
    else:
        raise ValueError(f'unsupported view: {view}')

    class_names = class_names_for_args(args)
    image = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    project, _ = make_projector(horizontal_range,
                                vertical_range,
                                args.width,
                                args.height,
                                pad=80,
                                flip_horizontal=flip_horizontal)

    draw_grid(image, project, horizontal_range, vertical_range,
              horizontal_name, vertical_name)
    draw_points(image, points, project, args.point_size, axes)

    for box, score, label in zip(boxes, scores, labels):
        if view == 'bev':
            box_drawer(image, box, label, score, project, class_names)
        else:
            box_drawer(image, box, label, score, project, axes, class_names)

    title = f'{args.title} {view}: {len(boxes)} boxes >= {args.score_thr:.2f}'
    cv2.putText(image, title, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(image, subtitle, (24, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (185, 185, 185), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), image):
        raise RuntimeError(f'failed to write {out_path}')
    print(f'saved {view} visualization to {out_path}')


def main():
    args = parse_args()
    points = load_points(args.pcd, args.load_dim)
    points = filter_points(points, args.xlim, args.ylim)
    points = maybe_sample_points(points, args.max_points)

    boxes, scores, labels = load_predictions(args.pred)
    keep = scores >= args.score_thr
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    views = ('bev', 'front', 'side') if args.view == 'all' else (args.view, )
    multiple_views = len(views) > 1
    for view in views:
        out_path = view_output_path(args.out, view, multiple_views)
        render_view(args, points, boxes, scores, labels, view, out_path)


if __name__ == '__main__':
    main()
