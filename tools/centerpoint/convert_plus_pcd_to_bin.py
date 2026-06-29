#!/usr/bin/env python3
"""Convert Plus/PCL binary_compressed PCD to raw float32 bin for 3D detectors.

The CenterPoint nuScenes config in this repo uses LoadPointsFromFile with
load_dim=5. For Plus PCD files with fields
  x y z intensity ring timestamp feature
the default output writes:
  x y z intensity ring
as N x 5 float32.

Waymo PointPillars configs use LoadPointsFromFile with load_dim=6 and
use_dim=[0, 1, 2, 3, 4]. For that case, use --format waymo to write:
  x y z intensity fifth_dim 0
as N x 6 float32.
"""

import argparse
import struct
from pathlib import Path

import numpy as np

try:
    import lzf
except ImportError:
    lzf = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Plus binary_compressed PCD to CenterPoint .bin.')
    parser.add_argument('--pcd', required=True, help='Input .pcd path.')
    parser.add_argument('--out', required=True, help='Output .bin path.')
    parser.add_argument(
        '--clip-intensity',
        action='store_true',
        help='Clip intensity to [0, 1]. Leave disabled by default.')
    parser.add_argument(
        '--centerpoint-range',
        action='store_true',
        help='Keep only points in x/y [-51.2, 51.2] and z [-5.0, 3.0].')
    parser.add_argument(
        '--waymo-range',
        action='store_true',
        help='Keep only points in Waymo PointPillars range: x/y '
        '[-74.88, 74.88] and z [-2.0, 4.0].')
    parser.add_argument(
        '--sample-points',
        type=int,
        default=0,
        help='Randomly sample this many points after filtering. 0 disables.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed used by --sample-points.')
    parser.add_argument(
        '--fifth-dim',
        choices=('ring', 'feature', 'zeros', 'timestamp'),
        default='ring',
        help='Feature written as the 5th float32 column.')
    parser.add_argument(
        '--format',
        choices=('nuscenes', 'waymo'),
        default='nuscenes',
        help='Output point format. nuscenes writes 5 floats/point; waymo '
        'writes 6 floats/point for Waymo PointPillars configs.')
    args = parser.parse_args()
    if args.centerpoint_range and args.waymo_range:
        parser.error('--centerpoint-range and --waymo-range are mutually exclusive')
    return args


def read_pcd_header(f):
    header_lines = []
    while True:
        line = f.readline()
        if not line:
            raise ValueError('PCD header ended before DATA line')
        decoded = line.decode('ascii', errors='replace').strip()
        header_lines.append(decoded)
        if decoded.startswith('DATA '):
            break

    header = {}
    for line in header_lines:
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        key = parts[0]
        header[key] = parts[1:]
    return header


def lzf_decompress(data, expected_size):
    if lzf is None:
        raise ImportError(
            'python-lzf is required for DATA binary_compressed PCD files. '
            'Install it with: pip install python-lzf')

    decompressed = lzf.decompress(data, expected_size)
    if decompressed is None:
        raise ValueError('python-lzf failed to decompress PCD payload')
    if len(decompressed) != expected_size:
        raise ValueError(
            f'LZF decompressed {len(decompressed)} bytes, expected '
            f'{expected_size}')
    return decompressed


def numpy_dtype(size, type_char):
    if type_char == 'F':
        if size == 4:
            return np.float32
        if size == 8:
            return np.float64
    if type_char == 'U':
        return {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[size]
    if type_char == 'I':
        return {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[size]
    raise ValueError(f'unsupported PCD field type {type_char} size {size}')


def parse_binary_compressed_payload(raw, header):
    fields = header['FIELDS']
    sizes = [int(v) for v in header['SIZE']]
    types = header['TYPE']
    counts = [int(v) for v in header.get('COUNT', ['1'] * len(fields))]
    points = int(header['POINTS'][0])

    if header['DATA'][0] != 'binary_compressed':
        raise ValueError(f'unsupported DATA type: {header["DATA"][0]}')

    compressed_size, uncompressed_size = struct.unpack('<II', raw[:8])
    compressed = raw[8:8 + compressed_size]
    if len(compressed) != compressed_size:
        raise ValueError('PCD payload shorter than compressed_size')
    data = lzf_decompress(compressed, uncompressed_size)

    field_widths = [size * count for size, count in zip(sizes, counts)]
    point_step = sum(field_widths)
    expected_uncompressed = points * point_step
    if uncompressed_size != expected_uncompressed:
        raise ValueError(
            f'uncompressed size {uncompressed_size} != '
            f'POINTS * point_step {expected_uncompressed}')

    # PCL binary_compressed stores fields in structure-of-arrays order:
    # all values for field0, then all values for field1, ...
    arrays = {}
    offset = 0
    for field, size, type_char, count, width in zip(fields, sizes, types,
                                                    counts, field_widths):
        if count != 1:
            raise ValueError(f'unsupported COUNT {count} for field {field}')
        dtype = numpy_dtype(size, type_char)
        byte_count = points * width
        arrays[field] = np.frombuffer(data[offset:offset + byte_count],
                                      dtype=dtype,
                                      count=points)
        offset += byte_count
    return arrays, points


def make_fifth_dim(arrays, mode):
    if mode == 'ring':
        return arrays['ring'].astype(np.float32)
    if mode == 'feature':
        if 'feature' not in arrays:
            raise KeyError('PCD missing field "feature"')
        return arrays['feature'].astype(np.float32)
    if mode == 'zeros':
        return np.zeros_like(arrays['x'], dtype=np.float32)
    if mode == 'timestamp':
        if 'timestamp' not in arrays:
            raise KeyError('PCD missing field "timestamp"')
        timestamp = arrays['timestamp'].astype(np.float64)
        return (timestamp - timestamp.min()).astype(np.float32)
    raise ValueError(f'unsupported fifth-dim mode: {mode}')


def convert(pcd_path, out_path, clip_intensity, centerpoint_range, waymo_range,
            sample_points, seed, fifth_dim, output_format):
    with open(pcd_path, 'rb') as f:
        header = read_pcd_header(f)
        raw = f.read()

    required = {'x', 'y', 'z', 'intensity', 'ring'}
    missing = required - set(header['FIELDS'])
    if missing:
        raise KeyError(f'PCD missing fields: {sorted(missing)}')

    arrays, points = parse_binary_compressed_payload(raw, header)
    intensity = arrays['intensity'].astype(np.float32)
    if clip_intensity:
        intensity = np.clip(intensity, 0.0, 1.0)

    out = np.stack([
        arrays['x'].astype(np.float32),
        arrays['y'].astype(np.float32),
        arrays['z'].astype(np.float32),
        intensity,
        make_fifth_dim(arrays, fifth_dim),
    ],
                   axis=1)
    finite = np.isfinite(out[:, :4]).all(axis=1)
    out = out[finite]

    before_range = len(out)
    if centerpoint_range:
        in_range = ((out[:, 0] >= -51.2) & (out[:, 0] <= 51.2) &
                    (out[:, 1] >= -51.2) & (out[:, 1] <= 51.2) &
                    (out[:, 2] >= -5.0) & (out[:, 2] <= 3.0))
        out = out[in_range]
    if waymo_range:
        in_range = ((out[:, 0] >= -74.88) & (out[:, 0] <= 74.88) &
                    (out[:, 1] >= -74.88) & (out[:, 1] <= 74.88) &
                    (out[:, 2] >= -2.0) & (out[:, 2] <= 4.0))
        out = out[in_range]

    before_sample = len(out)
    if sample_points > 0 and len(out) > sample_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(out), size=sample_points, replace=False)
        indices.sort()
        out = out[indices]

    if output_format == 'waymo':
        unused_sixth_dim = np.zeros((len(out), 1), dtype=np.float32)
        out = np.concatenate([out, unused_sixth_dim], axis=1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.astype(np.float32).tofile(out_path)

    mins = out[:, :3].min(axis=0)
    maxs = out[:, :3].max(axis=0)
    print(f'converted {pcd_path} -> {out_path}')
    print(f'points: {points} -> {len(out)} finite')
    print(f'format: {output_format}, dims: {out.shape[1]}')
    print(f'fifth_dim: {fifth_dim}')
    if centerpoint_range:
        print(f'centerpoint_range: {before_range} -> {before_sample}')
    if waymo_range:
        print(f'waymo_range: {before_range} -> {before_sample}')
    if sample_points > 0:
        print(f'sample_points: {before_sample} -> {len(out)}')
    print(f'xyz min: {mins.tolist()}')
    print(f'xyz max: {maxs.tolist()}')
    print(f'5th min/max: {float(out[:, 4].min())}, {float(out[:, 4].max())}')


def main():
    args = parse_args()
    convert(args.pcd, args.out, args.clip_intensity, args.centerpoint_range,
            args.waymo_range, args.sample_points, args.seed, args.fifth_dim,
            args.format)


if __name__ == '__main__':
    main()
