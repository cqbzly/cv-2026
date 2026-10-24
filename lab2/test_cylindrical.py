"""Run the cylindrical bonus on local images; no changes to base HW functions."""

import argparse
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np

import homework2 as hw
from cylindrical import cylindrical_warp, generate_cylindrical_panorama
from test_panorama import RECOMMENDED_STARTS


def main():
    parser = argparse.ArgumentParser(description='Cylindrical panorama bonus')
    parser.add_argument('--scene', choices=list(hw.PANORAMA_SCENES.values()))
    parser.add_argument('--frames', type=int, default=5,
                        help='Consecutive frame count; 0 means the complete sequence')
    parser.add_argument('--start', type=int, help='Zero-based first frame')
    focal = parser.add_mutually_exclusive_group()
    focal.add_argument('--focal', type=float, help='Shared focal length in pixels')
    focal.add_argument('--focal-scale', type=float, default=1.0,
                       help='f / image_width, default 1.0 is an INITIAL GUESS')
    focal.add_argument('--auto-focal', action='store_true',
                       help='Estimate f from a complete turn; requires --close-loop')
    parser.add_argument('--blend', choices=('average', 'feather'), default='feather')
    parser.add_argument('--threshold', type=float, default=3.0)
    parser.add_argument('--close-loop', action='store_true',
                        help='Full 360 only: align first/last and wrap the canvas')
    args = parser.parse_args()
    if args.frames < 0 or args.frames == 1:
        parser.error('--frames must be 0 (all) or at least 2')
    if args.close_loop and args.frames != 0:
        parser.error('--close-loop requires --frames 0 (a complete turn)')
    if args.close_loop and args.start not in (None, 0):
        parser.error('--close-loop requires the complete sequence from --start 0')
    if args.auto_focal and not args.close_loop:
        parser.error('--auto-focal requires --close-loop and a complete turn')

    root = Path(__file__).resolve().parent
    outdir = root / 'outputs' / 'cylindrical'
    outdir.mkdir(parents=True, exist_ok=True)
    for scene_id, scene in hw.PANORAMA_SCENES.items():
        if args.scene and args.scene != scene:
            continue
        folder = root / 'Problem2Images' / 'panoramas' / scene
        files = sorted([p for p in folder.iterdir()
                        if p.suffix.lower() in ('.jpg', '.jpeg', '.png')],
                       key=lambda p: [int(t) if t.isdigit() else t.lower()
                                      for t in re.split(r'(\d+)', p.name)])
        start = max(0, len(files) // 2 - args.frames // 2) if args.frames else 0
        if args.frames == 5:
            start = RECOMMENDED_STARTS.get(scene, start)
        if args.start is not None:
            start = args.start
        selected = files[start:] if args.frames == 0 else files[start:start + args.frames]
        if start < 0 or len(selected) < 2 or (args.frames and len(selected) != args.frames):
            raise ValueError('Choose at least two frames within the sequence')
        images = [cv2.imread(str(path)) for path in selected]
        if any(image is None for image in images):
            raise FileNotFoundError(f'Cannot load images in {folder}')
        f = args.focal if args.focal is not None else args.focal_scale * images[0].shape[1]
        if args.auto_focal:
            f = None
        description = 'estimate from full turn' if f is None else f'{f:.2f}px'
        print(f'{scene}: {len(images)} frames, f={description}', flush=True)
        begin = time.perf_counter()
        pano, info = generate_cylindrical_panorama(
            images, f, threshold=args.threshold, blend=args.blend,
            close_loop=args.close_loop, return_info=True)
        info.update(scene=scene, source_frames=[p.name for p in selected],
                    elapsed_seconds=round(time.perf_counter() - begin, 2))
        f = info['focal_length_pixels']
        suffix = '360' if args.close_loop else f'{len(images)}frames'
        prefix = outdir / f'cylindrical_{scene_id}_{suffix}'
        if not cv2.imwrite(str(prefix.with_suffix('.png')), pano):
            raise OSError(f'Cannot save {prefix}')
        prefix.with_suffix('.json').write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')

        # Inspect one frame and its coverage separately. White means covered;
        # black means absent, regardless of the actual image's pixel colors.
        projected, mask, origin = cylindrical_warp(images[0], f)
        cv2.imwrite(str(outdir / f'projection_{scene_id}.png'),
                    np.rint(np.clip(projected, 0, 255)).astype(np.uint8))
        cv2.imwrite(str(outdir / f'projection_{scene_id}_mask.png'),
                    np.rint(mask * 255).astype(np.uint8))
        print(f'Saved: {prefix}.png; f={f:.2f}px; '
              f'inliers {[p["inliers"] for p in info["pairs"]]}', flush=True)


if __name__ == '__main__':
    main()
