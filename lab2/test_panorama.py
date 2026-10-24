"""Generate the four Q4.4 scenes from consecutive original frames.

The provided planar model supports a limited field of view. Use five consecutive
frames as permitted by the assignment, rather than all 360 deg. Library and Xue
use windows with less severe stretching; the other scenes use the middle frames.
"""

import argparse
import json
from pathlib import Path
import re
import time

import cv2

import homework2 as hw

RECOMMENDED_STARTS = {'library': 8, 'Xue-Mountain-Entrance': 2}


def main():
    parser = argparse.ArgumentParser(description='Generate Q4.4 panoramas')
    parser.add_argument('--scene', choices=list(hw.PANORAMA_SCENES.values()))
    parser.add_argument('--frames', type=int, default=5,
                        help='Number of consecutive frames (default: 5)')
    parser.add_argument('--start', type=int,
                        help='Zero-based first frame; omitted uses a recommended window')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    outdir = root / 'outputs'
    outdir.mkdir(exist_ok=True)
    results = []

    for scene_id, scene in hw.PANORAMA_SCENES.items():
        if args.scene and scene != args.scene:
            continue
        folder = root / 'Problem2Images' / 'panoramas' / scene
        files = sorted(
            [p for p in folder.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')],
            key=lambda p: [int(t) if t.isdigit() else t.lower()
                           for t in re.split(r'(\d+)', p.name)]
        )
        default_start = max(0, len(files) // 2 - args.frames // 2)
        if args.frames == 5:
            default_start = RECOMMENDED_STARTS.get(scene, default_start)
        start = args.start if args.start is not None else default_start
        selected = files[start:start + args.frames]
        if args.frames < 2 or len(selected) != args.frames:
            raise ValueError('Choose at least two frames within the available sequence')
        images = [cv2.imread(str(p)) for p in selected]
        if any(im is None for im in images):
            raise FileNotFoundError(f'Cannot read frames in {folder}')

        print(f'{scene}: {[p.name for p in selected]}', flush=True)
        begin = time.perf_counter()
        pano = hw.generate_panorama(images)
        output = outdir / f'panorama_{scene_id}.png'
        if not cv2.imwrite(str(output), pano):
            raise OSError(f'Cannot save panorama: {output}')
        result = {
            'scene': scene, 'scene_id': scene_id,
            'source_frames': [p.name for p in selected],
            'anchor_frame': selected[len(selected) // 2].name,
            'canvas_width': int(pano.shape[1]), 'canvas_height': int(pano.shape[0]),
            'elapsed_seconds': round(time.perf_counter() - begin, 2),
            'projection': 'planar, middle anchor, original frames sampled once',
        }
        (outdir / f'panorama_{scene_id}.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        print(f'Saved: {output}', flush=True)

    return results


if __name__ == '__main__':
    main()
