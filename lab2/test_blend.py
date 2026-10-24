"""Test pair-image stitching with local paths, independent of the working directory."""

import argparse
from pathlib import Path

import cv2
import numpy as np

import homework2 as hw


def main():
    parser = argparse.ArgumentParser(description='Estimate H and save blended image pairs')
    parser.add_argument('--pair', nargs=2, metavar=('IMAGE1', 'IMAGE2'),
                        help='Image filenames inside Problem2Images')
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    outdir = root / 'outputs'
    outdir.mkdir(exist_ok=True)
    pairs = [args.pair] if args.pair else [
        ('1_1.jpg', '1_2.jpg'),
        ('2_1.jpg', '2_2.jpg'),
        ('3_1.JPG', '3_2.JPG'),
        ('4_1.jpg', '4_2.jpg'),  # Extra rotation/scale test, beyond Q4.3's three pairs.
    ]

    for name1, name2 in pairs:
        img1 = cv2.imread(str(root / 'Problem2Images' / name1))
        img2 = cv2.imread(str(root / 'Problem2Images' / name2))
        if img1 is None or img2 is None:
            raise FileNotFoundError(f'Cannot load image pair: {name1}, {name2}')

        # Q3 matches -> Q4.2 RANSAC -> Q4.3 warp and alpha blend.
        pixels1, pixels2 = hw.feature_matching(img1, img2)
        H = hw.align_pair(pixels1, pixels2)
        blended = hw.stitch_blend(img1, img2, H)

        # stitch_blend returns floating-point BGR colors on the 0..255 scale.
        # Clip before converting: astype(uint8) alone can wrap out-of-range values.
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        prefix = f'blend_{Path(name1).stem}_{Path(name2).stem}'
        output_path = outdir / f'{prefix}.png'
        if not cv2.imwrite(str(output_path), blended):
            raise OSError(f'Cannot save image: {output_path}')
        pair_id = Path(name1).stem.split('_')[0]
        cv2.imwrite(str(outdir / f'blend_{pair_id}.png'), blended)
        np.save(outdir / f'{prefix}_H.npy', H)

        print(f'{name1} + {name2}: {len(pixels1)} raw matches')
        print(f'Canvas: {blended.shape[1]} x {blended.shape[0]} pixels')
        print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
