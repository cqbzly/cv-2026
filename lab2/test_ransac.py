"""Run from any directory: python HW2/test_ransac.py --pair 1_1.jpg 1_2.jpg."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import homework2 as hw
from utils import ransac_with_inliers, reprojection_errors


def main():
    parser = argparse.ArgumentParser(description='Compare Q3 matches and RANSAC inliers')
    parser.add_argument('--pair', nargs=2, default=['1_1.jpg', '1_2.jpg'])
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--threshold', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    images = [cv2.imread(str(root / 'Problem2Images' / name)) for name in args.pair]
    for name, image in zip(args.pair, images):
        if image is None:
            raise FileNotFoundError(root / 'Problem2Images' / name)
    img1, img2 = images
    pixels1, pixels2 = hw.feature_matching(img1, img2)
    H, inliers = ransac_with_inliers(
        pixels1, pixels2, hw.compute_homography,
        iterations=args.iterations, threshold=args.threshold, seed=args.seed
    )
    p1, p2 = np.asarray(pixels1), np.asarray(pixels2)
    total = len(p1)
    count = int(inliers.sum())
    errors = reprojection_errors(H, p1, p2)
    stats = {
        'pair': args.pair,
        'raw_matches': total,
        'inlier_count': count,
        'inlier_ratio': count / total,
        'removed_count': total - count,
        'threshold_pixels': args.threshold,
        'iterations': args.iterations,
        'seed': args.seed,
        'mask_definition': 'Best sampled model consensus, used for refitting H',
        'refit_inlier_median_error_pixels': float(np.median(errors[inliers])),
        'refit_inlier_max_error_pixels': float(np.max(errors[inliers])),
        'refit_consensus_points_above_threshold': int((errors[inliers] >= args.threshold).sum()),
    }

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    canvas[:h2, w1:] = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(3, 1, figsize=(16, 15))
    views = [
        (np.ones(total, dtype=bool), f'Q3: raw matches ({total})', 'deepskyblue'),
        (inliers, f'RANSAC inliers: {count}/{total} ({count / total:.1%})', 'lime'),
        (~inliers, f'Removed matches ({total - count})', 'red'),
    ]
    for ax, (mask, title, color) in zip(axes, views):
        ax.imshow(canvas)
        for (x1, y1), (x2, y2) in zip(p1[mask], p2[mask]):
            ax.plot([x1, x2 + w1], [y1, y2], color=color,
                    linewidth=0.8, marker='.', markersize=3, alpha=0.8)
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()

    outdir = root / 'outputs'
    outdir.mkdir(exist_ok=True)
    prefix = f'ransac_{Path(args.pair[0]).stem}_{Path(args.pair[1]).stem}'
    plot_path = outdir / f'{prefix}.png'
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    np.savez(outdir / f'{prefix}.npz', H=H, inliers=inliers,
             pixels1=p1, pixels2=p2, refit_errors=errors)
    (outdir / f'{prefix}.json').write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f'Visualization: {plot_path}')


if __name__ == '__main__':
    main()
