"""Helpers for RANSAC and its report, without changing the homework interfaces."""

from itertools import combinations

import numpy as np


def reprojection_errors(H, src, dst):
    """Forward reprojection distances in pixels; invalid projections get inf."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    H = np.asarray(H, dtype=float)
    # Homographies differing by a nonzero scale must get the same validity test.
    norm = np.linalg.norm(H)
    if not np.isfinite(H).all() or not np.isfinite(norm) or norm == 0:
        return np.full(len(src), np.inf)
    projected_h = (H / norm @ np.column_stack((src, np.ones(len(src)))).T).T
    valid = (np.isfinite(projected_h).all(axis=1)
             & (np.abs(projected_h[:, 2]) > 1e-12))
    errors = np.full(len(src), np.inf)
    projected = projected_h[valid, :2] / projected_h[valid, 2:3]
    errors[valid] = np.linalg.norm(projected - dst[valid], axis=1)
    return errors


def ransac_with_inliers(pixels_1, pixels_2, compute_homography,
                        iterations=1000, threshold=3.0, seed=None):
    """Return (refitted H, best consensus mask), in original match order.

    compute_homography is the student's solver, passed as a callback to avoid
    circular imports. The mask is the largest sampled model's consensus set,
    which is then used to refit H, as required by Q4.2. Refit residuals may
    change slightly; the mask deliberately records that original consensus.
    """
    src = np.asarray(pixels_1, dtype=float)
    dst = np.asarray(pixels_2, dtype=float)

    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(len(src), dtype=bool)
    best_count = 0
    best_error = np.inf

    for _ in range(iterations):
        indices = rng.choice(len(src), size=4, replace=False)
        H = compute_homography(src[indices].tolist(), dst[indices].tolist())
        errors = reprojection_errors(H, src, dst)
        inliers = errors < threshold
        count = int(inliers.sum())
        if count < 4:
            continue
        error = float(np.mean(errors[inliers] ** 2))
        if count > best_count or (count == best_count and error < best_error):
            best_count = count
            best_error = error
            best_inliers = inliers


    H = compute_homography(src[best_inliers].tolist(), dst[best_inliers].tolist())
    return H, best_inliers
