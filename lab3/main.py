from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from pathlib import Path
import json

# Paths are relative to this file, so the script also works from another folder.
DATA_DIR = Path(__file__).resolve().parent / 'data'
SHOW_FIGURES = False

DEFAULT_OUTPUT = 'outputs'

def visualize_matches(I1, I2, matches):
    # display two images side-by-side with matches
    # this code is to help you visualize the matches, you don't need
    # to use it to produce the results for the assignment

    # The two images need not have the same dimensions.
    I3 = np.zeros((max(I1.height, I2.height), I1.width + I2.width, 3), dtype=np.uint8)
    I3[:I1.height, :I1.width, :] = I1
    I3[:I2.height, I1.width:, :] = I2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(I3).astype(np.uint8))
    ax.plot(matches[:,0],matches[:,1],  '+r')
    ax.plot( matches[:,2] + I1.size[0], matches[:,3], '+r')
    ax.plot([matches[:,0], matches[:,2]+I1.size[0]],[matches[:,1], matches[:,3]], 'r')
    if SHOW_FIGURES:
        plt.show()
    return fig

def normalize_points(pts):
    # Normalize 2D Euclidean points (N,2); return homogeneous points (N,3) and T (3,3).
    # Coordinatewise standardization or a uniform scale may be used.
    # Handle point sets for which your normalization is undefined.
    pts = np.asarray(pts, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid 
    mean_distance = np.linalg.norm(centered, axis=1).mean()
    scale = np.sqrt(2.0) / mean_distance 
    transform = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0],
    ])
    homogeneous = np.column_stack([pts, np.ones(len(pts))])
    normalized = homogeneous @ transform.T
    return normalized, transform


def fit_fundamental(matches, normalized=True):
    # Estimate F (3,3) from matches (N,4), N >= 8.
    # Support normalized and unnormalized eight-point estimation.
    # Return F in the original pixel coordinates with rank 2.
    matches = np.asarray(matches, dtype=np.float64)
    points1 = matches[:, :2]
    points2 = matches[:, 2:]
    if normalized:
        points1, transform1 = normalize_points(points1)
        points2, transform2 = normalize_points(points2)
    else:
        points1 = np.column_stack([points1, np.ones(len(matches))])
        points2 = np.column_stack([points2, np.ones(len(matches))])
        transform1 = np.eye(3)
        transform2 = np.eye(3)
    x1, y1 = points1[:, 0], points1[:, 1]
    x2, y2 = points2[:, 0], points2[:, 1]
    matrix = np.column_stack([
        x2 * x1,
        x2 * y1,
        x2,
        y2 * x1,
        y2 * y1,
        y2,
        x1,
        y1,
        np.ones(len(matches))
    ])
    _, singular_values, Vt = np.linalg.svd(matrix, full_matrices=(len(matches)==8))
    fundamental = Vt[-1].reshape(3, 3)
    U, singular_values, Vt = np.linalg.svd(fundamental)
    singular_values[-1] = 0.0
    fundamental = U @ np.diag(singular_values) @ Vt
    fundamental = transform2.T @ fundamental @ transform1 
    fundamental /= np.linalg.norm(fundamental)
    return fundamental


def visualize_fundamental(matches, F, I1, I2):
    # Visualize the fundamental matrix in image 2
    N = len(matches)
    M = np.c_[matches[:,0:2], np.ones((N,1))].transpose()
    L1 = np.matmul(F, M).transpose() # transform points from 
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to matches(:,3:4)
    l = np.sqrt(L1[:,0]**2 + L1[:,1]**2)
    if np.any(l < np.finfo(float).tiny):
        raise ValueError('An epipolar line is undefined at an epipole.')
    L = np.divide(L1, np.kron(np.ones((3,1)), l).transpose())   # rescale the line
    pt_line_dist = np.multiply(L, np.c_[matches[:, 2:4], np.ones((N, 1))]).sum(axis = 1)
    closest_pt = matches[:, 2:4] - np.multiply(L[:, 0:2],np.kron(np.ones((2, 1)), pt_line_dist).transpose())

    # find endpoints of segment on epipolar line (for display purposes)
    pt1 = closest_pt - np.c_[L[:,1], -L[:,0]] * 10    # offset from the closest point is 10 pixels
    pt2 = closest_pt + np.c_[L[:,1], -L[:,0]] * 10

    # display points and segments of corresponding epipolar lines
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(I2).astype(np.uint8))
    ax.plot(matches[:, 2],matches[:, 3],  '+r')
    ax.plot([matches[:, 2], closest_pt[:, 0]],[matches[:, 3], closest_pt[:, 1]], 'r')
    ax.plot([pt1[:, 0], pt2[:, 0]],[pt1[:, 1], pt2[:, 1]], 'g')
    if SHOW_FIGURES:
        plt.show()
    return fig

def epipolar_distances(matches, F):
    # Supplied geometric evaluator: symmetric point-to-line distance in pixels.
    # Unlike the old algebraic residual, this is invariant to the scale of F.
    matches = np.asarray(matches, dtype=float)
    F = np.asarray(F, dtype=float)
    norm = np.linalg.norm(F)
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise ValueError('Invalid fundamental matrix.')
    F = F / norm
    N = len(matches)
    points1_homogeneous = np.c_[matches[:, :2], np.ones(N)]
    points2_homogeneous = np.c_[matches[:, 2:], np.ones(N)]
    lines2 = points1_homogeneous @ F.T
    lines1 = points2_homogeneous @ F
    product = np.sum(points2_homogeneous * lines2, axis=1)
    norm1 = np.sum(lines1[:, :2] ** 2, axis=1)
    norm2 = np.sum(lines2[:, :2] ** 2, axis=1)
    if np.any(norm1 <= np.finfo(float).tiny) or np.any(norm2 <= np.finfo(float).tiny):
        raise ValueError('Undefined epipolar line.')
    return np.sqrt(0.5 * product**2 * (1 / norm1 + 1 / norm2))


def evaluate_fundamental(matches, F):
    # Keep the old scalar return value; now report a geometric error in pixels.
    residual = np.mean(epipolar_distances(matches, F))
    return residual

def find_epipoles(F):
    # Added Q1: return e1,e2 (3,) in homogeneous coordinates.
    # Use the convention x2.T @ F @ x1 = 0; points at infinity are valid.
    U, singular_values, Vt = np.linalg.svd(F)
    epipole1 = Vt[-1].copy()
    epipole2 = U[:, -1].copy()
    epipole1 /= np.linalg.norm(epipole1)
    epipole2 /= np.linalg.norm(epipole2)
    return epipole1, epipole2

## Task 2: Camera calibration

def calc_projection(points_2d, points_3d):
    # Estimate P (3,4) from points_2d (N,2) and points_3d (N,3).
    # Use the lecture's linear calibration method, N >= 6.
    # Coordinate normalization is optional. Return P in the original coordinates.
    # Reject degenerate configurations.
    n = len(points_2d)
    if len(points_2d) != len(points_3d) or len(points_3d) < 6:
        raise ValueError("Degenerate configuration.")
    centered_3d = points_3d - points_3d.mean(axis=0)
    if np.linalg.matrix_rank(centered_3d) < 3:
        raise ValueError("Degenerate configuration.")
    A = np.zeros((2 * n, 12), dtype=np.float64)
    for i in range(n):
        x, y, z = points_3d[i]
        u, v = points_2d[i]
        A[i << 1] = [x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u]
        A[i << 1 | 1] = [0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v]
    _, singular_values, Vt = np.linalg.svd(A, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(A.shape) * singular_values[0]
    if singular_values[-2] <= tolerance:
        raise ValueError("Degenerate configuration.")
    P = Vt[-1].reshape(3, 4)
    return P

def rq_decomposition(P):
    # Return K, R, T from P = scale * K[R|T].
    # Keep K[2,2]=1, positive diagonal entries in K, R.T @ R = I, and det(R)=1.
    # scipy.linalg.rq may be used.
    if np.linalg.det(P[:, :3]) < 0:
        P = -P
    intrinsic, rotation = scipy.linalg.rq(P[:, :3])
    signs = np.where(np.diag(intrinsic) < 0, -1.0, 1.0)
    correction = np.diag(signs)
    intrinsic = intrinsic @ correction 
    rotation = correction @ rotation
    translation = np.linalg.solve(intrinsic, P[:, 3])
    scale = intrinsic[2][2]
    intrinsic /= scale 
    return intrinsic, rotation, translation


def evaluate_points(P, points_2d, points_3d):
    # Supplied evaluator: project 3D points and sum Euclidean pixel distances.
    # :param P: projection matrix 3 x 4
    # :param points_2d: 2D points N x 2
    # :param points_3d: 3D points N x 3
    # :return points_3d_proj: project 3D points to 2D by P
    # :return residual: residual of points_3d_proj and points_2d

    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(P, points_3d.T).T
    if np.any(np.abs(points_3d_proj[:, 2]) < 1e-12):
        raise ValueError('A 3D point projects to infinity.')
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual

## Task 3: Vanishing points (added)

def fit_vanishing_point(segments):
    # Added Q3: segments (N,4) = [x1,y1,x2,y2] from one group.
    # Return a finite homogeneous point [u,v,1], fitted by minimizing
    # equally weighted squared perpendicular distances to the supporting lines.
    segments = np.asarray(segments, dtype=np.float64)
    starts = segments[:, :2]
    ends = segments[:, 2:]
    directions = ends - starts 
    lengths = np.linalg.norm(directions, axis=1)
    normals = np.column_stack([
        -directions[:, 1],
        directions[:, 0]
    ])
    normals /= lengths[:, None]
    offsets = -np.sum(normals * starts, axis=1)
    position, residuals, rank, singular_values = np.linalg.lstsq(normals, -offsets, rcond=None)
    if rank < 2:
        raise ValueError("Cannot determine a unique finite vanishing point.")
    return np.array([position[0], position[1], 1.0])


def focal_from_vanishing_points(v1, v2, principal_point):
    # Added Q3: finite homogeneous vanishing points v1,v2 (3,), principal_point (2,).
    # Orthogonal directions, zero skew, square pixels. Return f>0 and K (3,3).
    # Report incompatible observations instead of forcing a positive result.
    point1 = v1[:2] / v1[2]
    point2 = v2[:2] / v2[2]
    offset1 = point1 - principal_point 
    offset2 = point2 - principal_point 
    focal_squared = -np.dot(offset1, offset2)
    if(focal_squared < 0):
        raise ValueError("observation mismatches camera assumption.")
    focal = np.sqrt(focal_squared)
    intrinsic = np.array([
        [focal, 0.0, principal_point[0]],
        [0.0, focal, principal_point[1]],
        [0.0, 0.0, 1.0]
    ])
    return focal, intrinsic

## Task 4: Triangulation

def triangulate_points(P1, P2, point1, point2):
    # Original single-point interface: P1,P2 (3,4), point1,point2 (2,).
    # Use linear least squares and return point_3d (3,).
    # Report points without finite Euclidean coordinates.
    projection1 = np.asarray(P1, dtype=np.float64)
    projection2 = np.asarray(P2, dtype=np.float64)
    pixel1 = np.asarray(point1, dtype=np.float64)
    pixel2 = np.asarray(point2, dtype=np.float64)
    A = np.vstack([
        pixel1[0] * projection1[2] - projection1[0],
        pixel1[1] * projection1[2] - projection1[1],
        pixel2[0] * projection2[2] - projection2[0],
        pixel2[1] * projection2[2] - projection2[1]
    ])
    _, singular_values, Vt = np.linalg.svd(A)
    tolerance = np.finfo(np.float64).eps * max(A.shape) * singular_values[0]
    if np.count_nonzero(singular_values > tolerance) < 3:
        raise ValueError("The observations do not determine a unique 3D point.")
    homogeneous_point = Vt[-1]
    if abs(homogeneous_point[3]) <= 1e-12:
        raise ValueError("The triangulated point is too far.")
    point_3d = homogeneous_point[:3] / homogeneous_point[3]
    return point_3d

## Task 5: Fundamental matrix estimation without ground-truth matches

import cv2


def fit_fundamental_without_gt(image1, image2):
    # Original interface: two uint8 RGB images -> fundamental_matrix, matches.
    # Use SIFT, brute-force descriptor matching, and RANSAC.
    # matches must contain only RANSAC inliers, in original pixel coordinates (N,4).
    # OpenCV SIFT/BFMatcher/findFundamentalMat are allowed here, as in the old code.
    # You may return OpenCV's F. Refitting with your own eight-point method is optional.
    # Use the normalized eight-point method if implementing the estimator yourself.
    # Raise ValueError if fewer than eight usable matches remain.
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    nearest_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    accepted_matches = []
    for neighbors in nearest_matches:
        if len(neighbors) < 2:
            continue 
        best, second_best = neighbors 
        if best.distance < 0.75 * second_best.distance:
            accepted_matches.append(best)
    if len(accepted_matches) < 8:
        raise ValueError("too few matched points.")
    points1 = np.array(
        [
            keypoints1[match.queryIdx].pt
            for match in accepted_matches
        ],
        dtype=np.float64 
    )
    points2 = np.array(
        [
            keypoints2[match.trainIdx].pt
            for match in accepted_matches
        ],
        dtype=np.float64,
    )
    fundamental, mask = cv2.findFundamentalMat(
        points1,
        points2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.99,
        maxIters=5000
    )
    inlier_mask = mask.ravel().astype(bool)
    if np.count_nonzero(inlier_mask) < 8:
        raise ValueError("too few inliers.")
    matches = np.column_stack([
        points1[inlier_mask],
        points2[inlier_mask]
    ])
    return fundamental, matches

def visualize_vanishing_points(image, segments, vanishing_points):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)
    for group, color in [('A', 'tab:blue'), ('B', 'tab:orange')]:
        v = vanishing_points[group][:2] / vanishing_points[group][2]
        for i, segment in enumerate(segments[group]):
            ax.plot(segment[[0, 2]], segment[[1, 3]], color=color, label=group if i == 0 else None)
            midpoint = (segment[:2] + segment[2:]) / 2
            direction = segment[2:] - segment[:2]
            foot = midpoint + direction * np.dot(v-midpoint, direction) / np.dot(direction, direction)
            ax.plot([midpoint[0], foot[0]], [midpoint[1], foot[1]], color=color, alpha=.25)
        ax.scatter(*v, marker='x', s=70, color=color)
        ax.annotate(group, v, xytext=(6, 8), textcoords='offset points')
    ax.legend();ax.set_aspect('equal')
    ax.set(xlabel='x / pixels', ylabel='y / pixels', title='Q3: segments and vanishing points')
    if SHOW_FIGURES:
        plt.show()
    return fig


def _save_figure(fig, output_dir, name):
    fig.savefig(output_dir / name, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_assignment(output_dir=None):
    # The original sequential Task 0--5 workflow, with Q3 added.
    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, matrices = {}, {}

    ## Task 0: Load data and visualize
    library_image1 = Image.open(DATA_DIR / 'library1.jpg').convert('RGB')
    library_image2 = Image.open(DATA_DIR / 'library2.jpg').convert('RGB')
    library_matches = np.loadtxt(DATA_DIR / 'library_matches.txt')
    lab_image1 = Image.open(DATA_DIR / 'lab1.jpg').convert('RGB')
    lab_image2 = Image.open(DATA_DIR / 'lab2.jpg').convert('RGB')
    lab_matches = np.loadtxt(DATA_DIR / 'lab_matches.txt')
    _save_figure(visualize_matches(library_image1, library_image2, library_matches), output_dir, 'library_matches.png')
    _save_figure(visualize_matches(lab_image1, lab_image2, lab_matches), output_dir, 'lab_matches.png')

    ## Task 1: Fundamental matrix
    metrics['q1'] = {}
    for name, matches, image1, image2 in [
            ('library', library_matches, library_image1, library_image2),
            ('lab', lab_matches, lab_image1, lab_image2)]:
        metrics['q1'][name] = {}
        for normalized in [False, True]:
            F = fit_fundamental(matches, normalized=normalized)
            distances = epipolar_distances(matches, F)
            method = 'normalized' if normalized else 'unnormalized'
            metrics['q1'][name][method] = dict(mean_px=float(distances.mean()), median_px=float(np.median(distances)))
            matrices[f'{name}_{method}_F'] = F
            print(name, method, 'mean epipolar distance:', evaluate_fundamental(matches, F))
            _save_figure(visualize_fundamental(matches, F, image1, image2), output_dir, f'q1_{name}_{method}.png')
        e1, e2 = find_epipoles(F)
        e1, e2 = e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)
        metrics['q1'][name]['epipoles'] = dict(e1=e1, e2=e2,
            residual1=float(np.linalg.norm(F @ e1) / np.linalg.norm(F)),
            residual2=float(np.linalg.norm(F.T @ e2) / np.linalg.norm(F)))
        print(name, 'epipoles:', e1, e2)

    ## Task 2: Camera calibration and camera matrices (old Tasks 2 and 3)
    lab_points_3d = np.loadtxt(DATA_DIR / 'lab_3d.txt')
    projection_matrix = dict()
    metrics['q2'] = {}
    for key, points_2d in zip(['lab_a', 'lab_b'], [lab_matches[:, :2], lab_matches[:, 2:]]):
        P = calc_projection(points_2d, lab_points_3d)
        points_3d_proj, residual = evaluate_points(P, points_2d, lab_points_3d)
        distance = np.mean(np.linalg.norm(points_2d - points_3d_proj, axis=1))
        print(key, 'reprojection sum:', residual, 'mean:', distance)
        projection_matrix[key] = P
        errors = np.linalg.norm(points_2d - points_3d_proj, axis=1)
        metrics['q2'][key] = dict(reprojection_sum_px=float(residual), mean_px=float(errors.mean()),
                                  rmse_px=float(np.sqrt(np.mean(errors**2))))
        image = lab_image1 if key == 'lab_a' else lab_image2
        fig, ax = plt.subplots();ax.imshow(image)
        ax.plot(*points_2d.T, 'ro', fillstyle='none', label='Measured')
        ax.plot(*points_3d_proj.T, 'b+', label='Reprojected');ax.legend();ax.set_title(key)
        if SHOW_FIGURES: plt.show()
        _save_figure(fig, output_dir, f'q2_{key}.png')
    projection_library_a = np.loadtxt(DATA_DIR / 'library1_camera.txt')
    projection_library_b = np.loadtxt(DATA_DIR / 'library2_camera.txt')
    projection_matrix['library_a'] = projection_library_a
    projection_matrix['library_b'] = projection_library_b
    intrinsics = {}
    for key, P in projection_matrix.items():
        K, R, T = rq_decomposition(P.copy())
        C = -R.T @ T
        reconstructed = K @ np.c_[R, T]
        scale = np.sum(P * reconstructed) / np.sum(reconstructed**2)
        checks = dict(orthogonality_error=float(np.linalg.norm(R.T @ R - np.eye(3))),
                      det_R=float(np.linalg.det(R)),
                      P_error_up_to_scale=float(np.linalg.norm(P-scale*reconstructed) / np.linalg.norm(P)))
        if (checks['orthogonality_error'] > 1e-6 or abs(checks['det_R']-1) > 1e-6
                or checks['P_error_up_to_scale'] > 1e-6 or abs(K[2,2]-1) > 1e-6
                or np.any(np.diag(K) <= 0)):
            raise ValueError('Inconsistent K, R, T decomposition for '+key)
        metrics['q2'].setdefault(key, {}).update(dict(K=K, R=R, T=T, C=C, **checks))
        intrinsics[key] = K
        # Preserve the estimated/provided camera and its scale for triangulation.
        matrices[key+'_P'] = P.copy()
        print(key, 'K, R, T, C:', K, R, T, C, sep='\n')

    ## Task 3: Focal length from vanishing points (added)
    records = np.genfromtxt(DATA_DIR / 'lab1_segments.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
    segments = {group: np.array([[r['x1'], r['y1'], r['x2'], r['y2']]
                                for r in records if r['group'] == group]) for group in ['A', 'B']}
    vanishing_points = {group: fit_vanishing_point(values) for group, values in segments.items()}
    center = np.array(lab_image1.size, dtype=float) / 2
    focal, K = focal_from_vanishing_points(vanishing_points['A'], vanishing_points['B'], center)
    reference_K = intrinsics['lab_a']
    reference_focal = np.sqrt(reference_K[0, 0] * reference_K[1, 1])
    focal_q2, _ = focal_from_vanishing_points(vanishing_points['A'], vanishing_points['B'], reference_K[:2, 2])
    metrics['q3'] = dict(vanishing_points=vanishing_points, focal_px=focal, K=K,
                         reference_focal_px=float(reference_focal),
                         relative_difference=float(abs(focal-reference_focal)/reference_focal),
                         focal_with_q2_principal_point=focal_q2)
    print('Q3 focal lengths (image center / Q2 principal point / Q2 reference):', focal, focal_q2, reference_focal)
    _save_figure(visualize_vanishing_points(lab_image1, segments, vanishing_points), output_dir, 'q3_vanishing_points.png')

    ## Task 4: Triangulation
    lab_points_3d_estimated = []
    for point_2d_a, point_2d_b, point_3d_gt in zip(lab_matches[:, :2], lab_matches[:, 2:], lab_points_3d):
        point_3d_estimated = triangulate_points(projection_matrix['lab_a'], projection_matrix['lab_b'], point_2d_a, point_2d_b)
        lab_points_3d_estimated.append(point_3d_estimated)
    lab_points_3d_estimated = np.stack(lab_points_3d_estimated)
    _, residual_a = evaluate_points(projection_matrix['lab_a'], lab_matches[:, :2], lab_points_3d_estimated)
    _, residual_b = evaluate_points(projection_matrix['lab_b'], lab_matches[:, 2:], lab_points_3d_estimated)
    errors3d = np.linalg.norm(lab_points_3d_estimated-lab_points_3d, axis=1)
    metrics['q4'] = dict(lab_mean_3d_error=float(errors3d.mean()), lab_rmse_3d_error=float(np.sqrt(np.mean(errors3d**2))),
                         lab_reprojection_sum_px=[float(residual_a), float(residual_b)])
    print('Q4 lab mean 3D error:', errors3d.mean())
    fig = plt.figure();ax = fig.add_subplot(projection='3d')
    ax.scatter(*lab_points_3d.T, marker='o', facecolors='none', edgecolors='r', label='Given')
    ax.scatter(*lab_points_3d_estimated.T, marker='x', label='Triangulated');ax.legend()
    if SHOW_FIGURES: plt.show()
    _save_figure(fig, output_dir, 'q4_lab_3d.png')

    library_points_3d_estimated = []
    for point_2d_a, point_2d_b in zip(library_matches[:, :2], library_matches[:, 2:]):
        point_3d_estimated = triangulate_points(projection_matrix['library_a'], projection_matrix['library_b'], point_2d_a, point_2d_b)
        library_points_3d_estimated.append(point_3d_estimated)
    library_points_3d_estimated = np.stack(library_points_3d_estimated)
    _, residual_a = evaluate_points(projection_matrix['library_a'], library_matches[:, :2], library_points_3d_estimated)
    _, residual_b = evaluate_points(projection_matrix['library_b'], library_matches[:, 2:], library_points_3d_estimated)
    metrics['q4']['library_reprojection_sum_px'] = [float(residual_a), float(residual_b)]
    print('Q4 library reprojection sums:', residual_a, residual_b)
    matrices['lab_points_3d'] = lab_points_3d_estimated
    matrices['library_points_3d'] = library_points_3d_estimated

    ## Task 5: Fundamental matrix estimation without ground-truth matches
    metrics['q5'] = {}
    for name in ['house', 'gaudi']:
        image1 = Image.open(DATA_DIR / f'{name}1.jpg').convert('RGB')
        image2 = Image.open(DATA_DIR / f'{name}2.jpg').convert('RGB')
        F, matches = fit_fundamental_without_gt(np.array(image1), np.array(image2))
        residual = evaluate_fundamental(matches, F)
        metrics['q5'][name] = dict(F=F, inlier_count=len(matches), mean_inlier_residual_px=float(residual))
        print(name, 'inliers:', len(matches), 'mean residual:', residual)
        matrices[name+'_F'] = F;matrices[name+'_inlier_matches'] = matches
        _save_figure(visualize_matches(image1, image2, matches), output_dir, f'q5_{name}_inliers.png')
        _save_figure(visualize_fundamental(matches, F, image1, image2), output_dir, f'q5_{name}_epipolar.png')

    (output_dir / 'metrics.json').write_text(json.dumps(metrics, default=lambda a: a.tolist(), indent=2)+'\n')
    np.savez(output_dir / 'matrices.npz', **matrices)
    return metrics


if __name__ == '__main__':
    run_assignment()

