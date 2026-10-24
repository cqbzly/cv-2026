"""Numerical checks for projection, coverage, outliers and a complete turn."""

import unittest

import numpy as np

from cylindrical import (cylindrical_inverse_maps, cylindrical_points,
                         cylindrical_warp, generate_cylindrical_panorama,
                         ransac_translation)


class CylindricalGeometryTests(unittest.TestCase):
    def test_round_trip_and_reject_back_of_camera(self):
        rng = np.random.default_rng(2)
        points = rng.uniform([0, 0], [319, 239], (1000, 2))
        uv = cylindrical_points(points, (240, 320), 280)
        x, y = cylindrical_inverse_maps(uv[:, 0], uv[:, 1], (240, 320), 280)
        np.testing.assert_allclose(np.column_stack((x, y)), points, atol=1e-4)
        x, y = cylindrical_inverse_maps([np.pi * 280], [0], (240, 320), 280)
        self.assertEqual(float(x[0]), -1)
        self.assertEqual(float(y[0]), -1)

    def test_black_is_valid_and_partial_edges_keep_brightness(self):
        image = np.full((81, 121, 3), [0, 80, 200], dtype=np.uint8)
        color, mask, _ = cylindrical_warp(image, 100)
        self.assertTrue(np.any((mask > 0) & (mask < 1)))
        self.assertEqual(float(mask.min()), 0)
        np.testing.assert_allclose(color[mask > 0],
                                   np.broadcast_to([0, 80, 200], color[mask > 0].shape),
                                   atol=1e-4)
        black, mask, origin = cylindrical_warp(np.zeros((81, 121), np.uint8), 100)
        self.assertFalse(black.any())
        self.assertEqual(mask[-origin[1], -origin[0]], 1)
        # The top edge's midpoint must survive the bounding rectangle.
        self.assertEqual(mask[-40 - origin[1], -origin[0]], 1)

    def test_translation_with_thirty_percent_outliers(self):
        rng = np.random.default_rng(7)
        source = rng.normal(size=(100, 2)) * 30
        target = source + [40, -2] + rng.normal(scale=0.1, size=(100, 2))
        target[:30] = rng.uniform(-100, 100, size=(30, 2))
        shift, mask = ransac_translation(source, target, threshold=1)
        np.testing.assert_allclose(shift, [40, -2], atol=0.05)
        self.assertEqual(int(mask.sum()), 70)

    def test_complete_turn_with_analytic_scene_and_exact_matches(self):
        # Render a continuous periodic world analytically into eight camera views.
        # This tests direction/sign, accumulation, wrap seam and color sampling.
        f, h, w, n = 90, 101, 160, 8
        cx, cy = (w - 1) / 2, (h - 1) / 2
        yy, xx = np.indices((h, w))
        theta = np.arctan2(xx - cx, f)
        v = (yy - cy) * np.cos(theta)

        def scene(angle, height):
            angle, height = np.broadcast_arrays(angle, height)
            return np.stack((100 + 70 * np.sin(angle),
                             100 + 60 * np.cos(2 * angle),
                             100 + 50 * np.sin(height / 20)), axis=-1)

        yaws = np.arange(n) * 2 * np.pi / n
        images = [np.rint(scene(theta + yaw, v)).astype(np.uint8) for yaw in yaws]
        camera = {id(image): yaw for image, yaw in zip(images, yaws)}
        angles, heights = np.meshgrid(np.linspace(-np.pi, np.pi, 512, endpoint=False),
                                      [-15, 0, 15])

        def matches(image1, image2):
            points, visible = [], []
            for image in (image1, image2):
                delta = (angles - camera[id(image)] + np.pi) % (2 * np.pi) - np.pi
                x = cx + f * np.tan(delta)
                y = cy + heights / np.cos(delta)
                valid = (np.abs(delta) < np.pi / 2) & (x >= 0) & (x <= w - 1)
                valid &= (y >= 0) & (y <= h - 1)
                points.append(np.stack((x, y), axis=-1))
                visible.append(valid)
            valid = visible[0] & visible[1]
            return points[0][valid], points[1][valid]

        for focal, closed in ((f, False), (f, True), (None, True)):
            with self.subTest(closed=closed, focal=focal):
                panorama, info = generate_cylindrical_panorama(
                    images, focal, feature_matcher=matches, threshold=0.01,
                    close_loop=closed, return_info=True)
                self.assertAlmostEqual(info['focal_length_pixels'], f, delta=0.01)
                expected_offsets = (yaws - yaws[n // 2]) * f
                np.testing.assert_allclose(np.array(info['offsets'])[:, 0],
                                           expected_offsets, atol=0.02)
                u0, v0 = info['canvas_origin_uv']
                # Central band is covered for the entire turn.
                row = int(-v0)
                global_u = u0 + np.arange(panorama.shape[1])
                expected = scene(global_u / f + yaws[n // 2], 0)
                # The open canvas rounds its bounds outward, so its outermost
                # columns may be background rather than part of the scene.
                columns = slice(None) if closed else slice(2, -2)
                np.testing.assert_allclose(panorama[row, columns],
                                           expected[columns], atol=1.1)
                if closed:
                    np.testing.assert_allclose(info['closure_error_before_correction'],
                                               [0, 0], atol=0.02)
                    self.assertEqual(panorama.shape[1], int(np.ceil(2 * np.pi * f)))


if __name__ == '__main__':
    unittest.main()
