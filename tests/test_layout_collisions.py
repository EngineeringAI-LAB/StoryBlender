"""Layout validation must not start Blender or change its shared dependencies."""

from pathlib import Path
import sys
import math
import numpy as np
from scipy.optimize import lsq_linear
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from operators.layout_artist_operators.generate_layout_description import SceneLayoutVerifier


class LayoutCollisionTests(unittest.TestCase):
    def test_collision_volume_and_tolerance_without_external_processes(self):
        for offset, expected_errors in [(0.5, 1), (0.99, 0), (2.0, 0)]:
            with self.subTest(offset=offset):
                assets = [{"asset_id": name, "location": {"x": x, "y": 0, "z": 0},
                           "rotation": {"z": 0}} for name, x in [("a", 0), ("b", offset)]]
                dimensions = {name: {"w": 1, "d": 1, "h": 1} for name in ["a", "b"]}
                verifier = SceneLayoutVerifier({"assets": assets}, dimensions, scene_id=1)
                with patch("subprocess.Popen", side_effect=AssertionError("Unexpected external process")) as popen:
                    errors = verifier._check_occlusions()
                popen.assert_not_called()
                self.assertEqual(len(errors), expected_errors)
                if errors:
                    self.assertIn("vol 0.500000m³", errors[0]["detail"])

    def test_exact_gaps_and_contacts_for_rotated_and_thin_boxes_without_fcl(self):
        cases = [
            # diagonal XY and vertical separation use Euclidean surface distance
            (1, 1, 1, 0, 1.03, 1.04, 1.12, 0.13),
            # rotated square tangent, then separated along its local X axis
            (1, 1, 1, 45, math.sqrt(2), math.sqrt(2), 0, 1.0),
            # extremely thin rug-like boxes, including overlap and near-contact
            (3.5, 1.2, 0.00001, 0, 0.5, 0, 0, 0),
            (3.5, 1.2, 0.00001, 0, 0, 0, 0.02001, 0.02),
            (1, 1, 1, 0, 1.05, 0, 0, 0.05),
        ]
        with patch("trimesh.collision.CollisionManager", side_effect=AssertionError("Unsafe FCL solver")):
            for w, d, h, angle, x, y, z, gap in cases:
                with self.subTest(dimensions=(w, d, h), position=(x, y, z)):
                    assets = [
                        {"asset_id": "a", "location": {"x": 0, "y": 0, "z": 0}, "rotation": {"z": angle}},
                        {"asset_id": "b", "location": {"x": x, "y": y, "z": z}, "rotation": {"z": angle},
                         "anchor_asset_id": "a", "contact": True},
                    ]
                    dims = {name: {"w": w, "d": d, "h": h} for name in ("a", "b")}
                    verifier = SceneLayoutVerifier({"assets": assets}, dims, scene_id=1)
                    self.assertAlmostEqual(verifier._box_gap("a", "b"), gap, places=8)
                    self.assertEqual(verifier._check_contact(assets[1]) is None, gap <= verifier.CONTACT_TOL)
                    assets[1]["contact"] = False
                    verifier._check_occlusions()

    def test_rotated_disjoint_boxes_with_overlapping_axis_aligned_bounds(self):
        assets = [{"asset_id": name, "location": {"x": x, "y": y, "z": 0}, "rotation": {"z": 45}}
                  for name, x, y in [("a", 0, 0), ("b", -0.2, 0.2)]]
        dims = {name: {"w": 4, "d": 0.1, "h": 1} for name in ("a", "b")}
        verifier = SceneLayoutVerifier({"assets": assets}, dims, scene_id=1)
        self.assertAlmostEqual(verifier._box_gap("a", "b"), math.sqrt(0.08) - 0.1)
        self.assertEqual(verifier._check_occlusions(), [])

    def test_box_distances_match_independent_bounded_least_squares(self):
        rng = np.random.default_rng(52)
        for index in range(30):
            angles = rng.uniform(-180, 180, 2)
            sizes = rng.uniform(0.05, 4, (2, 3))
            if index % 3 == 0:
                sizes[:, 2] = 0.00001
            positions = rng.uniform(-2, 2, (2, 3))
            assets = [{"asset_id": name, "location": dict(zip(('x', 'y', 'z'), positions[i])),
                       "rotation": {"z": angles[i]}} for i, name in enumerate(('a', 'b'))]
            dims = {name: dict(zip(('w', 'd', 'h'), sizes[i])) for i, name in enumerate(('a', 'b'))}
            verifier = SceneLayoutVerifier({"assets": assets}, dims, scene_id=1)
            rotations = []
            for angle in np.deg2rad(angles):
                c, s = math.cos(angle), math.sin(angle)
                rotations.append(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))
            coefficients = np.column_stack((rotations[0] @ np.diag(sizes[0]),
                                             -rotations[1] @ np.diag(sizes[1])))
            centers = positions + sizes * np.array([0, 0, 0.5])
            reference = lsq_linear(coefficients, centers[1] - centers[0], bounds=(-0.5, 0.5),
                                   method="bvls", tol=1e-14, max_iter=1000)
            self.assertAlmostEqual(verifier._box_gap('a', 'b'), np.linalg.norm(reference.fun), places=6)


if __name__ == "__main__":
    unittest.main()
