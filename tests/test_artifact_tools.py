import importlib.util
import pathlib
import tempfile
import unittest

import cv2
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATE = load("validate_artifacts", ROOT / "tools/validate_artifacts.py")
VISUALIZE = load("visualize_gaussian_ply", ROOT / "tools/visualize_gaussian_ply.py")
DOWNSAMPLE = load("downsample_pcd", ROOT / "tools/downsample_pcd.py")


class ArtifactToolTests(unittest.TestCase):
    def test_tum_validation_rejects_non_monotonic_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "trajectory.txt"
            path.write_text(
                "2.0 0 0 0 0 0 0 1\n1.0 0 0 0 0 0 0 1\n",
                encoding="utf-8",
            )
            result = VALIDATE.validate_tum(path)
            self.assertFalse(result["ok"])

    def test_pcd_and_ply_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            pcd = root / "map.pcd"
            pcd.write_text("VERSION .7\nPOINTS 12\nDATA ascii\n", encoding="utf-8")
            ply = root / "map.ply"
            ply.write_text(
                "ply\nformat ascii 1.0\nelement vertex 7\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
                "property float opacity\n"
                "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
                "property float rot_0\nproperty float rot_1\n"
                "property float rot_2\nproperty float rot_3\nend_header\n",
                encoding="utf-8",
            )
            self.assertEqual(VALIDATE.read_pcd_points(pcd), 12)
            self.assertEqual(VALIDATE.read_ply_vertex_count(ply), 7)
            self.assertTrue(VALIDATE.validate_ply(ply)["ok"])

    def test_gaussian_visualizer_recovers_position_and_color(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "map.ply"
            path.write_text(
                "ply\nformat ascii 1.0\nelement vertex 1\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
                "end_header\n1 2 3 0 0 0\n",
                encoding="utf-8",
            )
            xyz, rgb = VISUALIZE.read_gaussian_ply(path)
            self.assertEqual(xyz.tolist(), [[1.0, 2.0, 3.0]])
            self.assertEqual(rgb.tolist(), [[128, 128, 128]])

    def test_pcd_downsample_leaf_avoids_pcl_int32_grid_overflow(self):
        leaf = DOWNSAMPLE.safe_leaf_size((424.0, 281.0, 278.0), 0.15)
        self.assertGreater(leaf, 0.15)
        self.assertLessEqual(
            DOWNSAMPLE.voxel_cell_count((424.0, 281.0, 278.0), leaf),
            2**31 - 1,
        )

    def test_rendered_image_rejects_nearly_blank_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            blank = np.full((100, 100, 3), 255, dtype=np.uint8)
            useful = blank.copy()
            useful[:40, :40] = (20, 80, 160)
            blank_path = root / "blank.png"
            useful_path = root / "useful.png"
            cv2.imwrite(str(blank_path), blank)
            cv2.imwrite(str(useful_path), useful)
            self.assertFalse(VALIDATE.validate_rendered_image(blank_path)["ok"])
            self.assertTrue(VALIDATE.validate_rendered_image(useful_path)["ok"])


if __name__ == "__main__":
    unittest.main()
