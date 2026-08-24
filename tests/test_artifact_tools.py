import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATE = load("validate_artifacts", ROOT / "tools/validate_artifacts.py")
VISUALIZE = load("visualize_gaussian_ply", ROOT / "tools/visualize_gaussian_ply.py")


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


if __name__ == "__main__":
    unittest.main()
