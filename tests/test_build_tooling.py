import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class BuildToolingTests(unittest.TestCase):
    def test_bootstrap_pins_dependencies(self):
        text = (ROOT / "tools" / "bootstrap_dependencies.sh").read_text()
        self.assertIn("6c886c8e5d83997806e00294826d528cea3581dd", text)
        self.assertIn("libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip", text)

    def test_cmake_has_no_author_absolute_path(self):
        text = (ROOT / "src" / "gs-livo" / "CMakeLists.txt").read_text()
        self.assertNotIn("/home/sheng", text)
        self.assertIn("GS_LIVO_LIB3DGS_ROOT", text)

    def test_lib3dgs_uses_local_source_targets(self):
        text = (ROOT / "src" / "lib3dgs" / "CMakeLists.txt").read_text()
        self.assertIn("add_subdirectory(${SIMPLE_KNN_SOURCE_DIR}", text)
        self.assertIn("add_subdirectory(${TINYPLY_SOURCE_DIR}", text)
        self.assertNotIn('set(SIMPLE_KNN_DIR "/usr/local")', text)


if __name__ == "__main__":
    unittest.main()
