import pathlib
import unittest
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


class HKairportConfigTests(unittest.TestCase):
    def test_sensor_topics_and_time_offset(self):
        config = yaml.safe_load(
            (ROOT / "src/gs-livo/config/HKairport01.yaml").read_text()
        )
        self.assertEqual(config["common"]["img_topic"], "/left_camera/image")
        self.assertEqual(config["common"]["lid_topic"], "/livox/lidar")
        self.assertEqual(config["common"]["imu_topic"], "/livox/imu")
        self.assertEqual(config["time_offset"]["img_time_offset"], 0.1)
        self.assertEqual(config["evo"]["seq_name"], "HKairport01")
        self.assertTrue(config["evo"]["pose_output_en"])
        self.assertTrue(config["pcd_save"]["pcd_save_en"])
        self.assertTrue(config["output"]["gaussian_save_en"])

    def test_camera_intrinsics(self):
        camera = yaml.safe_load(
            (ROOT / "src/gs-livo/config/camera_MARS_LVIG.yaml").read_text()
        )
        self.assertEqual(camera["cam_width"], 2448)
        self.assertEqual(camera["cam_height"], 2048)
        self.assertEqual(camera["scale"], 0.25)
        self.assertAlmostEqual(camera["cam_fx"], 1444.431662789634)
        self.assertAlmostEqual(camera["cam_cy"], 1043.601026568268)

    def test_launch_loads_dataset_files(self):
        text = (ROOT / "src/gs-livo/launch/mapping_hkairport01.launch").read_text()
        self.assertIn("HKairport01.yaml", text)
        self.assertIn("camera_MARS_LVIG.yaml", text)
        self.assertIn("output/root_dir", text)


if __name__ == "__main__":
    unittest.main()
