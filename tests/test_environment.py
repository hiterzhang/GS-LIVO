import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_environment", ROOT / "tools" / "check_environment.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EnvironmentCheckTests(unittest.TestCase):
    def test_parse_rosbag_topics(self):
        text = """
topics:
    - topic: /left_camera/image/compressed
      type: sensor_msgs/CompressedImage
    - topic: /livox/imu
      type: sensor_msgs/Imu
    - topic: /livox/lidar
      type: livox_ros_driver/CustomMsg
"""
        self.assertEqual(
            MODULE.parse_rosbag_topics(text),
            {
                "/left_camera/image/compressed",
                "/livox/imu",
                "/livox/lidar",
            },
        )

    def test_missing_required_topics(self):
        missing = MODULE.missing_required_topics({"/livox/imu"})
        self.assertEqual(
            missing,
            ["/left_camera/image/compressed", "/livox/lidar"],
        )


if __name__ == "__main__":
    unittest.main()
