import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class RunToolingTests(unittest.TestCase):
    def test_short_dry_run(self):
        completed = subprocess.run(
            [str(ROOT / "tools/run_hkairport01.sh"), "short", "--dry-run"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("offset=75", completed.stdout)
        self.assertIn("duration=20", completed.stdout)
        self.assertIn("rate=0.25", completed.stdout)


if __name__ == "__main__":
    unittest.main()
