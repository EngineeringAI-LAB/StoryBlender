"""Regression checks for cross-version wheel selection (no downloads)."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('download_wheels', Path(__file__).resolve().parents[1] / 'src/download_wheels.py')
wheels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wheels)


class WheelSelectionTests(unittest.TestCase):
    def test_python_311_native_wheel_cannot_satisfy_python_313(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'example-1.0-cp311-cp311-macosx_11_0_arm64.whl').touch()
            self.assertFalse(wheels.check_package_exists('example==1.0', root, 'macos-arm64'))
            (root / 'example-1.0-cp313-cp313-macosx_11_0_arm64.whl').touch()
            self.assertTrue(wheels.check_package_exists('example==1.0', root, 'macos-arm64'))
            self.assertFalse(wheels.check_package_exists('example==1.0', root, 'windows-x64'))

    def test_abi3_and_pure_wheels_are_accepted_but_stale_pins_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                'example-1.0-cp39-abi3-macosx_11_0_arm64.whl',
                'example-0.9-cp313-cp313-macosx_11_0_arm64.whl',
                'pure-1.0-py3-none-any.whl',
                'numpy-2.3.4-cp313-cp313-macosx_14_0_arm64.whl',
            ):
                (root / name).touch()
            self.assertTrue(wheels.check_package_exists('example==1.0', root, 'macos-arm64'))
            for platform in wheels.PLATFORMS:
                self.assertTrue(wheels.check_package_exists('pure==1.0', root, platform))
            selected = wheels.get_wheel_files(root, ['example==1.0', 'pure==1.0', 'numpy==2.3.4'])
            self.assertEqual(len(selected), 2)
            self.assertFalse(any('numpy' in w or '0.9' in w for w in selected))


if __name__ == '__main__':
    unittest.main()
