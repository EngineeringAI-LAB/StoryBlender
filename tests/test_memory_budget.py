"""Test memory cutoffs and worker lifetime without starting Blender or renders."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import memory_budget
from src.gradio_app import render_batch_runner as runner


class MemoryBudgetTests(unittest.TestCase):
    def test_compressed_footprint_and_system_warning_stop_work(self):
        for footprint, pressure, stopped in ((2, 1, False), (8, 1, True), (33.4, 1, True), (2, 2, True), (2, 4, True)):
            with self.subTest(footprint=footprint, pressure=pressure), patch.object(memory_budget, 'memory_snapshot', return_value={
                'process_bytes': footprint * memory_budget.GIB, 'total_bytes': 24 * memory_budget.GIB, 'pressure': pressure,
            }):
                if stopped:
                    with self.assertRaises(memory_budget.MemoryBudgetExceeded):
                        memory_budget.ensure_memory_budget('Camera blocking')
                else:
                    memory_budget.ensure_memory_budget('Camera blocking')

    def test_failed_monitor_does_not_silently_allow_work(self):
        with patch.object(memory_budget, 'memory_snapshot', side_effect=OSError('unavailable')):
            with self.assertRaisesRegex(memory_budget.MemoryBudgetExceeded, 'monitoring unavailable'):
                memory_budget.ensure_memory_budget('Render preflight')

    def test_preflight_refusal_never_launches_worker(self):
        with patch.object(runner, 'require_closed_blender_ui'), patch.object(runner.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(memory_budget.MemoryBudgetExceeded, 'memory pressure'):
                runner.run_batch('blender', 'project.blend', 'worker.py', [1],
                                 Mock(side_effect=memory_budget.MemoryBudgetExceeded('memory pressure')))
            launch.assert_not_called()

    def fake_blender(self, directory, status=0, wait=False):
        path = Path(directory) / 'fake blender'
        path.write_text(f'#!{sys.executable}\nimport sys,time\nfrom pathlib import Path\n'
                        f'with Path({str(Path(directory) / "jobs")!r}).open("a") as f: f.write(sys.argv[-1]+"\\n")\n'
                        f'time.sleep({30 if wait else 0})\nsys.exit({status})\n')
        path.chmod(0o755)
        return str(path)

    def test_jobs_use_separate_sequential_processes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'require_closed_blender_ui'):
            executable = self.fake_blender(directory)
            runner.run_batch(executable, 'project.blend', 'worker.py', [1, 2], lambda *a, **k: None, poll_interval=0.01)
            self.assertEqual((Path(directory) / 'jobs').read_text().splitlines(), ['0', '1'])

    def test_failed_job_prevents_later_jobs(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'require_closed_blender_ui'):
            executable = self.fake_blender(directory, status=7)
            with self.assertRaisesRegex(RuntimeError, 'exit code 7'):
                runner.run_batch(executable, 'project.blend', 'worker.py', [1, 2], lambda *a, **k: None, poll_interval=0.01)
            self.assertEqual((Path(directory) / 'jobs').read_text().splitlines(), ['0'])

    def test_memory_limit_terminates_worker_and_prevents_later_jobs(self):
        pids = []
        def over_budget(operation, pid=None):
            if pid is not None:
                pids.append(pid)
                raise memory_budget.MemoryBudgetExceeded('memory limit')
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'require_closed_blender_ui'):
            executable = self.fake_blender(directory, wait=True)
            with self.assertRaisesRegex(memory_budget.MemoryBudgetExceeded, 'memory limit'):
                runner.run_batch(executable, 'project.blend', 'worker.py', [1, 2], over_budget, poll_interval=0.01)
        self.assertEqual(len(pids), 1)
        with self.assertRaises(ProcessLookupError):
            os.kill(pids[0], 0)

    def test_open_blender_ui_blocks_batch(self):
        with patch.object(runner.sys, 'platform', 'darwin'), patch.object(runner.subprocess, 'check_output', side_effect=[
            '999 /Applications/Blender.app/Contents/MacOS/Blender\n',
            '/Applications/Blender.app/Contents/MacOS/Blender project.blend',
        ]):
            with self.assertRaisesRegex(RuntimeError, "Close Blender's UI"):
                runner.require_closed_blender_ui()


if __name__ == '__main__':
    unittest.main()
