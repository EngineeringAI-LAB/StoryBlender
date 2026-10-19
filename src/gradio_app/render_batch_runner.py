"""Supervise one Blender render worker at a time; also copied into exports."""

import argparse
import ast
import os
from pathlib import Path
import subprocess
import sys
import time


def require_closed_blender_ui():
    if sys.platform == "win32":
        return
    rows = subprocess.check_output(["ps", "-axo", "pid=,comm="], text=True).splitlines()
    for row in rows:
        parts = row.strip().split(None, 1)
        if len(parts) != 2 or Path(parts[1]).name.lower() != "blender" or int(parts[0]) == os.getpid():
            continue
        try:
            args = subprocess.check_output(["ps", "-p", parts[0], "-o", "args="], text=True)
        except subprocess.CalledProcessError:
            continue  # The process already exited.
        if not {"-b", "--background"}.intersection(args.split()):
            raise RuntimeError("Close Blender's UI before starting the headless render batch.")


def run_batch(blender, blend_file, worker_script, jobs, check_budget, poll_interval=0.5):
    """Keep compressed memory and render caches bounded by a worker lifetime."""
    require_closed_blender_ui()
    for index in range(len(jobs)):
        check_budget("Render preflight")
        command = [blender, "--background", blend_file, "--python-exit-code", "1",
                   "--python", worker_script, "--", "--job-index", str(index)]
        worker = subprocess.Popen(command)
        try:
            while worker.poll() is None:
                try:
                    check_budget(f"Render job {index + 1}/{len(jobs)}", pid=worker.pid)
                except Exception:
                    if worker.poll() is None:
                        raise
                time.sleep(poll_interval)
            if worker.returncode:
                raise RuntimeError(f"Render job {index + 1} failed with exit code {worker.returncode}; batch stopped.")
        finally:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
    print(f"All {len(jobs)} isolated render jobs completed.", flush=True)


def main(check_budget):
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", required=True)
    parser.add_argument("--blend", required=True)
    parser.add_argument("--worker", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    tree = ast.parse(Path(args.worker).read_text())
    jobs = next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "RENDER_JOBS" for target in node.targets))
    run_batch(args.blender, args.blend, args.worker, jobs, check_budget)
