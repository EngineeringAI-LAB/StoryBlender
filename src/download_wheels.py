#!/usr/bin/env python3
"""Download pinned Python 3.13 wheels for Blender 5.2 LTS.

Run with a Python environment containing pip and packaging. Every selected
platform must have all dependencies before the manifest is updated.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path
from packaging.tags import compatible_tags, cpython_tags, mac_platforms
from packaging.utils import canonicalize_name, parse_wheel_filename

PYTHON_VERSION = (3, 13)
BLENDER_BUNDLED_PACKAGES = {'bpy', 'numpy'}
PLATFORMS = {
    'macos-arm64': list(mac_platforms((14, 0), 'arm64')),
    'linux-x64': [
        *(f'manylinux_2_{minor}_x86_64' for minor in range(28, 16, -1)),
        'manylinux2014_x86_64',
    ],
    'windows-x64': ['win_amd64'],
}


def parse_requirements(requirements_file):
    """Read explicit PyPI pins; exclude packages provided by Blender."""
    requirements = []
    for line in Path(requirements_file).read_text().splitlines():
        line = line.partition('#')[0].strip()
        if not line:
            continue
        if not re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+', line):
            raise ValueError(f'Expected a pinned requirement, got {line!r}')
        name, version = line.split('==')
        name = canonicalize_name(name)
        if name not in BLENDER_BUNDLED_PACKAGES:
            requirements.append(f'{name}=={version}')
    return sorted(set(requirements))


def target_tags(platform):
    platforms = PLATFORMS[platform]
    return set(cpython_tags(PYTHON_VERSION, platforms=platforms)) | set(
        compatible_tags(PYTHON_VERSION, interpreter='cp313', platforms=platforms)
    )


def check_package_exists(requirement, wheels_dir, platform):
    name, version = requirement.split('==')
    tags = target_tags(platform)
    for path in Path(wheels_dir).glob('*.whl'):
        wheel_name, wheel_version, _, wheel_tags = parse_wheel_filename(path.name)
        if wheel_name == canonicalize_name(name) and str(wheel_version) == version:
            if wheel_tags & tags:
                return True
    return False


def download_package_wheels(requirements, wheels_dir, platform):
    missing = [r for r in requirements if not check_package_exists(r, wheels_dir, platform)]
    if not missing:
        return
    cmd = [
        sys.executable, '-m', 'pip', 'download', *missing,
        '--dest', str(wheels_dir), '--only-binary=:all:', '--no-deps',
        '--python-version=3.13', '--implementation=cp', '--abi=cp313',
    ]
    for tag in PLATFORMS[platform]:
        cmd.extend(['--platform', tag])
    subprocess.run(cmd, check=True)


def get_wheel_files(wheels_dir, requirements):
    pins = dict(r.split('==') for r in requirements)
    tags = set().union(*(target_tags(p) for p in PLATFORMS))
    wheels = []
    for path in Path(wheels_dir).glob('*.whl'):
        name, version, _, wheel_tags = parse_wheel_filename(path.name)
        if name in BLENDER_BUNDLED_PACKAGES or pins.get(name) != str(version):
            continue
        if wheel_tags & tags:
            wheels.append(f'./wheels/{path.name}')
    return sorted(wheels)


def update_manifest(manifest_file, wheel_paths):
    path = Path(manifest_file)
    wheels = 'wheels = [\n' + ''.join(f'  "{w}",\n' for w in wheel_paths) + ']'
    path.write_text(re.sub(r'wheels\s*=\s*\[[\s\S]*?\]', wheels, path.read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', choices=list(PLATFORMS), action='append')
    parser.add_argument('--check', action='store_true', help='Validate without downloading')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    wheels_dir = root / 'wheels'
    wheels_dir.mkdir(exist_ok=True)
    requirements = parse_requirements(root / 'src/requirements_blender.txt')
    platforms = args.platform or list(PLATFORMS)
    for platform in platforms:
        if not args.check:
            download_package_wheels(requirements, wheels_dir, platform)
        missing = [r for r in requirements if not check_package_exists(r, wheels_dir, platform)]
        if missing:
            raise RuntimeError(f'Missing Python 3.13 wheels for {platform}: {missing}')
        print(f'Validated {len(requirements)} dependencies for {platform}')
    # Partial downloads must not publish an incomplete manifest.
    if set(platforms) == set(PLATFORMS):
        update_manifest(root / 'blender_manifest.toml', get_wheel_files(wheels_dir, requirements))


if __name__ == '__main__':
    main()
