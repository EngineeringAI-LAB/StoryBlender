#!/usr/bin/env python3
"""
Script to download Python wheels for all packages in requirements_blender.txt
and update blender_manifest.toml with the wheel paths.
"""

import subprocess
import os
import sys
import re
from pathlib import Path
from tqdm import tqdm

# Platforms to download for all packages (ensures compatibility)
PLATFORMS = [
    'macosx_11_0_arm64',
    'manylinux_2_28_x86_64',
    'win_amd64'
]

PYTHON_VERSION = '3.11'

# Native packages supplied by Blender itself. Extension wheels are placed
# ahead of Blender's site-packages, so bundling these would shadow the builds
# tested with this Blender release.
BLENDER_BUNDLED_PACKAGES = {'bpy', 'numpy'}


def parse_requirements(requirements_file):
    """Parse conda-list output into pinned PyPI requirement strings.

    Keeping the recorded versions is important for native modules.  Silently
    upgrading one of them can produce a wheel that is newer than the libraries
    bundled with Blender, which is especially likely to fail DLL loading on
    Windows.
    """
    packages = []
    with open(requirements_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse conda list format: package_name version build channel
            parts = line.split()
            if len(parts) >= 2:
                package_name = parts[0]
                # Skip conda packages (not available on PyPI or built-in)
                skip_packages = {
                    'appnope', 'asttokens', 'bzip2', 'ca-certificates',
                    'comm', 'debugpy', 'decorator', 'executing', 'expat',
                    'ipykernel', 'ipython', 'jedi', 'jupyter_client', 'jupyter_core',
                    'krb5', 'libcxx', 'libedit', 'libffi', 'libsodium',
                    'matplotlib-inline', 'ncurses', 'nest-asyncio', 'openssl',
                    'parso', 'pexpect', 'pickleshare', 'pip', 'platformdirs',
                    'prompt-toolkit', 'psutil', 'ptyprocess', 'pure_eval',
                    'pygments', 'pyzmq', 'readline', 'setuptools', 'sqlite',
                    'stack_data', 'tk', 'tornado', 'traitlets', 'wheel',
                    'xz', 'zeromq', 'zlib',
                } | BLENDER_BUNDLED_PACKAGES
                
                if package_name not in skip_packages:
                    # Convert package name to PyPI format while preserving the
                    # version captured from the working environment.
                    pypi_name = package_name.replace('_', '-')
                    version = parts[1]
                    packages.append(f"{pypi_name}=={version}")
    
    return sorted(set(packages))


def _split_requirement(requirement):
    """Return the normalized distribution name and optional pinned version."""
    package, separator, version = requirement.partition('==')
    normalized_package = package.replace('-', '_').lower()
    return normalized_package, version if separator else None


def check_package_exists(requirement, wheels_dir, platform=None):
    """Check for the requested version and platform in the wheel directory."""
    package, version = _split_requirement(requirement)

    # Normalize package name (replace - with _ and vice versa for matching)
    version_pattern = version or '*'
    candidates = list(Path(wheels_dir).glob(f"{package}-{version_pattern}-*.whl"))
    if platform is None:
        return bool(candidates)

    for wheel in candidates:
        filename = wheel.name.lower()
        if filename.endswith('-none-any.whl'):
            return True
        if platform == 'win_amd64' and 'win_amd64' in filename:
            return True
        if platform == 'manylinux_2_28_x86_64':
            if 'manylinux' in filename and 'x86_64' in filename:
                return True
        if platform == 'macosx_11_0_arm64':
            if 'macosx' in filename and (
                'arm64' in filename or 'universal2' in filename
            ):
                return True
    return False


def download_package_wheels(requirement, wheels_dir, pbar=None):
    """
    Download wheels for a package across all platforms.
    This ensures we have both pure Python and platform-specific wheels as needed.
    """
    if pbar:
        pbar.set_description(f"Checking {requirement}")
    
    success = False
    
    # Try downloading for each platform
    # If it's pure Python, pip will download the same wheel multiple times (we'll dedupe later)
    # If it's platform-specific, we'll get the correct wheels for each platform
    for platform in PLATFORMS:
        if check_package_exists(requirement, wheels_dir, platform):
            if pbar:
                pbar.write(f"  ✓ {requirement} already has a wheel for {platform}")
            success = True
            continue

        if pbar:
            pbar.set_description(f"Downloading {requirement} for {platform}")

        cmd = [
            sys.executable, '-m', 'pip', 'download', requirement,
            '--dest', str(wheels_dir),
            '--only-binary=:all:',
            '--no-deps',
            f'--python-version={PYTHON_VERSION}',
            f'--platform={platform}'
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            success = True
        except subprocess.CalledProcessError:
            if pbar:
                pbar.write(
                    f"  Warning: Could not download {requirement} for {platform}"
                )
    
    if pbar:
        pbar.update(1)
    
    return success


def get_wheel_files(wheels_dir, requirements=None):
    """Get all wheel files in the wheels directory."""
    wheels = list(Path(wheels_dir).glob('*.whl'))

    # Do not put stale versions of explicitly pinned packages back into the
    # manifest, and never re-add a package supplied by Blender. This also
    # makes regeneration safe when upgrading a pin.
    pinned_versions = (
        dict(_split_requirement(item) for item in requirements)
        if requirements else {}
    )
    selected_wheels = []
    for wheel in wheels:
        parts = wheel.name.split('-', 2)
        if len(parts) >= 2:
            package = parts[0].replace('-', '_').lower()
            if package in BLENDER_BUNDLED_PACKAGES:
                continue
            pinned_version = pinned_versions.get(package)
            if pinned_version and parts[1] != pinned_version:
                continue
        selected_wheels.append(wheel)
    wheels = selected_wheels

    return sorted([f'./wheels/{w.name}' for w in wheels])


def update_manifest(manifest_file, wheel_paths):
    """Update the blender_manifest.toml file with the wheel paths."""
    print(f"\nUpdating {manifest_file}...")
    
    with open(manifest_file, 'r') as f:
        content = f.read()
    
    # Find the wheels section and replace it
    # Match the entire wheels array
    wheels_pattern = r'wheels\s*=\s*\[[\s\S]*?\]'
    
    # Format the wheels list
    wheels_str = 'wheels = [\n'
    for wheel in wheel_paths:
        wheels_str += f'  "{wheel}",\n'
    wheels_str += ']'
    
    # Replace the wheels section
    new_content = re.sub(wheels_pattern, wheels_str, content)
    
    with open(manifest_file, 'w') as f:
        f.write(new_content)
    
    print(f"Updated manifest with {len(wheel_paths)} wheels.")


def main():
    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    requirements_file = script_dir / 'requirements_blender.txt'
    wheels_dir = project_root / 'wheels'
    manifest_file = project_root / 'blender_manifest.toml'
    
    # Create wheels directory if it doesn't exist
    wheels_dir.mkdir(exist_ok=True)
    
    # Parse requirements
    print("Parsing requirements...")
    requirements = parse_requirements(requirements_file)
    print(f"Found {len(requirements)} packages to download.\n")
    
    # Download wheels for each package with progress bar
    with tqdm(total=len(requirements), desc="Downloading packages", unit="pkg") as pbar:
        for requirement in requirements:
            download_package_wheels(requirement, wheels_dir, pbar)
    
    # Download Windows-only conditional dependencies that pip may miss
    # when running on macOS/Linux (they are platform-specific extras).
    win_only_deps = ['win32-setctime', 'colorama']
    for dep in win_only_deps:
        if not check_package_exists(dep, wheels_dir, 'win_amd64'):
            print(f"Downloading Windows-only dependency: {dep}")
            try:
                subprocess.run(
                    ['pip', 'download', dep, '--dest', str(wheels_dir),
                     '--only-binary=:all:', '--no-deps'],
                    check=True, capture_output=True,
                )
            except subprocess.CalledProcessError:
                print(f"  Warning: Could not download {dep}")

    # Get all wheel files
    wheel_paths = get_wheel_files(wheels_dir, requirements)
    print(f"\nTotal wheels downloaded: {len(wheel_paths)}")
    
    # Update manifest
    update_manifest(manifest_file, wheel_paths)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
