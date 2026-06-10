"""
Test SSL certificate fix for verify_asset_availability.

Run this standalone (outside Blender) to diagnose and confirm the fix:
    python test_ssl_fix.py
"""

import os
import ssl
import sys

ASSET_ID = "fire_hydrant"
URL = f"https://api.polyhaven.com/info/{ASSET_ID}"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://polyhaven.com/",
}


def get_ssl_ca_bundle() -> str | bool:
    """
    Return a valid CA bundle path for requests.verify.

    Resolution order:
    1. certifi.where() — if the file actually exists on disk.
    2. ssl.get_default_verify_paths().cafile — system CA bundle (macOS / Linux).
    3. ssl.get_default_verify_paths().capath — system CA directory.
    4. False  — disable verification as a last resort (logs a warning).
    """
    # 1. Try certifi first
    try:
        import certifi
        ca = certifi.where()
        if os.path.isfile(ca):
            print(f"[ssl] Using certifi bundle: {ca}")
            return ca
        else:
            print(f"[ssl] certifi path does not exist: {ca}")
    except Exception as e:
        print(f"[ssl] certifi unavailable: {e}")

    # 2. Try system CA file
    paths = ssl.get_default_verify_paths()
    print(f"[ssl] ssl.get_default_verify_paths() = {paths}")
    if paths.cafile and os.path.isfile(paths.cafile):
        print(f"[ssl] Using system CA file: {paths.cafile}")
        return paths.cafile

    # 3. Try system CA directory
    if paths.capath and os.path.isdir(paths.capath):
        print(f"[ssl] Using system CA path: {paths.capath}")
        return paths.capath

    # 4. Last resort
    print("[ssl] WARNING: No valid CA bundle found. Disabling TLS verification.")
    return False


def main():
    import requests

    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"requests: {requests.__version__}")
    print("=" * 60)

    ca_bundle = get_ssl_ca_bundle()

    print(f"\nTesting {URL}")
    print(f"verify={ca_bundle!r}")

    try:
        resp = requests.get(URL, headers=HEADERS, timeout=10, verify=ca_bundle)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Asset name : {data.get('name', 'N/A')}")
            print(f"Asset type : {data.get('type', 'N/A')}")
            print("SUCCESS: verify_asset_availability fix works!")
        else:
            print(f"Unexpected status: {resp.status_code}")
    except Exception as e:
        print(f"FAILED: {e}")


if __name__ == "__main__":
    main()
