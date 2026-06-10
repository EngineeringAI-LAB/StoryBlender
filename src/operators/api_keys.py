try:
    from ..env_config import get_env
except ImportError:
    import sys
    from pathlib import Path

    for parent in Path(__file__).resolve().parents:
        if (parent / "env_config.py").is_file():
            sys.path.insert(0, str(parent))
            break

    from env_config import get_env


anyllm_api_key = get_env("ANYLLM_API_KEY")
anyllm_api_base = get_env("ANYLLM_API_BASE")
uthana_api_key = get_env("UTHANA_API_KEY")
