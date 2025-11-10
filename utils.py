from pathlib import Path
from typing import Optional


def find_file_by_prefix(item_dir: Path, prefix: str) -> Optional[Path]:
    """
    Find the first file in the item dir that starts with prefix.

    :param item_dir: Item directory path.
    :param prefix: Prefix string to match.
    :return: Path or None.
    """
    patterns = [f"{prefix}*", f"{prefix}.*"]

    for pat in patterns:
        res = list(item_dir.glob(pat))
        if len(res) > 0:
            return res[0]

    for f in item_dir.iterdir():
        if f.is_file() and f.name.lower().startswith(prefix.lower()):
            return f

    return None
