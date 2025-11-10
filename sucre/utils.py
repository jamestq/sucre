import os

from contextlib import contextmanager
from pathlib import Path

__all__ = ["change_dir"]

@contextmanager
def change_dir(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)

def concat_path(base: str | Path, *args):
    if isinstance(base, str):
        base = Path(base)
    for value in args:
        base = base / str(value)
    return base

def make_dir(path: str | Path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)    
    return path