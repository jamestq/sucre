import pandas as pd
from pathlib import Path

from .reader import *

def run(command: str, path: Path) -> pd.DataFrame:
    match command:
        case "import":
            return read_data(path)
        case _:
            raise ValueError(f"Unsupported command: {command}")
