import pandas as pd
from pathlib import Path

__all__ = ["read_data"]

def read_data(path: Path, **kwargs) -> pd.DataFrame:
    """Import data from a file.

    Only CSV, Excel and Parquet files are supported.

    Args:
        path (Path): The path to the file.

    Returns:
        pd.DataFrame: The imported data as a DataFrame.
    """
    match path.suffix.lower():
        case ".csv":
            return pd.read_csv(path)
        case ".xlsx":
            if "tab" not in kwargs:
                kwargs["tab"] = 0
            return pd.read_excel(path, sheet_name=kwargs["tab"])
        case ".parquet":
            return pd.read_parquet(path)
        case _:
            raise ValueError(f"Unsupported file type: {path.suffix}")
