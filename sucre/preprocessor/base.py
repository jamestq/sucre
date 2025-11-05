import pandas as pd
from pathlib import Path

__all__ = ["read_data", "combine"]

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
            kwargs["tab"] = 0 if "tab" not in kwargs else kwargs["tab"]
            return pd.read_excel(path, sheet_name=kwargs["tab"])
        case ".parquet":
            return pd.read_parquet(path)
        case _:
            raise ValueError(f"Unsupported file type: {path.suffix}")

def combine(**kwargs) -> pd.DataFrame:
    """Combine multiple DataFrames into one.

    Args:
        data (dict): A dictionary where keys are file paths and values are
                     dictionaries of keyword arguments to pass to the reader.

    Returns:
        pd.DataFrame: The combined DataFrame.
    """
    inputs: list[dict] = kwargs.get("inputs", [])
    df: pd.DataFrame | None = None
    id_columns: str | list[str] = []
    for input in inputs:
        path = Path(input.pop("path", ""))                
        new = read_data(path, **input)
        keep: list[str] = input.get("keep", [])
        drop: list[str] = input.get("drop", [])
        id_columns = input.get("id_columns", [])                                                    
        if not isinstance(id_columns, list) or not isinstance(keep, list) or not isinstance(drop, list):
            raise ValueError("id_columns, keep, and drop must be lists of strings")
        if df is None:
            df = new
        else:
            if id_columns:
                new.drop(columns=id_columns, axis=1, inplace=True)            
            if drop:
                new.drop(columns=drop, axis=1, inplace=True)
            if keep:
                new = new[keep] 
            old_df = df.copy()            
            df = pd.concat([df, new], axis=1)
            if df.shape[1] != old_df.shape[1] + new.shape[1]:
                raise ValueError("Concatenated DataFrame has unexpected number of columns")
            if df.shape[0] != old_df.shape[0] or df.shape[0] != new.shape[0]:
                raise ValueError("Concatenated DataFrame has unexpected number of rows")                
    if kwargs.get("output"):
        output_path = Path(kwargs["output"])
        match output_path.suffix.lower():
            case ".csv":
                df.to_csv(output_path, index=False)
            case ".xlsx":
                df.to_excel(output_path, index=False)
            case ".parquet":
                df.to_parquet(output_path, index=False)
            case _:
                raise ValueError(f"Unsupported output file type: {output_path.suffix}")