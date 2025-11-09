import pandas as pd
from pathlib import Path

__all__ = ["combine", "filter", "export_data", "read"]


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
            return pd.read_csv(path, index_col=None)
        case ".xlsx":
            kwargs["tab"] = 0 if "tab" not in kwargs else kwargs["tab"]
            return pd.read_excel(path, sheet_name=kwargs["tab"], index_col=None)
        case ".parquet":
            return pd.read_parquet(path)
        case _:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
def read(df_list: list[pd.DataFrame] = [], **kwargs) -> list[pd.DataFrame]:
    input = kwargs.get("input", None) if not df_list else None    
    if input:
        path = Path(input)
        df = read_data(path, **kwargs)
        df_list.append(df)
    if not df_list:
        raise ValueError("No DataFrame loaded")
    return df_list


def export_data(df_list: list[pd.DataFrame], **kwargs) -> None:
    if not kwargs.get("output", None) or not df_list:
        return
    output_path = Path(kwargs["output"])
    for i, df in enumerate(df_list):
        path = output_path if i == 0 else output_path.parent / f"{output_path.stem}_{i}{output_path.suffix}"
        match output_path.suffix.lower():
            case ".csv":
                df.to_csv(path, index=False)
            case ".xlsx":
                df.to_excel(path, index=False)
            case ".parquet":
                df.to_parquet(path, index=False)
            case _:
                raise ValueError(f"Unsupported output file type: {output_path.suffix}")


def combine(df_list: list[pd.DataFrame] = [], **kwargs) -> list[pd.DataFrame]:
    """Combine multiple DataFrames into one.

    Args:
        data (dict): A dictionary where keys are file paths and values are
                     dictionaries of keyword arguments to pass to the reader.

    Returns:
        pd.DataFrame: The combined DataFrame.
    """
    df = pd.concat(df_list, axis=1) if df_list else None
    inputs: list[dict] = kwargs.get("inputs", [])    
    for input in inputs:
        path = Path(input.pop("path", ""))
        new = read_data(path, **input)
        keep: list[str] = input.get("keep", [])
        drop: list[str] = input.get("drop", [])        
        if (            
            not isinstance(keep, list)
            or not isinstance(drop, list)
        ):
            raise ValueError("keep, and drop must be lists of strings")        
        if drop:
            new.drop(columns=drop, axis=1, inplace=True)
        if keep:
            new = new[keep]
        if df is None:
            df = new
        else:
            old_df = df.copy()
            df = pd.concat([df, new], axis=1)
            if df.shape[1] != old_df.shape[1] + new.shape[1]:
                raise ValueError(
                    "Concatenated DataFrame has unexpected number of columns"
                )
            if df.shape[0] != old_df.shape[0] or df.shape[0] != new.shape[0]:
                raise ValueError("Concatenated DataFrame has unexpected number of rows")        
    df.reset_index(inplace=True)    
    return [df]


def filter(df_list: list[pd.DataFrame] = [], **kwargs) -> list[pd.DataFrame]:
    df_list = read(df_list, **kwargs)
    filters = kwargs.get("filters", [])
    drop_columns = kwargs.get("drop", [])
    filtered_dfs = []
    for df in df_list:
        for filter in filters:
            operation = filter.get("operation", "")
            column = filter.get("column", "")
            value = filter.get("value", None)
            match operation:
                case "==":
                    df = df[df[column] == value]
                case "!=":
                    df = df[df[column] != value]
                case "<":
                    df = df[df[column] < value]
                case "<=":
                    df = df[df[column] <= value]
                case ">":
                    df = df[df[column] > value]
                case ">=":
                    df = df[df[column] >= value]
        df.drop(columns=drop_columns, inplace=True)   
        filtered_dfs.append(df)    
    return filtered_dfs