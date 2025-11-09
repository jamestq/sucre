import pandas as pd
import traceback

from pathlib import Path

from .preprocessor import *

from .ml import (
    train,
)

COMMANDS = {
    "combine_data": combine,
    "filter_data": filter,
    "encode_data": encode_data,
    "impute_data": impute_data,    
    "train": train,        
}

def save_wrapper(fn: callable, df_list: list[pd.DataFrame], **data):
    results: list[pd.DataFrame] = fn(df_list, **data)           
    export_data(results, **data)
    return results

def custom(command, **kwargs) -> callable:
    path = kwargs.get("path", None)
    if path is None:
        raise ValueError("No path provided for custom preprocessor")
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Custom preprocessor path does not exist: {path}")
    import importlib, sys
    # Load the module from the file path
    spec = importlib.util.spec_from_file_location("custom_module", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_module"] = module
    spec.loader.exec_module(module)
    # Get the function from the module
    if not hasattr(module, command):
        raise ValueError(f"Function '{command}' not found in {path}")
    func = getattr(module, command)
    return func


def run(command: str, data: dict, df_list: list[pd.DataFrame] = []) -> pd.DataFrame:
    fn = COMMANDS.get(command, None)    
    if fn is None:
        try:
            fn = custom(command, **data)
            data.pop("path", None)  # Remove path from data
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Unsupported command: {command}") from e
    return save_wrapper(fn, df_list, **data)    
