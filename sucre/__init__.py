import pandas as pd

from .preprocessor import (
    combine, 
    filter, 
    encode_data,
    impute_data, 
    export_data,
)

from .explorer import predict

COMMANDS = {
    "combine_data": combine,
    "filter_data": filter,
    "encode_data": encode_data,
    "impute_data": impute_data,
    "predict": explorer,
}

def save_wrapper(fn: callable, df: pd.DataFrame, **data):
    result: pd.DataFrame = fn(df, **data)           
    export_data(result, **data)
    return result


def run(command: str, data: dict, df: pd.DataFrame = None) -> pd.DataFrame:
    fn = COMMANDS.get(command, None)    
    if fn is None:
        raise ValueError(f"Unsupported command: {command}")
    return save_wrapper(fn, df, **data)
