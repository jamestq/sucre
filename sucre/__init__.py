import pandas as pd

from .preprocessor import combine, filter, encode_data

def run(command: str, data: dict, df: pd.DataFrame = None) -> pd.DataFrame:
    match command:        
        case "combine_data":
            return combine(df, **data)        
        case "filter_data":
            return filter(df, **data)
        case "encode_data":
            return encode_data(df, **data)
        case _:
            raise ValueError(f"Unsupported command: {command}")