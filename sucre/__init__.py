import pandas as pd

from .preprocessor import combine


def run(command: str, data: dict) -> pd.DataFrame:
    match command:        
        case "combine_data":
            return combine(**data)
        case _:
            raise ValueError(f"Unsupported command: {command}")
