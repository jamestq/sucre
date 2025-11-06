import pandas as pd
from pathlib import Path

from .base import read_data, export_data

__all__ = ["encode_data"]

def encode_data(df: pd.DataFrame | None = None, **kwargs) -> pd.DataFrame:
  input = kwargs.get("input", None) if df is None else None
  if input:
    path = Path(input)                
    df = read_data(path, **kwargs)        
  if df is None:
    raise ValueError("No DataFrame to filter")
  encodings = kwargs.get("encodings", {})
  for encoding in encodings:
    columns = encoding.get("columns", [])
    if columns and isinstance(columns, str):
      columns = [columns]
    for column in columns:
      order = encoding.get("order", None)
      default = encoding.get("default", None)
      conditions = encoding.get("conditions", {})
      if order:        
        transformer = {original : new for new, original in enumerate(order)}
        df[column] = df[column].map(transformer) 
      elif conditions and default is not None:
        for condition in conditions:
          operation = condition.get("operation", "")
          value = condition.get("value", None)
          encoded_value = condition.get("encoded_value", None)
          if operation == "==":
            df.loc[df[column] == value, column] = encoded_value
          elif operation == "!=":
            df.loc[df[column] != value, column] = encoded_value
          elif operation == "<":
            df.loc[df[column] < value, column] = encoded_value
          elif operation == "<=":
            df.loc[df[column] <= value, column] = encoded_value
          elif operation == ">":
            df.loc[df[column] > value, column] = encoded_value
          elif operation == ">=":
            df.loc[df[column] >= value, column] = encoded_value
      else:
        raise Warning(f"Encoding method not recognized or incomplete: {encoding}")
  export_data(df, **kwargs)
  return df