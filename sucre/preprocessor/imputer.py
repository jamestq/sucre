import pandas as pd, miceforest as mf

from .base import read

__all__ = ["impute_data"]

def impute_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:    
    mean_value = df[column].mean()
    df[column].fillna(mean_value, inplace=True)
    return df

def impute_median(df: pd.DataFrame, column: str) -> pd.DataFrame:    
    median_value = df[column].median()
    df[column].fillna(median_value, inplace=True)
    return df

def impute_iterative(df: pd.DataFrame, column: str, exclude_columns: list[str] = []) -> pd.DataFrame:    
    data = df.select_dtypes(include=['number']).drop(columns=exclude_columns)    
    kds = mf.ImputationKernel(data=data, num_datasets=1, random_state=42, save_all_iterations_data=True)    
    kds.mice(5, verbose=True)
    data = kds.complete_data(dataset=0)
    if data is None: 
        raise Warning("Iterative imputation failed.")    
    df[column] = data[column]
    return df      

def match_impute(df: pd.DataFrame, column: str, method: str, exclude_columns: list[str] = []) -> pd.DataFrame:
    match method:
        case "mean":
            return impute_mean(df, column)
        case "median":
            return impute_median(df, column)
        case "iterative":
            return impute_iterative(df, column, exclude_columns=exclude_columns)
        case _:
            raise Warning(f"Imputation method not recognized: {method}")
    return df

def impute_data(df: pd.DataFrame | None = None, **kwargs) -> pd.DataFrame:
  df = read(df, **kwargs)   
  df.reset_index(inplace=True) 
  imputations = kwargs.get("imputations", [])  
  for imputation in imputations:
    columns = imputation.get("columns", [])
    if columns and isinstance(columns, str):
        columns = [columns]
    method = imputation.get("method", None) 
    exclude_columns = imputation.get("exclude_columns", [])
    if exclude_columns and isinstance(exclude_columns, str):
        exclude_columns = [exclude_columns]                 
    for column in columns:
        if method is None:
            raise Warning(f"Imputation method not recognized: {method}")
        else:
            df = match_impute(df, column, method, exclude_columns=exclude_columns)
  default_method = kwargs.get("default_method", None)
  if default_method is not None:
    empty_cols = df.columns[df.isnull().any()].tolist()
    for column in empty_cols:
        df = match_impute(df, column, default_method)  
  return df
              