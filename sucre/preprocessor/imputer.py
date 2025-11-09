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

def impute_iterative(df: pd.DataFrame, columns: list[str], exclude_columns: list[str] = [], num_datasets: int = 1) -> list[pd.DataFrame]:    
    """Perform iterative imputation and return multiple imputed datasets"""    
    data = df.select_dtypes(include=['number']).drop(columns=exclude_columns)    
    kds = mf.ImputationKernel(data=data, num_datasets=num_datasets, random_state=42, save_all_iterations_data=True)    
    kds.mice(5, verbose=True)    
    # Return multiple imputed datasets
    imputed_dfs = []
    for i in range(num_datasets):
        data_imputed = kds.complete_data(dataset=i)
        if data_imputed is None: 
            raise Warning(f"Iterative imputation failed for dataset {i}.")
        df_copy = df.copy()
        for column in columns:
            df_copy[column] = data_imputed[column]
        imputed_dfs.append(df_copy)
    
    return imputed_dfs

def match_impute(df: pd.DataFrame, column: str, method: str) -> list[pd.DataFrame]:
    """Returns a list of DataFrames (single item for non-iterative methods, multiple for iterative)"""
    match method:
        case "mean":
            return [impute_mean(df, column)]
        case "median":
            return [impute_median(df, column)]        
        case _:
            raise Warning(f"Imputation method not recognized: {method}")

def impute_data(df_list: list[pd.DataFrame] = [], **kwargs) -> list[pd.DataFrame]:
    df_list = read(df_list, **kwargs)
    
    result_dfs = []
    num_datasets = kwargs.get("num_datasets", 1)  # Number of datasets for iterative imputation
    
    for df in df_list:        
        df = df.reset_index(drop=True)
        current_dfs = [df]  # Start with the original dataframe
        
        imputations = kwargs.get("imputations", [])  
        for imputation in imputations:
            columns = imputation.get("columns", [])
            if columns and isinstance(columns, str):
                columns = [columns]
            method = imputation.get("method", None) 
            exclude_columns = imputation.get("exclude_columns", [])
            if exclude_columns and isinstance(exclude_columns, str):
                exclude_columns = [exclude_columns]
            
            # Process each column and handle multiple datasets
            temp_dfs = []
            if method == "iterative":
                for curr_df in current_dfs:
                    imputed = impute_iterative(curr_df, columns, 
                                              exclude_columns=exclude_columns, 
                                              num_datasets=num_datasets)                    
                    temp_dfs.extend(imputed)                                
            else:
                for column in columns:
                    if method is None:
                        raise Warning(f"Imputation method not recognized: {method}")
                    
                    # Apply imputation to all current dataframes
                    for curr_df in current_dfs:
                        imputed = match_impute(curr_df, column, method, 
                                            exclude_columns=exclude_columns, 
                                            num_datasets=num_datasets)
                        temp_dfs.extend(imputed)            
            if temp_dfs:
                current_dfs = temp_dfs
        
        # Handle default method for remaining null columns
        default_method = kwargs.get("default_method", None)
        if default_method is not None:
            temp_dfs = []
            for curr_df in current_dfs:
                empty_cols = curr_df.columns[curr_df.isnull().any()].tolist()
                if empty_cols:
                    for column in empty_cols:
                        imputed = match_impute(curr_df, column, default_method)
                        curr_df = imputed[0]  # Default methods return single df
                temp_dfs.append(curr_df)
            current_dfs = temp_dfs
        
        result_dfs.extend(current_dfs)
    
    return result_dfs
