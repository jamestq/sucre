import pandas as pd 

from pycaret.classification import setup, create_model, compare_models

from sucre import read

__all__ = ["train"]

def initializer(df: pd.DataFrame, targets: list[str], normalizers: list[str], transformers: list[str]):
    for target in targets:
        data = df.copy().drop(columns=targets)
        data[target] = df.copy()[target]        
        for normalizer in normalizers:
            setup(data=data, target=target, normalize=True, normalize_method=normalizer, experiment_name=f"Training for {target} with {normalizer}", fold_strategy="stratifiedkfold")
            yield data
        for transformer in transformers:
            setup(data=data, target=target, transformation=True, transformation_method=transformer, experiment_name=f"Training for {target} with {transformer}", fold_strategy="stratifiedkfold")
            yield data

def train(df: pd.DataFrame | None = None, **kwargs):
  df = read(df, **kwargs)
  dropped_columns = kwargs.get("drop", [])
  if dropped_columns:
    df.drop(columns=dropped_columns, inplace=True)    
  targets = kwargs.get("targets", [])
  if not targets:
    raise ValueError("No target columns specified for training.")
  normalizers = kwargs.get("normalize", [])
  transformers = kwargs.get("transform", [])
  models = kwargs.get("models", [])  
  for df_instance in initializer(df, targets, normalizers, transformers):
    model_instances = [create_model(model) for model in models]       
    compare_models(include=model_instances)
    

