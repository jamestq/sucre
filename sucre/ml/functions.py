import pandas as pd 
import os, contextlib

from pathlib import Path

from pycaret.classification import setup, create_model, compare_models, pull
 
from sucre import read

from .neural_network import nn_classifier

__all__ = ["train"]

def initializer(df: pd.DataFrame, targets: list[str], normalizers: list[str], transformers: list[str]):
    for target in targets:
        data = df.copy().drop(columns=targets)
        data[target] = df.copy()[target]        
        for normalizer in normalizers:
            name = f"{target}_{normalizer}"
            setup(data=data, target=target, normalize=True, normalize_method=normalizer, experiment_name=f"Training for {target} with {normalizer}", fold_strategy="stratifiedkfold")
            yield data, target, normalizer
        for transformer in transformers:
            name = f"{target}_{transformer}"
            setup(data=data, target=target, transformation=True, transformation_method=transformer, experiment_name=f"Training for {target} with {transformer}", fold_strategy="stratifiedkfold")
            yield data, target, transformer

def save_results(target: str, results: dict):
  with pd.ExcelWriter(f"{target}.xlsx", engine="xlsxwriter") as writer:
    workbook=writer.book
    for data_transformer, model_results in results.items():
      for model_name, result in model_results.items():
        worksheet=workbook.add_worksheet(f"{data_transformer}_{model_name}")
        writer.sheets[f"{data_transformer}_{model_name}"] = worksheet
        result.to_excel(writer, sheet_name=f"{data_transformer}_{model_name}", startrow=0 , startcol=0)  

def export_data(results: dict, **kwargs):
  if kwargs.get("output", None) is None:
      return
  output = Path(kwargs["output"])  
  output.mkdir(parents=True, exist_ok=True)
  os.makedirs(output, exist_ok=True)
  # Change working directory to output  
  cwd = Path.cwd()
  os.chdir(output)
  for target, result in results.items():
    save_results(target, result)
  os.chdir(cwd)


def init_model(model_name: str):
  match model_name.lower().strip():
    case "neural_network":        
        return nn_classifier
    case _:
        return create_model(model_name)
     
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
  training_results = dict()
  for _, target, data_transformer in initializer(df, targets, normalizers, transformers):            
    model_results = dict()
    for model_name in models:
      model = init_model(model_name)
      if model_name == "neural_network":
        model_results[f"{model_name}"] = model()
      else:
        model_results[f"{model_name}"] = pull()                             
    if target in training_results:
      training_results[target][data_transformer] = model_results
    else:
      training_results[target] = {data_transformer: model_results}
  export_data(training_results, **kwargs)  