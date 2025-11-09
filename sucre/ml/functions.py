import pandas as pd 
import os

from pathlib import Path

from pycaret.classification import setup, create_model, plot_model, pull, get_config
 
from sucre import read

from .neural_network import nn_classifier

__all__ = ["train"]

def initializer(df: pd.DataFrame, **kwargs):
    
    targets = kwargs.get("targets", [])
    if not targets:
      raise ValueError("No target columns specified for training.")

    normalizers = kwargs.get("normalize", [])
    transformers = kwargs.get("transform", [])

    feature_selection_settings = {
      "feature_selection": kwargs.get("feature_selection", True),
      "feature_selection_method": kwargs.get("feature_selection_method", "univariate"),
      "n_features_to_select": kwargs.get("n_features_to_select", 20)
    }
    
    for target in targets:
        data = df.copy().drop(columns=targets)        
        data[target] = df.copy()[target]   
        data[target] = data[target].astype("float")   
        for normalizer in normalizers:
            name = f"{target}_{normalizer}"
            setup(data=data, target=target, normalize=True, normalize_method=normalizer, session_id=42,experiment_name=name, fold_strategy="stratifiedkfold", use_gpu=True, **feature_selection_settings)            
            yield data, target, normalizer, get_config("X_train_transformed").shape[1]
        for transformer in transformers:
            name = f"{target}_{transformer}"
            setup(data=data, target=target, transformation=True, transformation_method=transformer, session_id=42, experiment_name=name, fold_strategy="stratifiedkfold", use_gpu=True, **feature_selection_settings)
            yield data, target, transformer, get_config("X_train_transformed").shape[1]
        if not transformers and not normalizers:
            name = f"{target}_none"
            setup(data=data, target=target, session_id=42, experiment_name=name, fold_strategy="stratifiedkfold", **feature_selection_settings)
            yield data, target, "notransformed", get_config("X_train_transformed").shape[1]

def save_results(target: str, results: dict):
  with pd.ExcelWriter(f"{target}.xlsx", engine="xlsxwriter") as writer:
    workbook=writer.book
    for data_transformer, model_results in results.items():
      for model_name, result in model_results.items():
        worksheet=workbook.add_worksheet(f"{data_transformer}_{model_name}")
        writer.sheets[f"{data_transformer}_{model_name}"] = worksheet
        result.to_excel(writer, sheet_name=f"{data_transformer}_{model_name}", startrow=0 , startcol=0)  

def export_data(index, results: dict, **kwargs):
  if kwargs.get("output", None) is None:
      return
  output = Path(kwargs["output"]) / str(index)
  output.mkdir(parents=True, exist_ok=True)
  os.makedirs(output, exist_ok=True)
  # Change working directory to output  
  cwd = Path.cwd()
  os.chdir(output)
  for target, result in results.items():
    save_results(target, result)
  os.chdir(cwd)
     
def get_plots(index, model, model_name, target, transformer, **kwargs):  
  output = Path(kwargs["output"]) / str(index) / "plots" / target / model_name / transformer  
  plot_types = ["confusion_matrix", "pr", "auc"]
  output.mkdir(parents=True, exist_ok=True)
  os.makedirs(output, exist_ok=True)
  # Change working directory to output  
  cwd = Path.cwd()
  os.chdir(output)      
  for plot_type in plot_types:    
    plot_model(model, plot=plot_type, save=True, scale=3)       
  os.chdir(cwd)
     
def train(df_list: list[pd.DataFrame] = [], **kwargs):
  df = read(df_list, **kwargs)
  for index, df in enumerate(df_list):
    if not isinstance(df, pd.DataFrame):
      raise ValueError("Input data must be a pandas DataFrame.")
    dropped_columns = kwargs.get("drop", [])
    if dropped_columns:
      df.drop(columns=dropped_columns, inplace=True)        
    models = kwargs.get("models", [])    
    training_results = dict()
    for _, target, data_transformer, size in initializer(df, **kwargs):            
      model_results = dict()
      for model_name in models:      
        model = create_model(model_name) if model_name != "neural_network" else nn_classifier(size)                  
        model_results[f"{model_name}"] = pull()   
        get_plots(index, model, model_name, target, data_transformer, **kwargs)                            
      if target in training_results:
        training_results[target][data_transformer] = model_results
      else:
        training_results[target] = {data_transformer: model_results}
    export_data(index, training_results, **kwargs)