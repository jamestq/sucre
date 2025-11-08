import pandas as pd 
import os

from pathlib import Path

from pycaret.classification import setup, create_model, plot_model, pull
 
from sucre import read

from .neural_network import nn_classifier, generate_nn_plots

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
        for normalizer in normalizers:
            name = f"{target}_{normalizer}"
            setup(data=data, target=target, normalize=True, normalize_method=normalizer, experiment_name=name, fold_strategy="stratifiedkfold", use_gpu=True, **feature_selection_settings)
            yield data, target, normalizer        
        for transformer in transformers:
            name = f"{target}_{transformer}"
            setup(data=data, target=target, transformation=True, transformation_method=transformer, experiment_name=name, fold_strategy="stratifiedkfold", use_gpu=True, **feature_selection_settings)
            yield data, target, transformer
        if not transformers and not normalizers:
            name = f"{target}_none"
            setup(data=data, target=target, experiment_name=name, fold_strategy="stratifiedkfold", **feature_selection_settings)
            yield data, target, "notransformed"

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
     
def get_plots(model, model_name, target, transformer, **kwargs):
  plot_types = ["confusion_matrix", "pr", "auc"]
  output = Path(kwargs["output"]) / "plots" / target / model_name / transformer  
  output.mkdir(parents=True, exist_ok=True)
  os.makedirs(output, exist_ok=True)
  # Change working directory to output  
  cwd = Path.cwd()
  os.chdir(output)      
  for plot_type in plot_types:    
    plot_model(model, plot=plot_type, save=True, scale=3)       
  os.chdir(cwd)

def get_nn_plots(model, model_name, target, transformer, **kwargs):
  """Generate plots for neural network models"""
  output = Path(kwargs["output"]) / "plots" / target / model_name / transformer  
  output.mkdir(parents=True, exist_ok=True)
  cwd = Path.cwd()
  os.chdir(output)
  
  save_path = f"neural_network"
  generate_nn_plots(model.y_test, model.y_pred, save_path)
  
  os.chdir(cwd)
     
def train(df: pd.DataFrame | None = None, **kwargs):
  df = read(df, **kwargs)
  dropped_columns = kwargs.get("drop", [])
  if dropped_columns:
    df.drop(columns=dropped_columns, inplace=True)        
  models = kwargs.get("models", [])    
  training_results = dict()
  for _, target, data_transformer in initializer(df, **kwargs):            
    model_results = dict()
    for model_name in models:
      model = init_model(model_name)
      if model_name == "neural_network":
        result, final_model = model()
        model_results[f"{model_name}"] = result
        get_nn_plots(final_model, model_name, target, data_transformer, **kwargs)
      else:
        model_results[f"{model_name}"] = pull()   
        get_plots(model, model_name, target, data_transformer, **kwargs)                            
    if target in training_results:
      training_results[target][data_transformer] = model_results
    else:
      training_results[target] = {data_transformer: model_results}
  export_data(training_results, **kwargs)