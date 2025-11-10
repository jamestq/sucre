import pandas as pd 
import os

from pathlib import Path

from pycaret.classification import setup, create_model, plot_model, pull, get_config, predict_model, compare_models
 
from sucre import read

from ..utils import change_dir, concat_path, make_dir

from .neural_network import nn_classifier

__all__ = ["train"]

def initializer(df: pd.DataFrame, **kwargs):
    
    targets = kwargs.get("targets", [])
    if not targets:
      raise ValueError("No target columns specified for training.")

    normalizers = kwargs.get("normalize", [])
    transformers = kwargs.get("transform", [])

    setup_settings = {
      "session_id": 42,
      "fold_strategy": "stratifiedkfold",
      "use_gpu": True,
      "train_size": 0.9
    }

    feature_selection_settings = {
      "feature_selection": kwargs.get("feature_selection", True),
      "feature_selection_method": kwargs.get("feature_selection_method", "univariate"),
      "n_features_to_select": kwargs.get("n_features_to_select", 20)
    }
    
    for target in targets:
        data = df.copy().drop(columns=targets)        
        target_column = df.copy()[target].astype("float")        
        for normalizer in normalizers:
            name = f"{target}_{normalizer}"
            setup(data=data, target=target_column, normalize=True, normalize_method=normalizer, experiment_name=name, **setup_settings, **feature_selection_settings)
            yield data, target, normalizer, get_config("X_train_transformed").shape[1]
        for transformer in transformers:
            name = f"{target}_{transformer}"
            setup(data=data, target=target_column, transformation=True, transformation_method=transformer, experiment_name=name, **setup_settings, **feature_selection_settings)
            yield data, target, transformer, get_config("X_train_transformed").shape[1]
        if not transformers and not normalizers:
            name = f"{target}_none"
            setup(data=data, target=target_column, experiment_name=name, **setup_settings, **feature_selection_settings)
            yield data, target, "notransformed", get_config("X_train_transformed").shape[1]        

def save_data(output, *args, **kwargs):    
  output = make_dir(concat_path(Path(output), *args))   
  if output is None:
    print("No output path provided, skipping save.")
    return
  file_suffix = kwargs.pop("file_suffix", "")    
  with change_dir(output):
    for item, df in kwargs.items():           
       if isinstance(df, pd.DataFrame):
          file_name = f"{item}_{file_suffix}".strip() if file_suffix else item
          df.to_excel(f"{file_name}.xlsx", index=True)    

def get_plots(index, model, model_name, target, transformer, **kwargs):
  if kwargs.get("output", None) is None:
      print("No output path provided, skipping save.")
      return
  output = make_dir(concat_path(Path(kwargs["output"]), index, target, model_name, transformer))
  plot_types = ["confusion_matrix", "pr", "auc", "calibration", "class_report", "error"]  
  with change_dir(output):
    for plot_type in plot_types:    
      plot_model(model, plot=plot_type, save=True, scale=3)  
     
def train(df_list: list[pd.DataFrame] = [], **kwargs):
  df = read(df_list, **kwargs)
  for index, df in enumerate(df_list):
    if not isinstance(df, pd.DataFrame):
      raise ValueError("Input data must be a pandas DataFrame.")
    dropped_columns = kwargs.get("drop", [])
    if dropped_columns:
      df.drop(columns=dropped_columns, inplace=True)        
    models = kwargs.get("models", [])        
    for _, target, data_transformer, size in initializer(df, **kwargs):
      model_instances = []                  
      for model_name in models:      
        model = create_model(model_name) if model_name != "neural_network" else nn_classifier(size)                  
        training = pull()                                                       
        get_plots(index, model, model_name, target, data_transformer, **kwargs)            
        prediction = predict_model(model)     
        metrics = pull()                   
        save_data(kwargs.get("output", None), index, target, model_name, data_transformer, training=training, prediction=prediction, metrics=metrics)
        model_instances.append(model)
      compare_models(model_instances)
      comparison = pull()
      save_data(kwargs.get("output", None), index, target, comparison=comparison, file_suffix=data_transformer)            