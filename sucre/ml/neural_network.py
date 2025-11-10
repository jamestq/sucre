import torch, numpy as np

from pycaret.classification import create_model

from skorch.helper import DataFrameTransformer

from skorch import NeuralNetBinaryClassifier

class CustomNeuralNetBinaryClassifier(NeuralNetBinaryClassifier):

  def convert_float32(self, X):
    if not torch.backends.mps.is_available():
      return X
    if isinstance(X, dict):
        X = {k: v.astype(np.float32) if hasattr(v, 'astype') else v for k, v in X.items()}
    elif hasattr(X, 'astype'):
        X = X.astype(np.float32)   
    return X

  def predict(self, X):
    X = self.convert_float32(X)    
    return super().predict(X)

  def predict_proba(self, X):
    X = self.convert_float32(X)
    return super().predict_proba(X)

class GlucoseClassifier(torch.nn.Module):
    
  def __init__(self, input_size):
    super(GlucoseClassifier, self).__init__()
    self.network = torch.nn.Sequential(
        torch.nn.Linear(input_size, 128),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(128),
        torch.nn.Dropout(0.3),        
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),        
        torch.nn.BatchNorm1d(64),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(32, 1)
    )    
    self.predict = False
  
  def forward(self, X=None, **kwargs):     
    if X is None:      
      cat_tensors = []
      for v in kwargs.values():
        # Ensure it's a tensor with correct dtype and shape
        if not isinstance(v, torch.Tensor):
          raise ValueError("All inputs must be torch Tensors")        
        # Ensure it's 2D (add dimension if needed)
        if v.dim() == 1:
            v = v.unsqueeze(1)        
        cat_tensors.append(v)      
      X = torch.cat(cat_tensors, dim=1)        
    return self.network(X).squeeze()
  

class CustomDataFrameTransformer(DataFrameTransformer):

  def __init__(self):
    # Set float_dtype to np.float32 for MPS compatibility
    # Keep int_dtype as default (np.int64) for categorical data
    super().__init__(
        treat_int_as_categorical=False,
        float_dtype=np.float32,
        int_dtype=np.int64
    )
   
  def fit(self, df, y=None, **fit_params):
    self = super().fit(df, y, **fit_params)
    if hasattr(df, "columns"):
      self.feature_names_in_ = df.columns.to_numpy()
    return self

  def inverse_transform(self, X):
    """Passthrough inverse_transform for PyCaret compatibility.
    
    This method is required by PyCaret's predict_model but doesn't
    need to do anything since we're not modifying the predictions.
    """
    return X

def nn_classifier(input_size, epochs=30, lr=0.1, batch_size=32):

  from skorch import NeuralNetBinaryClassifier
  from sklearn.pipeline import Pipeline  

  # Automatically select best available device
  if torch.cuda.is_available():
      device = 'cuda'
  elif torch.backends.mps.is_available():
      device = 'mps'
  else:
      device = 'cpu'

  glucose_classifier = CustomNeuralNetBinaryClassifier(
      module=GlucoseClassifier,
      module__input_size=input_size,
      max_epochs=epochs,
      lr=lr,
      batch_size=batch_size,
      train_split=None,      
      device=device
  )
  
  pipe = Pipeline(
    [
        ("transform", CustomDataFrameTransformer()),
        ("net", glucose_classifier),        
    ],    
  )

  return create_model(pipe)