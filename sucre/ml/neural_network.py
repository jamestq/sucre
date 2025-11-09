import torch

from pycaret.classification import create_model

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
  
  def forward(self, X):
    return self.network(X).squeeze()

def nn_classifier(input_size, epochs=30, lr=0.1, batch_size=32, random_state=42):

  from skorch import NeuralNetBinaryClassifier, NeuralNetBinaryRegressor
  from sklearn.pipeline import Pipeline
  from skorch.helper import DataFrameTransformer

  glucose_classifier = NeuralNetBinaryClassifier(
      module=GlucoseClassifier,
      module__input_size=input_size,
      max_epochs=epochs,
      lr=lr,
      batch_size=batch_size,
      train_split=None,      
  )
  
  pipe = Pipeline(
    [
        ("transform", DataFrameTransformer()),
        ("net", glucose_classifier),
    ],    
  )

  return create_model(pipe)