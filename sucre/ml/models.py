import torch.nn as nn

from skorch import NeuralNetClassifier
from sklearn.pipeline import Pipeline
from skorch.helper import DataFrameTransformer

class Net(nn.Module):
    def __init__(self, num_inputs=12, num_units_d1=200, num_units_d2=100):
        super(Net, self).__init__()

        self.dense0 = nn.Linear(num_inputs, num_units_d1)
        self.nonlin = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units_d1, num_units_d2)
        self.output = nn.Linear(num_units_d2, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.softmax(self.output(X))
        return X
    
class customNLLLoss(nn.Module):
    
    criterion = nn.NLLLoss()

    def __init__(self):
        super().__init__()

    def forward(self, logits, target):
        return self.criterion(logits, target.long())
    
def neural_network() -> Pipeline:    

  net = NeuralNetClassifier(
      module=Net,
      criterion=customNLLLoss, ### Including this as it throws an error of dtypes for target
      max_epochs=30,
      lr=0.1,
      batch_size=32,
      train_split=None
  )

  # Reference: https://github.com/pycaret/pycaret/issues/700#issuecomment-879700610
  nn_pipe = Pipeline(
      [
          ("transform", DataFrameTransformer()),
          ("net", net),
      ]
  )

  return nn_pipe
