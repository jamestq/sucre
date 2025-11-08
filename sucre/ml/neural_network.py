# Linear schedule with warmup
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pycaret.classification import get_config

def classification_metrics(y_test, y_pred):
    y_pred_binary = (y_pred >= 0.5).astype(int)  # Convert probabilities to binary predictions
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    roc_auc = roc_auc_score(y_test, y_pred)

    return {
        "Accuracy": float(accuracy),
        "ROC AUC": float(roc_auc),
        "Recall": float(recall),
        "Precision": float(precision),        
        "F1": float(f1),     
        "Kappa": float((2 * (precision * recall)) / (precision + recall + 1e-8)),  # Adding small epsilon to avoid division by zero
        "MCC": float(((accuracy * (1 - accuracy)) - ((1 - precision) * (1 - recall))) / 
                     (((accuracy + (1 - precision)) * (accuracy + (1 - recall)))**0.5 + 1e-8))  # Matthews correlation coefficient   
    }

def linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
   def lr_lambda(current_step):
       if current_step < num_warmup_steps:
           # Warmup: linearly increase from 0 to 1
           return float(current_step) / float(max(1, num_warmup_steps))
       else:
           # Linear decay: decrease from 1 to 0
           return max(0.0, float(num_training_steps - current_step) / 
                     float(max(1, num_training_steps - num_warmup_steps)))
   
   return LambdaLR(optimizer, lr_lambda)

class GlucosePredictor(nn.Module):
    def __init__(self, input_size):
        super(GlucosePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),  # Single output for regression
        )

    def forward(self, x):
        return self.network(x).squeeze()

class GlucoseClassifier(nn.Module):
    def __init__(self, input_size):
        super(GlucoseClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),  # Single output for regression
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x).squeeze())
    
def nn_classifier(epochs=200, warmpup_ratio=0.1, k_folds=10):   

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_test, y_train, y_test = get_config("X_train_transformed"), get_config("X_test_transformed"), get_config("y_train_transformed"), get_config("y_test_transformed")     
    # Ensure consistent format - convert to numpy arrays safely
    # PyCaret returns DataFrames/Series, sklearn can return either DataFrames or numpy arrays
    X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
    X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else np.asarray(X_test)
    y_train_np = y_train.values if isinstance(y_train, (pd.Series, pd.DataFrame)) else np.asarray(y_train)
    y_test_np = y_test.values if isinstance(y_test, (pd.Series, pd.DataFrame)) else np.asarray(y_test)    
    X_test_tensor = torch.FloatTensor(X_test_np)
    y_test_tensor = torch.FloatTensor(y_test_np)
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_metrics = []
    fold_models = []
    
    print(f"\nStarting {k_folds}-Fold Stratified Cross Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_train_np), 1):
        print(f"\n{'='*50}")
        print(f"Fold {fold}/{k_folds}")
        print(f"{'='*50}")
        
        # Split data for this fold
        X_fold_train = torch.FloatTensor(X_train_np[train_idx])
        y_fold_train = torch.FloatTensor(y_train_np[train_idx])
        X_fold_val = torch.FloatTensor(X_train_np[val_idx])
        y_fold_val = y_train_np[val_idx]
        
        # Data loaders
        train_dataset = TensorDataset(X_fold_train, y_fold_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Initialize model for this fold
        model = GlucoseClassifier(X_train_np.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        warmup_epochs = int(epochs * warmpup_ratio)
        scheduler = linear_schedule_with_warmup(optimizer, warmup_epochs, epochs)
        
        model.train()
        train_losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            scheduler.step()

            if epoch % 20 == 0:
                print(f"Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}")

        # Evaluate on validation fold
        model.eval()
        with torch.no_grad():
            X_fold_val_tensor = X_fold_val.to(device)
            y_val_pred = model(X_fold_val_tensor).cpu().numpy()
        
        fold_metric = classification_metrics(y_fold_val, y_val_pred)
        fold_metrics.append(fold_metric)
        fold_models.append(model)
        
        print(f"\nFold {fold} Validation Metrics:")
        for metric, value in fold_metric.items():
            print(f"  {metric}: {value:.4f}")
    
    # Create DataFrame with CV metrics
    cv_data = []
    for i, metrics in enumerate(fold_metrics, 1):
        cv_data.append({
            'Fold': i,
            'Accuracy': metrics['Accuracy'],
            'AUC': metrics['ROC AUC'],
            'Recall': metrics['Recall'],
            'Prec.': metrics['Precision'],
            'F1': metrics['F1'],
            'Kappa': metrics['Kappa'],
            'MCC': metrics['MCC']
        })
    
    # Calculate mean and std
    metrics_df = pd.DataFrame(cv_data)
    mean_row = {
        'Fold': 'Mean',
        'Accuracy': metrics_df['Accuracy'].mean(),
        'AUC': metrics_df['AUC'].mean(),
        'Recall': metrics_df['Recall'].mean(),
        'Prec.': metrics_df['Prec.'].mean(),
        'F1': metrics_df['F1'].mean(),
        'Kappa': metrics_df['Kappa'].mean(),
        'MCC': metrics_df['MCC'].mean()
    }
    std_row = {
        'Fold': 'Std',
        'Accuracy': metrics_df['Accuracy'].std(),
        'AUC': metrics_df['AUC'].std(),
        'Recall': metrics_df['Recall'].std(),
        'Prec.': metrics_df['Prec.'].std(),
        'F1': metrics_df['F1'].std(),
        'Kappa': metrics_df['Kappa'].std(),
        'MCC': metrics_df['MCC'].std()
    }
    
    # Append mean and std rows
    cv_metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    print(f"\n{'='*50}")
    print("Cross-Validation Results:")
    print(f"{'='*50}")
    print(cv_metrics_df.to_string(index=False))
    
    # Train final model on full training set and evaluate on test set
    print(f"\n{'='*50}")
    print("Training Final Model on Full Training Set...")
    print(f"{'='*50}")
    
    X_train_tensor = torch.FloatTensor(X_train_np)
    y_train_tensor = torch.FloatTensor(y_train_np)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    final_model = GlucoseClassifier(X_train_np.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(final_model.parameters(), lr=0.001)
    warmup_epochs = int(epochs * warmpup_ratio)
    scheduler = linear_schedule_with_warmup(optimizer, warmup_epochs, epochs)
    
    final_model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = final_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        
        if epoch % 20 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}")
    
    # Final evaluation on test set
    final_model.eval()
    with torch.no_grad():
        X_test = X_test_tensor.to(device)
        y_pred = final_model(X_test).cpu().numpy()
    
    test_metrics = classification_metrics(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print("Test Set Metrics:")
    print(f"{'='*50}")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return cv_metrics_df, final_model
