from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F # Added import
from PopFnn import POPFNN
from torch.cuda.amp import autocast, GradScaler
from data_utils import load_iris_data, load_heart_data, load_wine_data, load_abalon_data, load_Kp_chess_data, load_K_chess_data_splitted, load_K_chess_data_OneHot, load_poker_data, load_Kp_chess_data_ord
from anfisHelper import initialize_mfs_with_kmeans, initialize_mfs_with_fcm, set_rule_subset


device = "cuda" if torch.cuda.is_available() else "cpu"

def train_popfnn(model, Xtr, ytr, epochs=200, lr=1e-3):
    initialize_mfs_with_kmeans(model, Xtr)
    #initialize_mfs_with_fcm(model, Xtr)
    model.pop_init(Xtr, ytr)              


    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr, ytr),
        batch_size=256, shuffle=True, pin_memory=False)
    
    scaler = GradScaler()
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = 0
    for epoch in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()

            with autocast():                  # FP16/FP32 mixed
                logits = model(xb)
                loss   = loss_fn(logits, yb)
            


            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
    
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            print(model.R)

    return model         

def train_popfnn_ssl(X_l, y_l, X_p, y_p, w_p,
                     input_dim, num_classes,
                     num_mfs=4, epochs=50, lr=5e-4, seed=42): # Added default seed
    X_all = torch.cat([X_l, X_p]).to(device)
    y_all = torch.cat([y_l, y_p]).to(device)
    w_all = torch.cat([torch.ones(len(y_l), device=device), w_p.to(device)])

    model = POPFNN(input_dim, num_classes, num_mfs=num_mfs).to(device) # seed is not typically a param for POPFNN constructor
    model.pop_init(X_l.to(device), y_l.to(device))  # POPFNN specific initialization

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_all)
        loss = (loss_fn(logits, y_all) * w_all).mean()
        loss.backward()
        opt.step()
    return model

import torch
from sklearn.metrics import silhouette_score
import numpy as np

# Make sure your POPFNN class definition is available in the scope
# from PopFnn import POPFNN # Or however you import it

def calculate_popfnn_silhouette(model: POPFNN, X_data_tensor: torch.Tensor):
    """
    Calculates the Silhouette Score for the clusters formed by POPFNN's predictions.

    The Silhouette Score measures how similar a sample is to its own predicted class
    (cohesion) compared to other classes (separation).

    Args:
        model (POPFNN): The trained POPFNN model.
        X_data_tensor (torch.Tensor): The input data (features) as a PyTorch tensor.
                                      This should be the data for which you want
                                      to evaluate cluster separation (e.g., X_test or X_train).

    Returns:
        float or None: The average Silhouette Score over all samples, or None if
                       it cannot be computed (e.g., if only one class is predicted
                       or an error occurs).
    """
    model.eval()  # Set the model to evaluation mode
    device = next(model.parameters()).device  # Get the model's device
    X_data_tensor = X_data_tensor.to(device)

    with torch.no_grad(): # Disable gradient calculations for inference
        # Assuming the model outputs logits for classification
        logits = model(X_data_tensor)
        predicted_labels_tensor = logits.argmax(dim=1)

    # Move data to CPU and convert to NumPy arrays for scikit-learn
    # Note: If X_data_tensor is very large, ensure you have enough CPU RAM.
    X_data_np = X_data_tensor.cpu().numpy()
    predicted_labels_np = predicted_labels_tensor.cpu().numpy()

    # The silhouette_score function requires at least 2 labels and at most n_samples-1 labels.
    n_labels = len(np.unique(predicted_labels_np))
    n_samples = X_data_np.shape[0]

    if n_samples <= 1: # Cannot compute for a single sample or empty data
        print("Silhouette score cannot be computed: Not enough samples.")
        return None

    if 2 <= n_labels < n_samples:
        try:
            score = silhouette_score(X_data_np, predicted_labels_np)
            return score
        except Exception as e:
            print(f"An error occurred while calculating silhouette score: {e}")
            return None
    else:
        print(f"Silhouette score cannot be computed. Number of unique predicted labels: {n_labels}. "
              f"Required: 2 <= n_labels < n_samples ({n_samples}).")
        return None





    