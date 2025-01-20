# data_utils.py

import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

def load_iris_data(test_size=0.2, random_state=42):
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_iris, y_iris, test_size=test_size, shuffle=True, random_state=random_state
    )
    # Optionale NaN-Behandlung
    X_train_np = np.nan_to_num(X_train_np, nan=0.0)
    X_test_np  = np.nan_to_num(X_test_np,  nan=0.0)

    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_np)
    X_test_np  = scaler.transform(X_test_np)

    # PyTorch Tensor
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test  = torch.tensor(X_test_np,  dtype=torch.float32)
    y_test  = torch.tensor(y_test_np,  dtype=torch.long)

    return X_train, y_train, X_test, y_test

def load_heart_data(test_size=0.2, random_state=42):
    heart = fetch_ucirepo(id=45)
    X_df = heart.data.features
    y_df = heart.data.targets.squeeze()

    # => Split
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_df, y_df, test_size=test_size, shuffle=True, random_state=random_state
    )
    # NaN -> 0
    X_train_np = np.nan_to_num(X_train_np, nan=0.0)
    X_test_np  = np.nan_to_num(X_test_np,  nan=0.0)

    # scale
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_np)
    X_test_np  = scaler.transform(X_test_np)

    # to torch
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np.to_numpy(), dtype=torch.long)
    X_test  = torch.tensor(X_test_np,  dtype=torch.float32)
    y_test  = torch.tensor(y_test_np.to_numpy(),  dtype=torch.long)

    return X_train, y_train, X_test, y_test

    