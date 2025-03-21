# mlp_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnected(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, num_classes):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden1)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(hidden1, hidden2)
        self.act2 = nn.Softmax()
        self.l3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = self.act1(self.l1(x))
        x = self.act2(self.l2(x))
        x = self.l3(x)  # => [batch_size, num_classes]
        return x

def fit_mlp(model, X_train, y_train, epochs=150, lr=0.001):
    """
    Einfaches Training für MLP.
    X_train: Tensor [N, input_dim], y_train: Tensor [N]
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

