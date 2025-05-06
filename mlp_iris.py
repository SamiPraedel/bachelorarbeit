import torch
import torch.nn as nn
from sklearn import metrics

from data_utils import load_iris_data, load_abalon_data, load_heart_data, load_Kp_chess_data, load_K_chess_data_splitted


class FullyConnected(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, hidden3, num_classes):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden1)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(hidden1, hidden2)
        self.act2 = nn.ReLU()
        self.l3 = nn.Linear(hidden2, hidden3)
        self.act3 = nn.Softmax()
        self.l4 = nn.Linear(hidden3, num_classes)

    def forward(self, x):
        x = self.act1(self.l1(x))
        x = self.act2(self.l2(x))
        x = self.act3(self.l3(x))  # => [batch_size, num_classes]
        x = self.l4(x)
        return x

def fit_mlp(model, X_train, y_train, epochs=1000, lr=0.001):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    #X_train, y_train, X_test, y_test = load_iris_data()
    #X_train, y_train, X_test, y_test,_ = load_abalon_data()
    #X_train, y_train, X_test, y_test = load_heart_data()
    #X_train, y_train, X_test, y_test,_ = load_Kp_chess_data()
    X_train, y_train, X_test, y_test = load_K_chess_data_splitted()
    
    number_classes = torch.unique(y_train).size(0)
    model = FullyConnected(X_train.size(1), 512, 256, 128, number_classes)
    fit_mlp(model, X_train, y_train)
    model.eval()
    outputs = model(X_test) 
    preds = torch.argmax(outputs, dim=1)
    preds_np   = preds.detach().cpu().numpy()
    y_test_np  = y_test.detach().cpu().numpy()
    accuracy = metrics.accuracy_score(y_test_np, preds_np) * 100
    print(accuracy)