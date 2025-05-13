from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch, torch.nn as nn, torch.optim as optim
from PopFnn import POPFNN
from data_utils import load_iris_data



def train_popfnn(model, Xtr, ytr, epochs=200, lr=1e-3):
    model = POPFNN(d=4, C=3, M=3)
    model.pop_init(Xtr, ytr)                     # 1-Pass
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(200):
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward(); opt.step()


if __name__ == "__main__":
    
    Xtr,Ytr,Xte,Yte = load_iris_data()
    a = Xtr.shape[1]
    print(a)
    net = POPFNN(4, 3)
    train_popfnn(net, Xtr, Ytr)
    logits = net(Xte)               # [N_test, C]
    preds  = torch.argmax(logits, dim=1)

    # 5) Accuracy ausrechnen
    acc = (preds == Yte).float().mean().item()
    print(f"Test-Accuracy auf Iris: {acc*100:.2f}%")
    