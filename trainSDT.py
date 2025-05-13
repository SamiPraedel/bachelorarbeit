# run_sdt.py
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, pandas as pd, random
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

from fuzzyDecisionTree import SoftDecisionTree
from data_utils import load_K_chess_data_splitted, load_Kp_chess_data, load_iris_data, load_abalon_data, load_heart_data
from sklearn.preprocessing import LabelEncoder
from data.openD import loadK
torch.manual_seed(0);  random.seed(0);  np.random.seed(0)

# ----------------------------------------------------------
def load_chessKR(onehot=True):
    #dataset  = fetch_ucirepo(id=22)           # King-Rook vs King+Pawn
    dataset = loadK()
    
    dataset.rename(columns={
        'V1': 'white_king_file',
        'V2': 'white_rook_file',
        'V3': 'black_king_file'
    }, inplace=True)
    
    lbl_white_king=LabelEncoder()
    dataset['white_king_file']=lbl_white_king.fit_transform(dataset['white_king_file'])
    lbl_white_rook=LabelEncoder()
    dataset['white_rook_file']=lbl_white_rook.fit_transform(dataset['white_rook_file'])
    lbl_black_king=LabelEncoder()
    dataset['black_king_file']=lbl_black_king.fit_transform(dataset['black_king_file'])
    lbl_result=LabelEncoder()
    dataset['Class']=lbl_result.fit_transform(dataset['Class'])
    
    # Using data.iloc[:,:-1].values we get the feature variables
    x=dataset.iloc[:,:-1].values

    # Using data.iloc[:,-1].values we get the target variable
    y=dataset.iloc[:,-1].values
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=100)
    
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test  = torch.tensor(x_test, dtype=torch.float32)
    y_test  = torch.tensor(y_test, dtype=torch.long)
    
    return x_train, y_train, x_test, y_test

# ----------------------------------------------------------
def train_sdt(depth=6, lr=3e-4, epochs=200):
    #Xtr,ytr,Xte,yte = load_chessKR(onehot=True)
    #Xtr,ytr,Xte,yte = load_K_chess_data_splitted()
    Xtr,ytr,Xte,yte = load_Kp_chess_data()
    #Xtr,ytr,Xte,yte = load_iris_data()
    #Xtr,ytr,Xte,yte = load_heart_data()


    n_feat = Xtr.size(1); n_class = int(ytr.max())+1
    model  = SoftDecisionTree(n_feat, n_class, depth)
    opt  = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    tau, tau_min, decay = 2.0, 0.3, 0.992

    for ep in range(1,epochs+1):
        model.train(); opt.zero_grad()
        logits = model(Xtr, tau)
        loss = loss_fn(logits, ytr)
        loss.backward(); opt.step()

        # -------- Eval ----------
        if ep%50==0 or ep==epochs:
            model.eval()
            with torch.no_grad():
                acc = (model(Xte, tau=0.3).argmax(1)==yte).float().mean()*100
            print(f"Ep{ep:<4} loss {loss.item():.4f}  acc {acc:.2f}%  τ={tau:.2f}")
        tau = max(tau_min, tau*decay)
    return model

if __name__ == "__main__":
    train_sdt()
