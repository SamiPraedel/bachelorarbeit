from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch, torch.nn as nn, torch.optim as optim
from PopFnn import POPFNN
from torch.cuda.amp import autocast, GradScaler
from data_utils import load_iris_data, load_heart_data, load_wine_data, load_abalon_data, load_K_chess_data, load_Kp_chess_data, load_K_chess_data_splitted, load_K_chess_data_OneHot, load_Poker_data, load_Kp_chess_data_ord
from anfisHelper import initialize_mfs_with_kmeans, initialize_mfs_with_fcm, set_rule_subset

def train_popfnn(model, Xtr, ytr, epochs=5, lr=1e-3):
    initialize_mfs_with_kmeans(model, Xtr)
    #initialize_mfs_with_fcm(model, Xtr)
    model.pop_init(Xtr, ytr)              

    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr, Ytr),
        batch_size=1024, shuffle=True, pin_memory=False)

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
            #print(model.R)

    return model         




if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    #Xtr, Ytr, Xte, Yte = load_iris_data()
    #Xtr, Ytr, Xte, Yte = load_Kp_chess_data_ord()
    #Xtr, Ytr, Xte, Yte = load_Poker_data()
    Xtr, Ytr, Xte, Yte = load_K_chess_data_splitted()
    #Xtr, Ytr, Xte, Yte,_ = load_abalon_data()
    
    
    Xtr = Xtr.to(device)     
    Ytr = Ytr.to(device)  
    Xte = Xte.to(device)  
    Yte = Yte.to(device)


    net  = POPFNN(d=Xtr.shape[1], C=torch.unique(Ytr).shape[0], num_mfs=3).to(device)        
    net  = train_popfnn(net, Xtr, Ytr, epochs=1000) 

    net.eval()                             
    with torch.no_grad():
        preds = net(Xte).argmax(dim=1)

    acc = (preds == Yte).float().mean().item()
    print(f"Test-Accuracy auf Iris: {acc*100:.2f}%")


    