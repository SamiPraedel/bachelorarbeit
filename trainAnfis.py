import torch 
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import GradScaler
from line_profiler import profile
import matplotlib.pyplot as plt
import os

from anfisHelper import initialize_mfs_with_kmeans

@profile
def train_anfis_noHyb(model, X, Y, num_epochs, lr, dataloader):
    device = torch.device("cpu")
    model.to(device)
    
    initialize_mfs_with_kmeans(model, X)  # X_train as np array

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    trainset = torch.utils.data.TensorDataset(X, Y)
    #dataloader = dataloader
    dataloader = DataLoader(trainset, batch_size=32, shuffle=True)
    
    losses = []
    for epoch in range(1, num_epochs + 1):

        epoch_loss = 0.0
        for batch_X, batch_Y in dataloader:
            outputs, firing_strengths, mask = model(batch_X)  # outputs: [batch_size, num_classes]
            ce_loss = criterion(outputs, batch_Y)
            lb_loss  = model.load_balance_loss(firing_strengths, mask)
            loss     = ce_loss #+ lb_loss
            #print(lb_loss)
                      
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.widths.data.clamp_(min=1e-3, max=1)
            model.centers.data.clamp_(min=0, max=1)

        epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
       
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
            print("Final centers:", model.centers)
            print("Final widths:", model.widths)


    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.title('Trainingskurve (Loss über Epochen)')
    plt.grid(True)
    #plt.show()

@profile
def train_anfis_hybrid(model, X, Y, num_epochs, lr):
    # device = torch.device("cpu")  
    # model.to(device)
    
    initialize_mfs_with_kmeans(model, X)
    
    model.train()
    scaler = GradScaler()
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    torch.set_num_threads(os.cpu_count())

    # Only optimize membership function params with an optimizer
    optimizer = optim.Adam([model.centers, model.widths], lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    trainset = torch.utils.data.TensorDataset(X, Y)
    dataloader = DataLoader(trainset, batch_size=512, num_workers=0, shuffle=True)
    N, P = X.shape
    k = int(N * 0.1)
    
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad(set_to_none=True)
            outputs, firing_strengths, x_ext = model(batch_X)
            loss = criterion(outputs, batch_Y)         
            loss.backward()
            optimizer.step()
            scaler.update()
            model.widths.data.clamp_(min=1e-3, max=1)
            model.centers.data.clamp_(min=0, max=1)
            
            
        if (epoch) % 10 == 0: 
            
            with torch.no_grad():
                indices = torch.randperm(N)[:k]
                
                outputs, firing_strengths, x_ext = model(X[indices])
                Y_onehot = F.one_hot(Y[indices], num_classes=model.num_classes).float()
                
                model.update_consequents(
                        firing_strengths.detach(), 
                        x_ext.detach(), 
                        Y_onehot
                    )
        epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if (epoch) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    # Plot losses
    # plt.figure()
    # plt.plot(range(1, num_epochs+1), losses, marker='o', linestyle='-')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Curve')
    # plt.grid(True)
    # plt.show()