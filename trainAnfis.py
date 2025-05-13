import torch 
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import GradScaler
from line_profiler import profile
import matplotlib.pyplot as plt
import os
import torch._dynamo
from collections import Counter

from anfisHelper import initialize_mfs_with_kmeans, initialize_mfs_with_fcm, set_rule_subset

@profile
def train_anfis_noHyb(model, X, Y, num_epochs, lr, dataloader):
    device = torch.device("cpu")
    model.to(device)
    
    initialize_mfs_with_kmeans(model, X)  # X_train as np array
    #initialize_mfs_with_fcm(model, X)
    

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.3)
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
            loss     = ce_loss + lb_loss
            #print(lb_loss)
                      
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.widths.data.clamp_(min=0.2, max=0.8)
            model.centers.data.clamp_(min=0, max=1)

        scheduler.step()

        epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
       
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


@profile
def train_anfis_hybrid(model, X, Y, num_epochs, lr):
    device = torch.device("cpu")  
    model.to(device)
    
    initialize_mfs_with_kmeans(model, X)
    #initialize_mfs_with_fcm(model, X)
    model.widths.data *= 2.0                      # grob verdoppeln
    model.widths.data.clamp_(min=0.06, max=0.6) 
    
    model.train()
    scaler = GradScaler()
    torch._dynamo.config.suppress_errors = True
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    torch.set_num_threads(os.cpu_count())
    
    # with torch.no_grad():
    #     k = min(2000, model.all_rules.size(0))
    #     X_sample = X[:5000]                 # zufällige Stichprobe aus den Trainingsdaten
    #     fs       = model._forward_mf_only(X_sample)  # [N, R]: Firing-Stärken aller Regeln
    #     coverage = fs.sum(0)                        # [R]: Gesamt-Firing pro Regel über alle Samples
    #     top_idx  = torch.topk(coverage, k).indices
    #     set_rule_subset(model, top_idx)
        
    
    # freq = Counter(Y.tolist())
    # print(freq)
    # tot  = len(Y)
    # weights = torch.tensor([tot/freq[c] for c in range(model.num_classes)],
    #                     dtype=torch.float32, device=device)
    #criterion = nn.CrossEntropyLoss(weight=weights) # +1%
    
    criterion = nn.CrossEntropyLoss()

    # Only optimize membership function params with an optimizer
    optimizer = optim.Adam([model.centers, model.widths], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.3)

    trainset = torch.utils.data.TensorDataset(X, Y)
    dataloader = DataLoader(trainset, batch_size=1024, num_workers=0, shuffle=True)
    N, P = X.shape
    k = int(N * 1)
    
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        with torch.no_grad():
            indices = torch.randperm(N)[:k]
            
            outputs, firing_strengths, x_ext = model(X[indices])
            Y_onehot = F.one_hot(Y[indices], num_classes=model.num_classes).float()
            
            model.update_consequents(
                    firing_strengths.detach(), 
                    x_ext.detach(), 
                    Y_onehot
                )
        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad(set_to_none=True)
            outputs, firing_strengths, x_ext = model(batch_X)
            loss = criterion(outputs, batch_Y)         
            loss.backward()
            optimizer.step()
            scaler.update()
            model.widths.data.clamp_(min=0.2, max=0.8)
            model.centers.data.clamp_(min=0, max=1)
            
        scheduler.step()
        

        epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if (epoch) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
            