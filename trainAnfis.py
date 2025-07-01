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

# Assuming these files are in the Python path or same directory
from anfis_hybrid import HybridANFIS
from anfis_nonHyb import NoHybridANFIS
from anfisHelper import initialize_mfs_with_kmeans, initialize_mfs_with_fcm, set_rule_subset, create_weighted_sampler # Removed incomplete import
from visualizers import plot_epoch_curves # Added specific import for plotting

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@profile
def train_anfis_noHyb(
    model: NoHybridANFIS, 
    X: torch.Tensor, 
    Y: torch.Tensor, 
    num_epochs: int, 
    lr: float,
    X_val: torch.Tensor = None, 
    y_val: torch.Tensor = None
):
    model.to(device)
    
    initialize_mfs_with_kmeans(model, X)  # X_train as np array
    #initialize_mfs_with_fcm(model, X)

    model.widths.data *= 1.

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.3)
    criterion = nn.CrossEntropyLoss()
    
    #dataloader = create_weighted_sampler(X, Y, batch_size=1024)
    trainset = torch.utils.data.TensorDataset(X, Y)
    dataloader = DataLoader(trainset, batch_size=128, num_workers=0, shuffle=True)
    
    # Use the dataloader passed as an argument
    # Initialize history dictionary to store metrics
    history = {"train_loss": [], "train_acc": []}
    if X_val is not None and y_val is not None:
        history["val_loss"] = []
        history["val_acc"] = []
        X_val, y_val = X_val.to(device), y_val.to(device)

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_X, batch_Y in dataloader:
            model.train() # Ensure model is in training mode
            batch_X = batch_X.to(device, non_blocking=True)
            batch_Y = batch_Y.to(device, non_blocking=True)
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

            # Accumulate loss for the epoch (inside batch loop)
            epoch_loss += loss.item() # Use .item() here as loss is a tensor
            
            # Training accuracy
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += batch_Y.size(0)
            correct_train += (predicted_train == batch_Y).sum().item()

        # --- Code moved outside the batch loop ---
        #scheduler.step() # If uncommented, should be here

        # Calculate average loss for the epoch (outside batch loop)
        avg_loss = epoch_loss / len(dataloader)
        history["train_loss"].append(avg_loss)
        train_accuracy = 100 * correct_train / total_train
        history["train_acc"].append(train_accuracy)

        # Validation step
        if X_val is not None and y_val is not None:
            model.eval() # Set model to evaluation mode
            with torch.no_grad():
                val_outputs, _, _ = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                history["val_loss"].append(val_loss.item())

                _, predicted_val = torch.max(val_outputs.data, 1)
                total_val = y_val.size(0)
                correct_val = (predicted_val == y_val).sum().item()
                val_accuracy = 100 * correct_val / total_val
                history["val_acc"].append(val_accuracy)
            model.train() # Set back to training mode

        # Print epoch progress (outside batch loop)
        if epoch % 10 == 0: # Use epoch directly for 1-based indexing check
            log_msg = f"Epoch [{epoch}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%"
            if X_val is not None and y_val is not None:
                log_msg += f", Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy:.2f}%"
            print(log_msg)

    # Optional: Rule pruning example (ensure 'cov' is defined if uncommented)
    # cov_thr = 0.002
    # cov_rel = cov / cov.sum()              # cov aus rule_stats
    # keep    = cov_rel > cov_thr

    # model.rules        = model.rules[keep]
    # model.consequents  = nn.Parameter(model.consequents[keep])
    # model.num_rules    = keep.sum().item()
    
    # ----- Plot training loss after training -----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plot_epoch_curves(history, "loss", ax=axes[0], smooth=3)
    axes[0].set_title(f"Loss Curves for {model.__class__.__name__} (NoHybrid)")
    plot_epoch_curves(history, "acc", ax=axes[1], smooth=3)
    axes[1].set_title(f"Accuracy Curves for {model.__class__.__name__} (NoHybrid)")
    plt.tight_layout()
    
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    plot_save_path = os.path.join(viz_dir, f"{model.__class__.__name__}_NoHybrid_training_curves.png")
    plt.savefig(plot_save_path)
    plt.close()
    print(f"Training curves saved to {plot_save_path}")



@profile
def train_anfis_hybrid(
    model: HybridANFIS, 
    X: torch.Tensor, 
    Y: torch.Tensor, 
    num_epochs: int, 
    lr: float, # Note: lr is used for optimizer for MFs
    X_val: torch.Tensor = None, 
    y_val: torch.Tensor = None
):
    model.to(device)
    
    initialize_mfs_with_kmeans(model, X)
    #initialize_mfs_with_fcm(model, X)
    # model.widths.data *= 1.8                    # grob verdoppeln
    # model.widths.data.clamp_(min=0.06, max=0.9) 
    
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
    dataloader = DataLoader(trainset, batch_size=128, num_workers=0, shuffle=True)
    N, P = X.shape
    k = int(N * 0.6)
    
    # Initialize history dictionary to store metrics
    history = {"train_loss": [], "train_acc": []}
    if X_val is not None and y_val is not None:
        history["val_loss"] = []
        history["val_acc"] = []
        X_val, y_val = X_val.to(device), y_val.to(device)

    for epoch in range(num_epochs):
        model.train() # Ensure model is in training mode
        epoch_loss = 0.0
        correct_train = 0
        total_train = 0
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
            model.train() # Ensure model is in training mode for MF optimization
            batch_X = batch_X.to(device, non_blocking=True)
            batch_Y = batch_Y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs, firing_strengths, x_ext = model(batch_X)
            loss = criterion(outputs, batch_Y)         
            loss.backward()
            optimizer.step()

            model.widths.data.clamp_(min=0.2, max=0.8)
            model.centers.data.clamp_(min=0, max=1)
            
            epoch_loss += loss.item()
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += batch_Y.size(0)
            correct_train += (predicted_train == batch_Y).sum().item()
            
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        history["train_loss"].append(avg_loss)
        train_accuracy = 100 * correct_train / total_train
        history["train_acc"].append(train_accuracy)

        # Validation step
        if X_val is not None and y_val is not None:
            model.eval() # Set model to evaluation mode
            with torch.no_grad():
                val_outputs, _, _ = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                history["val_loss"].append(val_loss.item())

                _, predicted_val = torch.max(val_outputs.data, 1)
                total_val = y_val.size(0)
                correct_val = (predicted_val == y_val).sum().item()
                val_accuracy = 100 * correct_val / total_val
                history["val_acc"].append(val_accuracy)
            model.train() # Set back to training mode for LSE and MF opt in next epoch

        if (epoch) % 10 == 0:
            log_msg = f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.6f}, Train Acc: {train_accuracy:.2f}%"
            if X_val is not None and y_val is not None:
                log_msg += f", Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy:.2f}%"
            print(log_msg)

    # ----- Plot training loss after training -----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plot_epoch_curves(history, "loss", ax=axes[0], smooth=3)
    axes[0].set_title(f"Loss Curves for {model.__class__.__name__} (Hybrid)")
    plot_epoch_curves(history, "acc", ax=axes[1], smooth=3)
    axes[1].set_title(f"Accuracy Curves for {model.__class__.__name__} (Hybrid)")
    plt.tight_layout()
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    plot_save_path = os.path.join(viz_dir, f"{model.__class__.__name__}_Hybrid_training_curves.png")
    plt.savefig(plot_save_path)
    plt.close()
    print(f"Training curves saved to {plot_save_path}")

def train_hybrid_anfis_ssl(X_l, y_l, X_p, y_p, w_p,
                           input_dim, num_classes,
                           num_mfs=4, max_rules=1000,
                           epochs=50, lr=5e-3, seed=42): # Added default seed for consistency
    X_all = torch.cat([X_l, X_p]).to(device)
    y_all = torch.cat([y_l, y_p]).to(device)
    w_all = torch.cat([torch.ones(len(y_l), device=device), w_p.to(device)])
    y_onehot = F.one_hot(y_all, num_classes).float()

    model = HybridANFIS(input_dim, num_classes, num_mfs, max_rules, seed=seed).to(device)
    opt = torch.optim.Adam([
        {'params': model.centers, 'lr': lr},
        {'params': model.widths, 'lr': lr},
    ])

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits, norm_fs, x_ext = model(X_all)
        loss = (w_all * F.cross_entropy(logits, y_all, reduction='none')).mean()
        loss.backward()
        opt.step()
        model.update_consequents(norm_fs.detach(), x_ext.detach(), y_onehot)
    return model

def train_nohybrid_anfis_ssl(X_l, y_l, X_p, y_p, w_p,
                             input_dim, num_classes,
                             num_mfs=7, max_rules=2000, zeroG=False,
                             epochs=100, lr=5e-3, seed=42): # Added default seed
    X_all = torch.cat([X_l, X_p]).to(device)
    y_all = torch.cat([y_l, y_p]).to(device)
    # Original NoHybrid pipeline squared the pseudo-label weights
    w_all = torch.cat([torch.ones(len(y_l), device=device), w_p.to(device)**2])

    model = NoHybridANFIS(input_dim, num_classes, num_mfs, max_rules, seed=seed, zeroG=zeroG).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits, norm_fs, mask = model(X_all)
        loss_main = (w_all * ce_loss_fn(logits, y_all)).mean()
        loss_aux = model.load_balance_loss(norm_fs.detach(), mask) # Assumes alpha is handled in model
        loss = loss_main + loss_aux
        loss.backward()
        opt.step()
    return model

