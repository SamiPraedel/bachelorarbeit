import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from line_profiler import profile
import os
from torch.cuda.amp import GradScaler   # für’s Skalieren der Gradienten


class HybridANFIS(nn.Module):
    def __init__(self, input_dim, num_classes, num_mfs, max_rules, seed):
        seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        super(HybridANFIS, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_mfs = num_mfs  
        self.num_rules = num_mfs ** input_dim 
        
        self.centers = nn.Parameter(torch.ones(input_dim, num_mfs))
        self.widths = nn.Parameter(torch.ones(input_dim, num_mfs)) 
        

        if self.num_rules <= max_rules:
            self.rules = torch.cartesian_prod(*[torch.arange(self.num_mfs) for _ in range(self.input_dim)])
        else:
            self.rules = torch.randint(low=0,
                    high=self.num_mfs,
                    size=(max_rules, self.input_dim))
            self.num_rules = max_rules
  
        self.consequents = nn.Parameter(torch.ones(self.num_rules, input_dim + 1, num_classes))
        self.consequents.requires_grad = False
        self.register_buffer("rule_idx", self.rules.t())  # shape [d, R]


    def gaussian_mf(self, x, center, width):
        #shape: batch_size, num_mfs
        gaus = torch.exp(-((x - center) ** 2) / (2 * width ** 2))
        return gaus

    
    def forward(self, x):
        device = x.device
        batch_size = x.size(0)

        x_exp      = x.unsqueeze(2)                      # [B, d, 1]
        centers    = self.centers.unsqueeze(0)           # [1, d, m]
        widths     = self.widths.unsqueeze(0)            # [1, d, m]
        mfs  = torch.exp(-((x_exp - centers) ** 2) /
                            (2 * widths ** 2) + 1e-9)
            
        mfs.to(device)        
        rules_expand = self.rule_idx.unsqueeze(0).expand(batch_size, -1, -1)  # [B, d, R]
        rule_mfs = torch.gather(mfs, 2, rules_expand)

        
        firing_strengths = rule_mfs.prod(dim=1)
        norm_fs = firing_strengths / (firing_strengths.sum(dim=1, keepdim=True) + 1e-9)
        
        # 1) Bias anfügen – x_ext braucht selbst keine Gradienten
        ones   = x.new_ones(x.size(0), 1)
        x_ext  = torch.cat([x, ones], dim=1)          # [B, d+1]

        # 2) Nur x_ext vom Autograd trennen
        x_ext_ng = x_ext.detach()                     # <‑‑ kein gradient hier

        # 3) Phi bauen (norm_fs hat weiterhin requires_grad=True!)
        phi = norm_fs.unsqueeze(2) * x_ext_ng.unsqueeze(1)   # [B, R, d+1]

        # 4) Flatten & MatMul (self.consequents hat requires_grad=False)
        B, R, I   = phi.shape
        phi_flat  = phi.view(B, R * I)
        beta_flat = self.consequents.view(R * I, self.num_classes)

        y_hat = phi_flat @ beta_flat                  # [B, C]  – grad_fn vorhanden!
                                # [B, C]

        return y_hat, norm_fs, x_ext
                
        # Shape: [batch_size, num_classes]
        #return rule_outputs, normalized_firing_strengths, x_ext
    

    def update_consequents(self, normalized_firing_strengths, x_ext, Y):
        """
        Update consequent parameters using Least Squares Estimation (LSE).
        :param normalized_firing_strengths: Normalized rule activations, shape: [batch_size, num_rules]
        :param x_ext: Extended inputs (with bias), shape: [batch_size, input_dim + 1]
        :param y: Target outputs (one-hot encoded), shape: [batch_size, num_classes]
        """

        batch_size = normalized_firing_strengths.size(0)

        Phi = normalized_firing_strengths.unsqueeze(2) * x_ext.unsqueeze(1)  # Shape: [batch_size, num_rules, input_dim + 1]

        Phi = Phi.view(batch_size, self.num_rules * (self.input_dim + 1))  # Flattened design matrix

        #B = torch.linalg.lstsq(Phi, Y).solution
        
        Phi_T_Phi = Phi.t() @ Phi
        Phi_T_Y   = Phi.t() @ Y
        lam = 1e-3
        I = torch.eye(Phi_T_Phi.size(0), device=Phi.device)
        B = torch.linalg.solve(Phi_T_Phi + lam * I, Phi_T_Y)

        # Reshape in die Form der consequent Parameter: [num_rules, input_dim+1, num_classes]
        self.consequents.data = B.view(self.num_rules, self.input_dim + 1, self.num_classes)
        
    def initialize_mfs_with_kmeans(self, data):
        """
        data: NumPy array of shape (num_samples, input_dim).
            Each column is one feature/dimension.

        Returns:
        centers: NumPy array of shape (input_dim, num_mfs)
        widths:  NumPy array of shape (input_dim, num_mfs)
        """
        
        input_dim = data.shape[1]
        num_mfs = self.num_mfs  # Anzahl der Membership Functions pro Dimension

        centers_list = []
        widths_list = []

        for i in range(input_dim):
            # Daten für Dimension i extrahieren:
            X_dim = data[:, i].reshape(-1, 1)  # Shape: [num_samples, 1]

            # K-Means auf diese Dimension anwenden:
            km = KMeans(n_clusters=num_mfs, random_state=0, n_init='auto')
            km.fit(X_dim)

            # Clusterzentren als 1D-Array extrahieren:
            centers_i = km.cluster_centers_.flatten()
            # Berechne die Standardabweichung pro Cluster:
            stds_i = []
            labels = km.labels_
            for c_idx in range(num_mfs):
                cluster_points = X_dim[labels == c_idx]
                # Falls der Datentyp nicht bereits ein NumPy-Array ist, umwandeln:
                if not isinstance(cluster_points, np.ndarray):
                    cluster_points = cluster_points.numpy()
                if len(cluster_points) > 1:
                    std_val = np.std(cluster_points)
                else:
                    std_val = 0.1  # Fallback für einen einzelnen Datenpunkt
                std_val = max(std_val, 0.1)  # Minimaler positiver Wert
                stds_i.append(std_val)

            # --- Hier folgt der Code zur Sortierung der Cluster ---
            # Bündele die Zentren und die stds zu Paaren:
            pairs = sorted(zip(centers_i, stds_i), key=lambda pair: pair[0])
            # Entpacke die sortierten Paare:
            centers_i, stds_i = zip(*pairs)
            centers_i = np.array(centers_i)
            stds_i = np.array(stds_i)
            # --- Ende Sortierung ---

            # Füge die sortierten Werte zur Gesamtliste hinzu:
            centers_list.append(centers_i)
            widths_list.append(stds_i)

        # Wandle die Listen in 2D-Arrays um:
        centers = np.array(centers_list, dtype=np.float32)  # Shape: (input_dim, num_mfs)
        widths  = np.array(widths_list,  dtype=np.float32)   # Shape: (input_dim, num_mfs)

        print(f"K-Means-based centers:\n{centers}")
        print(f"K-Means-based widths:\n{widths}")

        return centers, widths
        


@profile
def train_hybrid_anfis(model, X, Y, num_epochs, lr):
    # device = torch.device("cpu")  
    # model.to(device)
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