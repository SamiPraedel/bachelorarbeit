import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import umap
from create_plots import plot_firing_strength_heatmap, plot_firing_strengths, plot_sample_firing_strengths, plot_umap_fixed_rule
from line_profiler import profile


#-------------------
# Modell
#-------------------



class NoHybridANFIS(nn.Module):
    def __init__(self, input_dim, num_classes, num_mfs, max_rules, seed, zeroG):
        
       # num_mfs_tensor = torch.tensor[3,3,3,2,2,2]
        
        seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        super(NoHybridANFIS, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        #self.num_mfs = num_mfs_tensor.gather(dim = 0)
        self.num_mfs = num_mfs
        self.num_rules = num_mfs ** input_dim  # Total number of rules
        self.zeroG = zeroG

        # Membership function parameters (Gaussian)
        self.centers = nn.Parameter(torch.ones(input_dim, num_mfs))  # Centers
        #print(self.centers[0])
        self.widths = nn.Parameter(torch.ones(input_dim, num_mfs))  # Widths
        #self.widths = nn.Parameter()
        if self.num_rules <= max_rules:
            self.rules = torch.cartesian_prod(*[torch.arange(self.num_mfs) for _ in range(self.input_dim)])
        else:
            self.rules = torch.randint(low=0,
                    high=self.num_mfs,
                    size=(max_rules, self.input_dim))
            self.num_rules = max_rules

        #print(self.rules.shape)
        # full_rules.shape = [8192, 13]

        # max_rules
        # idx = torch.randperm(full_rules.size(0))[:max_rules]
        # self.rules = full_rules[idx]  # => shape [max_rules, input_dim]
        # self.num_rules = self.rules.size(0)  # =max_rules

        # Direkte Zufallserzeugung von max_rules Regeln
        # => self.rules wird [max_rules, input_dim] 
        #    mit Werten in [0, num_mfs-1].


        # Consequent parameters (initialized randomly)
        if zeroG:
            self.consequents = nn.Parameter(torch.rand(self.num_rules, num_classes))
        else:
            self.consequents = nn.Parameter(torch.rand(self.num_rules, input_dim + 1, num_classes))
        



    def gaussian_mf(self, x, center, width):
        gaus = torch.exp(-((x - center) ** 2) / (2 * width ** 2))
        return gaus
    
    def triangular_mf(self, x, center, width, eps=1e-9):

        left  = (x - (center - width)) / (width + eps)      # slope up
        right = ((center + width) - x) / (width + eps)      # slope down

        # The membership is the minimum of these two slopes, clipped to [0,1].
        tri = torch.min(left, right)
        tri = torch.clamp(tri, min=0.0, max=1.0)
        return tri
    
    


    def forward(self, x):
        batch_size = x.size(0)
        mfs = []
        for i in range(self.input_dim):
            x_i = x[:, i].unsqueeze(1)  # Shape: [batch_size, 1]
            center_i = self.centers[i]  # Shape: [num_mfs]
            width_i = self.widths[i]    # Shape: [num_mfs]
            mf_i = self.gaussian_mf(x_i, center_i, width_i)  # Shape: [batch_size, num_mfs]
            mfs.append(mf_i)

        mfs = torch.stack(mfs, dim=1)  # Shape: [batch_size, input_dim, num_mfs]

        # rules.shape => [num_rules, input_dim]
        rules_idx = self.rules.unsqueeze(0).expand(batch_size, -1, -1).permute(0, 2, 1)
        # rules_idx.shape => [batch_size, input_dim, num_rules]

        rule_mfs = torch.gather(mfs, dim=2, index=rules_idx)  #rule_mfs => [batch_size, input_dim, num_rules]

        fiering_strengths = torch.prod(rule_mfs, dim=1)  #[batch_size, num_rules]        
        
        topk_p = 0.1

        K = max(1, int(topk_p * self.num_rules))
        vals, idx = torch.topk(fiering_strengths, k=K, dim=1)
        mask = torch.zeros_like(fiering_strengths).scatter_(1, idx, 1.)
        firing = fiering_strengths * mask
        normalized_firing_strengths = firing / (firing.sum(1, keepdim=True)+1e-9)
        
        #normalized_firing_strengths = fiering_strengths / (fiering_strengths.sum(dim=1, keepdim=True) + eps) 

        x_ext = torch.cat([x, torch.ones(batch_size, 1)], dim=1)  # Add bias term, shape: [batch_size, input_dim + 1]

        # Schritt 1: Berechne die Regel-MF-Werte [B, R, C]
        if self.zeroG:
            outputs = self.consequents
            rule_outputs = torch.einsum('br,rc->bc', normalized_firing_strengths, outputs)
        else:
            rule_mfs = torch.einsum('bi,rjc->brc', x_ext, self.consequents)  # [B, R, C]
            # Schritt 2: Gewichtete Summe der Regel-MF-Werte [B, C]
            rule_outputs = torch.einsum('br, brc->bc', normalized_firing_strengths, rule_mfs)  # [B, C]        
        
        # Shape: [batch_size, num_classes]
        return rule_outputs, normalized_firing_strengths, mask

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


    def load_balance_loss(self, router_probs, mask, alpha=0.01):
        """
        router_probs : [B, R]   = p_i(x)
        mask         : [B, R]   = 0/1 indicator (top‑K selection)
        """
        T, R = router_probs.shape
        # fraction of *tokens* per rule  f_i
        f = mask.float().mean(0)                 # [R]
        # fraction of *probability mass* per rule P_i
        P = router_probs.mean(0)                 # [R]
        lb = alpha * R * (f * P).sum()
        return lb


    
@profile
def train_anfis(model, X, Y, num_epochs, lr, dataloader):
    device = torch.device("cpu")  # Oder "cuda" wenn GPU verfügbar ist
    model.to(device)
    model.train()
    # Optimiert nur MF-Parameter (centers, widths)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    
    trainset = torch.utils.data.TensorDataset(X, Y)
    #dataloader = dataloader
    dataloader = DataLoader(trainset, batch_size=32, shuffle=True)
    


    losses = []
    for epoch in range(1, num_epochs + 1):

        
        # # 1) Forward Pass
        # outputs, firing_strengths, x_ext = model(X)
        # loss = criterion(outputs, Y)


        # # 2) Backprop auf MF-Parameter
        # optimizer.zero_grad()
        
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_([model.centers, model.widths], max_norm=1.0)

        # optimizer.step()
        epoch_loss = 0.0
        for batch_X, batch_Y in dataloader:
            # Forward Pass
            outputs, firing_strengths, mask = model(batch_X)  # outputs: [batch_size, num_classes]
            #print(outputs)
            ce_loss = criterion(outputs, batch_Y)
            lb_loss  = model.load_balance_loss(firing_strengths, mask)
            loss     = ce_loss + lb_loss
                      

            # Backpropagation auf MF-Parameter
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.widths.data.clamp_(min=1e-3, max=1)
            model.centers.data.clamp_(min=0, max=1)
            # accuracy = (outputs == batch_Y).float().mean().item() * 100
            # print(accuracy)


        


        epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
       




        # Optional: Ausgeben
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
            print(firing_strengths.shape)
            print("Final centers:", model.centers)
            print("Final widths:", model.widths)


    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.title('Trainingskurve (Loss über Epochen)')
    plt.grid(True)
    #plt.show()
    
    







