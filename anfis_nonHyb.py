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


#-------------------
# Modell
#-------------------



class NoHybridANFIS(nn.Module):
    def __init__(self, input_dim, num_classes, num_mfs, max_rules, seed, zeroG):


        seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        super(NoHybridANFIS, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
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

        # Step 1: Compute membership values
        mfs = []
        for i in range(self.input_dim):
            x_i = x[:, i].unsqueeze(1)  # Shape: [batch_size, 1]
            center_i = self.centers[i]  # Shape: [num_mfs]
            width_i = self.widths[i]    # Shape: [num_mfs]
            mf_i = self.gaussian_mf(x_i, center_i, width_i)  # Shape: [batch_size, num_mfs]
            #mf_i = self.triangular_mf(x_i, center_i, width_i)
            #print(mf_i)
            mfs.append(mf_i)

        mfs = torch.stack(mfs, dim=1)  # Shape: [batch_size, input_dim, num_mfs]

        # Step 2: Compute rule activations
        #full_rules = torch.cartesian_prod(*[torch.arange(self.num_mfs) for _ in range(self.input_dim)])

        # rules.shape => [num_rules, input_dim]

        rules_idx = self.rules.unsqueeze(0).expand(batch_size, -1, -1).permute(0, 2, 1)
        

        # rules_idx.shape => [batch_size, input_dim, num_rules]

        # Now gather along dim=2 in 'mfs'
        # mfs.shape => [batch_size, input_dim, num_mfs]
        rule_mfs = torch.gather(mfs, dim=2, index=rules_idx)
        
        # rule_mfs.shape => [batch_size, input_dim, num_rules]

        # Multiply membership values across input_dim
        fiering_strengths = torch.prod(rule_mfs, dim=1)
        
 

        # shape => [batch_size, num_rules]


        # Step 3: Normalize rule activations
        eps = 1e-9
        
        normalized_firing_strengths = fiering_strengths / (fiering_strengths.sum(dim=1, keepdim=True) + eps)
        #normalized_firing_strengths = fiering_strengths + eps
        
        
        # soft= nn.Softmax(dim=1)
        # normalized_firing_strengths = soft(fiering_strengths)

        # Step 4: Compute rule outputs
        x_ext = torch.cat([x, torch.ones(batch_size, 1)], dim=1)  # Add bias term, shape: [batch_size, input_dim + 1]
        # rule_outputs = torch.einsum('br,brc->bc', normalized_firing_strengths, 
        #                              torch.einsum('bi,rjc->brc', x_ext, self.consequents))

        # Schritt 1: Berechne die Regel-MF-Werte [B, R, C]
        if self.zeroG:
            outputs = self.consequents
            rule_outputs = torch.einsum('br,rc->bc', normalized_firing_strengths, outputs)
        else:
            rule_mfs = torch.einsum('bi,rjc->brc', x_ext, self.consequents)  # [B, R, C]
            # Schritt 2: Gewichtete Summe der Regel-MF-Werte [B, C]
            rule_outputs = torch.einsum('br, brc->bc', normalized_firing_strengths, rule_mfs)  # [B, C]
        
        # print("absoluts -----", fiering_strengths)
        # print(normalized_firing_strengths)
        
        
        
        # Shape: [batch_size, num_classes]
        return rule_outputs, normalized_firing_strengths, x_ext

    def initialize_mfs_with_kmeans(self, data):
        """
        data: NumPy array of shape (num_samples, input_dim).
            Each column is one feature/dimension.

        Returns:
        centers: NumPy array of shape (input_dim, num_mfs)
        widths:  NumPy array of shape (input_dim, num_mfs)
        """
        
        input_dim = data.shape[1]
        # Decide how many membership functions (clusters) you want per dimension:
        # (You can override 'self.num_mfs' or pass it as a parameter if you like.)
        num_mfs = self.num_mfs

        centers_list = []
        widths_list = []

        for i in range(input_dim):
            # Extract the data for dimension i:
            X_dim = data[:, i].reshape(-1, 1)  # shape: [num_samples, 1]

            # Run K-Means with 'num_mfs' clusters for this dimension:
            km = KMeans(n_clusters=num_mfs, random_state=0, n_init='auto')
            km.fit(X_dim)


            # Sort cluster centers to have a consistent left-to-right ordering:
            # cluster_centers_ is shape [num_mfs, 1]
            centers_i = km.cluster_centers_.flatten()
            print(type(centers_i))
            # Compute std for each cluster to represent the Gaussian width:
            # This is a simple approach: compute standard deviation of points
            # belonging to that cluster. If a cluster is too small, fallback to a small default width:
            stds_i = []
            labels = km.labels_
            for c_idx in range(num_mfs):
                cluster_points = X_dim[labels == c_idx]
                if not (isinstance(cluster_points, np.ndarray)):
                    cluster_points = cluster_points.numpy()
                
                print(type(cluster_points))
                if len(cluster_points) > 1:
                    std_val = np.std(cluster_points)
                else:
                    # If only 1 data point in cluster, pick a small default
                    std_val = 0.1
                std_val = max(std_val, 0.2)  # clamp at some minimal positive
                stds_i.append(std_val)

            # Optionally, reorder stds_i to match the sorted order of centers_i
            # Because after sorting centers_i, we should also reorder the stds accordingly.
            # One approach is to sort by the cluster center:
            # pairs = sorted(zip(centers_i, stds_i), key=lambda p: p[0])
            # centers_i, stds_i = zip(*pairs)
            # centers_i = np.array(centers_i)
            # stds_i    = np.array(stds_i)

            centers_list.append(centers_i)
            widths_list.append(stds_i)

        centers = np.array(centers_list, dtype=np.float32)  # shape => (input_dim, num_mfs)
        widths  = np.array(widths_list,  dtype=np.float32)  # shape => (input_dim, num_mfs)

        print(f"K-Means-based centers:\n{centers}")
        print(f"K-Means-based widths:\n{widths}")

        return centers, widths

    

def train_anfis(model, X, Y, num_epochs, lr, dataloader):
    device = torch.device("cpu")  # Oder "cuda" wenn GPU verfügbar ist
    model.to(device)
    model.train()
    # Optimiert nur MF-Parameter (centers, widths)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
            outputs, firing_strengths, x_ext = model(batch_X)  # outputs: [batch_size, num_classes]
            #print(outputs)
            loss = criterion(outputs, batch_Y)
                      

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
    
    


def plot_firing_strengths(model, X, cmap='viridis'):

    model.eval()
    with torch.no_grad():
        _, norm_fs, _ = model(X)
    
    # norm_fs hat Shape [N, num_rules]
    norm_fs_np = norm_fs.cpu().numpy()
    
    # Wähle für jeden Datenpunkt z. B. den maximalen Firing Strength-Wert als Farbe
    colors = norm_fs_np.max(axis=1)
    
    # UMAP-Anwendung: Reduziere die Dimension von norm_fs auf 2
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(norm_fs_np)
    
    # Plot erstellen
    plt.figure(figsize=(10,8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=cmap, s=50)
    plt.colorbar(scatter, label='Max Firing Strength')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title('UMAP-Visualisierung der Firing Strengths')
    plt.grid(True)
    plt.show()
    

def plot_firing_strength_heatmap(self, x_min, x_max, y_min, y_max, grid_size=100, rule_index=None):
        """
        Erzeugt eine Heatmap der Firing Strengths über einen 2D-Eingaberaum.
        Annahme: Das Modell hat 2 Eingangsdimensionen.
        
        :param x_min: Minimaler x-Wert.
        :param x_max: Maximaler x-Wert.
        :param y_min: Minimaler y-Wert.
        :param y_max: Maximaler y-Wert.
        :param grid_size: Auflösung des Gitters.
        :param rule_index: Index einer spezifischen Regel. Falls None, wird der maximale Firing Strength-Wert verwendet.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        # Erzeuge ein Gitter im Eingaberaum
        x_vals = np.linspace(x_min, x_max, grid_size)
        y_vals = np.linspace(y_min, y_max, grid_size)
        xx, yy = np.meshgrid(x_vals, y_vals)
        # Erstelle Eingabepunkte (hier gehen wir davon aus, dass es 2 Features gibt)
        grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
        
        # Führe einen Forward-Pass durch, um die Firing Strengths zu erhalten
        self.eval()
        with torch.no_grad():
            _, firing_strengths, _ = self(grid_tensor)
        
        # firing_strengths hat die Form [N, num_rules]
        if rule_index is not None:
            # Wähle den Firing Strength-Wert der angegebenen Regel
            selected_fs = firing_strengths[:, rule_index].cpu().numpy()
        else:
            # Wähle für jeden Punkt den maximalen Firing Strength-Wert
            selected_fs = firing_strengths.max(dim=1)[0].cpu().numpy()
        
        # Reshape in die Gitterform
        selected_fs = selected_fs.reshape(grid_size, grid_size)
        
        # Plotten der Heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(selected_fs, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis')
        plt.colorbar(label="Firing Strength")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Firing Strength Heatmap" + (f" (Regel {rule_index})" if rule_index is not None else " (maximal)"))
        plt.show()

def plot_umap_fixed_rule(model, X, rule_index, cmap='viridis'):
    """
    Erzeugt einen UMAP-Plot der Firing Strengths, wobei die Punkte
    nach der Aktivierung einer festen Regel (rule_index) eingefärbt werden.
    
    Args:
        model: Das ANFIS-Modell.
        X: Eingabedaten (Tensor, z. B. [N, input_dim]).
        rule_index: Index der festen Regel, deren Aktivierung zur Farbgebung verwendet wird.
        cmap: Colormap für den Plot.
    """
    import umap.umap_ as umap
    import matplotlib.pyplot as plt
    
    model.eval()
    with torch.no_grad():
        # Berechne die normalized firing strengths: Shape [N, num_rules]
        _, firing_strengths, _ = model(X)
    
    # Konvertiere in ein NumPy-Array
    firing_strengths_np = firing_strengths.cpu().numpy()  # [N, num_rules]
    
    # Prüfe, ob der Regelindex gültig ist
    if rule_index < 0 or rule_index >= firing_strengths_np.shape[1]:
        raise ValueError(f"rule_index {rule_index} liegt außerhalb des gültigen Bereichs [0, {firing_strengths_np.shape[1]-1}]")
    
    # Verwende UMAP, um die gesamte Firing-Strength-Vektor (über alle Regeln) auf 2D zu reduzieren
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(firing_strengths_np)  # [N, 2]
    
    # Verwende den Firing-Strength-Wert der festen Regel als Farbe
    colors = firing_strengths_np[:, rule_index]
    
    plt.figure(figsize=(10,8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=cmap, s=50)
    plt.colorbar(scatter, label=f'Firing Strength von Regel {rule_index}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title(f'UMAP Plot der Firing Strengths (Feste Regel {rule_index})')
    plt.grid(True)
    plt.show()

import matplotlib.pyplot as plt
import torch

def plot_sample_firing_strengths(model, sample, rule_names=None):
    """
    Plottet für ein einzelnes Sample die Firing Strengths aller Regeln als Balkendiagramm.
    
    :param model: Das ANFIS-Modell.
    :param sample: Ein einzelnes Eingabebeispiel als Torch-Tensor der Form [input_dim].
    :param rule_names: Optionale Liste von Namen für die Regeln (z.B. ['Rule 0', 'Rule 1', ...])
    """
    model.eval()
    with torch.no_grad():
        # Führe einen Forward-Pass durch. Beachte: sample.unsqueeze(0) macht aus [input_dim] einen Batch von 1
        _, firing_strengths, _ = model(sample.unsqueeze(0))  # Shape: [1, num_rules]
    
    # Entferne den Batch-Dimension und wandle in NumPy um
    fs = firing_strengths.squeeze(0).cpu().numpy()
    
    num_rules = fs.shape[0]
    x = list(range(num_rules))
    
    plt.figure(figsize=(10, 5))
    plt.bar(x, fs, color='skyblue')
    plt.xlabel("Regel-Index")
    plt.ylabel("Firing Strength")
    if rule_names is None:
        plt.title("Firing Strengths für das ausgewählte Sample")
    else:
        plt.title("Firing Strengths für das ausgewählte Sample\n" + ", ".join(rule_names))
    plt.xticks(x, rule_names if rule_names is not None else x, rotation=45)
    plt.tight_layout()
    plt.show()



