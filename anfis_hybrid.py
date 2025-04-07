import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#-------------------
# Modell
#-------------------



class HybridANFIS(nn.Module):
    def __init__(self, input_dim, num_classes, num_mfs, max_rules, seed):

        seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # if torch.cuda.is_available():
        #      torch.cuda.manual_seed_all(seed)
        super(HybridANFIS, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_mfs = num_mfs  
        self.num_rules = num_mfs ** input_dim  # Total number of rules

        # Membership function parameters (Gaussian)
        self.centers = nn.Parameter(torch.ones(input_dim, num_mfs))  # Centers
        #print(self.centers[0])
        self.widths = nn.Parameter(torch.ones(input_dim, num_mfs))  # Widths
        if self.num_rules <= max_rules:
            self.rules = torch.cartesian_prod(*[torch.arange(self.num_mfs) for _ in range(self.input_dim)])
        else:
            self.rules = torch.randint(low=0,
                    high=self.num_mfs,
                    size=(max_rules, self.input_dim))
            self.num_rules = max_rules
        #print(self.rules, "rules -------------------------")

        # full_rules.shape = [8192, 13]

        # max_rules
        # idx = torch.randperm(full_rules.size(0))[:max_rules]
        # self.rules = full_rules[idx]  # => shape [max_rules, input_dim]
        # self.num_rules = self.rules.size(0)  # =max_rules

        # Direkte Zufallserzeugung von max_rules Regeln
        # => self.rules wird [max_rules, input_dim] 
        #    mit Werten in [0, num_mfs-1].


        # Consequent parameters (initialized randomly)
        #print(input_dim + 1)
        self.consequents = nn.Parameter(torch.ones(self.num_rules, input_dim + 1, num_classes))
        self.consequents.requires_grad = False



    def gaussian_mf(self, x, center, width):
        #shape: batch_size, num_mfs
        gaus = torch.exp(-((x - center) ** 2) / (2 * width ** 2))
        return gaus

    def forward(self, x):
        batch_size = x.size(0)
        eps = 1e-9

        # Step 1: Compute membership values
        mfs = []
        for i in range(self.input_dim):
            x_i = x[:, i].unsqueeze(1)  # Shape: [batch_size, 1]
            #print(x_i, "x wert")
            center_i = self.centers[i]  # Shape: [num_mfs]
            width_i = self.widths[i]    # Shape: [num_mfs]
            mf_i = self.gaussian_mf(x_i, center_i, width_i) + eps # Shape: [batch_size, num_mfs]
            #print(mf_i, "mf i")
            mfs.append(mf_i)

        mfs = torch.stack(mfs, dim=1)  # Shape: [batch_size, input_dim, num_mfs]

        #print(mfs[0], "mfs(0)")

        # Step 2: Compute rule activations
        #full_rules = torch.cartesian_prod(*[torch.arange(self.num_mfs) for _ in range(self.input_dim)])

        # rules.shape => [num_rules, input_dim]

        rules_idx = self.rules.unsqueeze(0).expand(batch_size, -1, -1).permute(0, 2, 1)
        #print(rules_idx, "rule index")
        

        # rules_idx.shape => [batch_size, input_dim, num_rules]

        # Now gather along dim=2 in 'mfs'
        # mfs.shape => [batch_size, input_dim, num_mfs]
        rule_mfs = torch.gather(mfs, dim=2, index=rules_idx)
       
       # print(rule_mfs, "rule_mfs")
        #print(rule_mfs.shape)
        
        # rule_mfs.shape => [batch_size, input_dim, num_rules]

   
        
        # Multiply membership values across input_dim
        fiering_strengths = torch.prod(rule_mfs, dim=1)
        

        # shape => [batch_size, num_rules]


        # Step 3: Normalize rule activations
        
        
        normalized_firing_strengths = fiering_strengths / (fiering_strengths.sum(dim=1, keepdim=True) + eps)
        # if torch.le(torch.tensor(0),normalized_firing_strengths).any():
        #     print(normalized_firing_strengths)
        #     raise Exception("Sorry, no numbers below zero")
        # soft= nn.Softmax(dim=1)
        # normalized_firing_strengths = soft(fiering_strengths)
        rule_mfs = torch.ones(batch_size, self.num_rules, self.input_dim)

        rule_outputs_list = []

        # Step 4: Compute rule outputs
        x_ext = torch.cat([x, torch.ones(batch_size, 1)], dim=1)  # Add bias term, shape: [batch_size, input_dim + 1]

        #rule_mfs = torch.einsum('bi,rjc->brc', x_ext, self.consequents)  
        #print(rule_mfs.shape)
        # for r in range(self.num_rules):
        #     out_r = torch.matmul(x_ext, self.consequents[r])
        #     rule_outputs_list.append(out_r)

        # rule_mfs = torch.stack(rule_outputs_list, dim=1)


        
        rule_outputs = torch.einsum('br,brc->bc', normalized_firing_strengths, 
                                   torch.einsum('bi,rjc->brc', x_ext, self.consequents))

        # Schritt 1: Berechne die Regel-MF-Werte [B, R, C]
        # [B, R, C]
        # output_list = []
        # rule_mfs = rule_mfs.permute(0, 2, 1)
        # for b in range(batch_size):
        #     out_r = torch.matmul(rule_mfs[b], normalized_firing_strengths[b])
        #     output_list.append(out_r)
        
        # rule_outputs = torch.stack(output_list, dim=0)


        # Schritt 2: Gewichtete Summe der Regel-MF-Werte [B, C]

        #rule_outputs = torch.einsum('br, brc->bc', normalized_firing_strengths, rule_mfs)  # [B, C] 

        #print(rule_outputs.shape, "rule_outputs.shape")
        
        
        
        # Shape: [batch_size, num_classes]
        return rule_outputs, normalized_firing_strengths, x_ext
    

    def update_consequents(self, normalized_firing_strengths, x_ext, Y, A_total, B_total):
        """
        Update consequent parameters using Least Squares Estimation (LSE).
        :param normalized_firing_strengths: Normalized rule activations, shape: [batch_size, num_rules]
        :param x_ext: Extended inputs (with bias), shape: [batch_size, input_dim + 1]
        :param y: Target outputs (one-hot encoded), shape: [batch_size, num_classes]
        """

       ## batch_size = normalized_firing_strengths.size(0)

        # Prepare the design matrix (Phi)
        #batch_size x num_rules        batch_size x input_dim + 1

        ##Phi = normalized_firing_strengths.unsqueeze(2) * x_ext.unsqueeze(1)  # Shape: [batch_size, num_rules, input_dim + 1]

       ## Phi = Phi.view(batch_size, self.num_rules * (self.input_dim + 1))  # Flattened design matrix
#        for i in rang(batch_size):
 #           B[i] = torch.linalg.lstsq(Phi[i], Y)

    
 


        ##B = torch.linalg.lstsq(Phi, Y).solution

        ##if torch.isnan(B).any():
          ##  print("NaN in LSE-solution B!")
        

        #B = torch.linalg.solve(Phi_T_Phi, Phi_T_Y)  # => [P, C]

        Phi = normalized_firing_strengths.unsqueeze(2) * x_ext.unsqueeze(1)
        # Jetzt flache Phi zu [batch_size, num_rules * (input_dim+1)] ab:

        batch_size_actual = Phi.shape[0]
        P = self.num_rules * (self.input_dim + 1)
   
        Phi = Phi.view(batch_size_actual, P)

        
        


        # Löse das Gleichungssystem: A_total * solution = B_total
        # solution hat dann die Form: [P, num_classes]
        #solution = torch.linalg.solve(A_total, B_total)
        solution = torch.linalg.lstsq(Phi, Y).solution


        # Reshape in die Form der consequent Parameter: [num_rules, input_dim+1, num_classes]
        self.consequents.data = solution.view(self.num_rules, self.input_dim + 1, self.num_classes)

            # Anschließende Reshape
        # self.consequents.data = B.view(self.num_rules, self.input_dim + 1, self.num_classes)
        #print("Batchweise LSE-Update abgeschlossen.")


def train_hybrid_anfis(model, X, Y, num_epochs, lr):
    device = torch.device("cpu")  # or "cuda"
    model.to(device)
    model.train()
    
    # Only optimize membership function params with an optimizer
    optimizer = optim.Adam([model.centers, model.widths], lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    trainset = torch.utils.data.TensorDataset(X, Y)
    dataloader = DataLoader(trainset, batch_size=64, shuffle=True)

    losses = []
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        
        # 1) === Mini-Batch Pass for MF parameters ===
        print("hier")
        A_total = None  # Summe von Phi^T Phi
        B_total = None  # Summe von Phi^T Y
        for batch_X, batch_Y in dataloader:
            # Forward
            outputs, firing_strengths, x_ext = model(batch_X)
            loss = criterion(outputs, batch_Y)
            
            # Backprop for MF parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.widths.data.clamp_(min=1e-3, max=1)
            model.centers.data.clamp_(min=0, max=1)
            
            

            with torch.no_grad():
            # Rebuild full design matrix Phi and target Y for the entire dataset
            # (Better to avoid double-coded logic by reusing a function)
            # forward on the full dataset

                #outputs_all, firing_strengths_all, x_ext_all = model(X)

                
                # Convert to one-hot if classification
                Y_onehot_all = F.one_hot(batch_Y, num_classes=model.num_classes).float()
                
                model.update_consequents(
                    firing_strengths,  # shape: [N, num_rules]
                    x_ext,             # shape: [N, input_dim+1]
                    Y_onehot_all,          # shape: [N, num_classes]
                    A_total,
                    B_total
                )
        epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        
        # 2) === LSE update over the entire dataset at once ===


        # 3) Print or log
        if (epoch+1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    # Plot losses
    plt.figure()
    plt.plot(range(1, num_epochs+1), losses, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.show()


import umap.umap_ as umap
import matplotlib.pyplot as plt

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

def train_hybrid_anfis_old(model, X, Y, num_epochs, lr):
    
    print("beginn")
    
    device = torch.device("cpu")  # Oder "cuda" wenn GPU verfügbar ist
    model.to(device)
    model.train()
    # Optimiert nur MF-Parameter (centers, widths)
    optimizer = optim.Adam([model.centers, model.widths], lr=lr)
    criterion = nn.CrossEntropyLoss()
    trainset = torch.utils.data.TensorDataset(X, Y)
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
   
            outputs, firing_strengths, x_ext = model(batch_X)
     
            

            # Forward Pass
              # outputs: [batch_size, num_classes]

            #print(outputs, "output ----------")
            #Y_onehot = torch.squeeze(Y_onehot)
            # print(outputs.shape, " output")
            # print(batch_Y.shape)
            #batch_Y = batch_Y.squeeze(1)
            loss = criterion(outputs, batch_Y)  


            # Backpropagation auf MF-Parameter
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_([model.centers, model.widths], max_norm=1.0)
            optimizer.step()

            epochs = []
            
            accuracies = []
            # 3) LSE-Update der Konklusionsgewichte
         
            with torch.no_grad():
                # Y als One-Hot
                Y_onehot = F.one_hot(batch_Y, num_classes=model.num_classes).float()
                
                model.update_consequents(
                     firing_strengths.detach(), 
                     x_ext.detach(), 
                     Y_onehot
                 )
      
                # model.fit_coeff(firing_strengths.detach(), 
                #     x_ext.detach(), 
                #     Y_onehot)
            
            
        epochs.append(epoch)


        print(epochs)


        epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)



        # Optional: Ausgeben
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
            
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.title('Trainingskurve (Loss über Epochen)')
    plt.grid(True)
    plt.show()