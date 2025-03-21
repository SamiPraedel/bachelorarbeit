import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


#-------------------
# Modell
#-------------------



class NoHybridANFIS(nn.Module):
    def __init__(self, input_dim, num_classes, num_mfs, max_rules, seed):


        seed = seed
        #random.seed(seed)
        #np.random.seed(seed)
        #torch.manual_seed(seed)
        #if torch.cuda.is_available():
         #    torch.cuda.manual_seed_all(seed)
        super(NoHybridANFIS, self).__init__()
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

        print(self.rules.shape)
        # full_rules.shape = [8192, 13]

        # max_rules
        # idx = torch.randperm(full_rules.size(0))[:max_rules]
        # self.rules = full_rules[idx]  # => shape [max_rules, input_dim]
        # self.num_rules = self.rules.size(0)  # =max_rules

        # Direkte Zufallserzeugung von max_rules Regeln
        # => self.rules wird [max_rules, input_dim] 
        #    mit Werten in [0, num_mfs-1].


        # Consequent parameters (initialized randomly)
        self.consequents = nn.Parameter(torch.rand(self.num_rules, input_dim + 1, num_classes))


    def gaussian_mf(self, x, center, width):
        gaus = torch.exp(-((x - center) ** 2) / (2 * width ** 2))
        if torch.isnan(center).any():
            print("center in gaus")
        
        if torch.isnan(width).any():
            print("width in gauss")
        
        if torch.isinf(gaus).any():
            print("x in gauss  ", torch.exp(-((x - center) ** 2) / (2 * width ** 2)))

        return gaus

    def forward(self, x):
        batch_size = x.size(0)

        # Step 1: Compute membership values
        mfs = []
        for i in range(self.input_dim):
            x_i = x[:, i].unsqueeze(1)  # Shape: [batch_size, 1]
            center_i = self.centers[i]  # Shape: [num_mfs]
            width_i = self.widths[i]    # Shape: [num_mfs]
            mf_i = self.gaussian_mf(x_i, center_i, width_i)  # Shape: [batch_size, num_mfs]
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
        # soft= nn.Softmax(dim=1)
        # normalized_firing_strengths = soft(fiering_strengths)

        # Step 4: Compute rule outputs
        x_ext = torch.cat([x, torch.ones(batch_size, 1)], dim=1)  # Add bias term, shape: [batch_size, input_dim + 1]
        # rule_outputs = torch.einsum('br,brc->bc', normalized_firing_strengths, 
        #                              torch.einsum('bi,rjc->brc', x_ext, self.consequents))

        # Schritt 1: Berechne die Regel-MF-Werte [B, R, C]
        rule_mfs = torch.einsum('bi,rjc->brc', x_ext, self.consequents)  # [B, R, C]

        # Schritt 2: Gewichtete Summe der Regel-MF-Werte [B, C]
        rule_outputs = torch.einsum('br, brc->bc', normalized_firing_strengths, rule_mfs)  # [B, C]


        
        
        
        # Shape: [batch_size, num_classes]
        return rule_outputs, normalized_firing_strengths, x_ext

    

def train_anfis(model, X, Y, num_epochs, lr):
    device = torch.device("cpu")  # Oder "cuda" wenn GPU verfügbar ist
    model.to(device)
    model.train()
    # Optimiert nur MF-Parameter (centers, widths)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    
    trainset = torch.utils.data.TensorDataset(X, Y)
    dataloader = DataLoader(trainset, batch_size=60, shuffle=True)
    


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
            # accuracy = (outputs == batch_Y).float().mean().item() * 100
            # print(accuracy)


        


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


