import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#-------------------
# Modell
#-------------------


class HybridANFIS(nn.Module):
    def __init__(self, input_dim, num_classes, num_mfs, max_rules):
        super(HybridANFIS, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_mfs = num_mfs
        self.num_rules = num_mfs ** input_dim  # Total number of rules

        # Membership function parameters (Gaussian)
        self.centers = nn.Parameter(torch.rand(input_dim, num_mfs))  # Centers
        #print(self.centers[0])
        self.widths = nn.Parameter(torch.rand(input_dim, num_mfs))  # Widths

        # Im __init__ deines Modells
        full_rules = torch.cartesian_prod(*[torch.arange(self.num_mfs) for _ in range(self.input_dim)])
        # full_rules.shape = [8192, 13]

        # max_rules
        idx = torch.randperm(full_rules.size(0))[:max_rules]
        self.rules = full_rules[idx]  # => shape [max_rules, input_dim]
        self.num_rules = self.rules.size(0)  # =max_rules

        # Consequent parameters (initialized randomly)
        self.consequents = nn.Parameter(torch.rand(self.num_rules, input_dim + 1, num_classes))
        self.consequents.requires_grad = False


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
            if torch.isnan(x).any():
                print("vor unsqueeze")
            x_i = x[:, i].unsqueeze(1)  # Shape: [batch_size, 1]
            if torch.isnan(x_i).any():
                print("nach unsqueeze")
            #print(self.centers)
            center_i = self.centers[i]  # Shape: [num_mfs]


            width_i = self.widths[i]    # Shape: [num_mfs]

            if torch.isnan(center_i).any():
                print("center in gaus")
        
            if torch.isnan(width_i).any():
                print("width in gauss")

  

            mf_i = self.gaussian_mf(x_i, center_i, width_i)  # Shape: [batch_size, num_mfs]
            if torch.isnan(self.gaussian_mf(x_i, center_i, width_i)).any():
                print("mfi")
                if torch.isnan(x_i).any():
                    print(" hihi ")



            mfs.append(mf_i)

        mfs = torch.stack(mfs, dim=1)  # Shape: [batch_size, input_dim, num_mfs]
        if torch.isnan(mfs).any():
            print("width in gauss")
        



 
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
   

        # Step 4: Compute rule outputs
        x_ext = torch.cat([x, torch.ones(batch_size, 1)], dim=1)  # Add bias term, shape: [batch_size, input_dim + 1]
        rule_outputs = torch.einsum('br,brc->bc', normalized_firing_strengths, 
                                    torch.einsum('bi,rjc->brc', x_ext, self.consequents))
        
        
        
        # Shape: [batch_size, num_classes]
        return rule_outputs, normalized_firing_strengths, x_ext

    def update_consequents(self, normalized_firing_strengths, x_ext, Y):
        """
        Update consequent parameters using Least Squares Estimation (LSE).
        :param normalized_firing_strengths: Normalized rule activations, shape: [batch_size, num_rules]
        :param x_ext: Extended inputs (with bias), shape: [batch_size, input_dim + 1]
        :param y: Target outputs (one-hot encoded), shape: [batch_size, num_classes]
        """
        batch_size = normalized_firing_strengths.size(0)

        # Prepare the design matrix (Phi)
        Phi = normalized_firing_strengths.unsqueeze(2) * x_ext.unsqueeze(1)  # Shape: [batch_size, num_rules, input_dim + 1]
        Phi = Phi.view(batch_size, self.num_rules * (self.input_dim + 1))  # Flattened design matrix

        # Solve the least-squares problem: Phi.T @ Phi @ consequents = Phi.T @ y
        #Phi_T_Phi = torch.matmul(Phi.T, Phi)
  
 

        B = torch.linalg.lstsq(Phi, Y).solution

        #PhiT = Phi.transpose(0,1)
        #lambda_ = 1e-3
        #A = PhiT @ Phi
        #A += lambda_ * torch.eye(A.shape[0], device=A.device)
        #B = torch.linalg.solve(A, PhiT @ Y)
        

        if torch.isnan(B).any():
            print("NaN in LSE-solution B!")

        #B = torch.linalg.solve(Phi_T_Phi, Phi_T_Y)  # => [P, C]

        # Anschließende Reshape
        self.consequents.data = B.view(self.num_rules, self.input_dim + 1, self.num_classes)

def train_hybrid_anfis(model, X, Y, num_epochs, lr):
    # Optimiert nur MF-Parameter (centers, widths)
    optimizer = optim.AdamW([model.centers, model.widths], lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):

        
        # 1) Forward Pass
        outputs, firing_strengths, x_ext = model(X)
        loss = criterion(outputs, Y)


        # 2) Backprop auf MF-Parameter
        optimizer.zero_grad()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_([model.centers, model.widths], max_norm=1.0)

        optimizer.step()
  
        
        # 3) LSE-Update der Konklusionsgewichte
        with torch.no_grad():
            # Y als One-Hot
            Y_onehot = F.one_hot(Y, num_classes=model.num_classes).float()
            model.update_consequents(
                firing_strengths.detach(), 
                x_ext.detach(), 
                Y_onehot
            )


        # Optional: Ausgeben
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")



