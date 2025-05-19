import torch
import torch.nn as nn
import numpy as np
import random

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
        self.max_rules = max_rules
        
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
            
        mfs = mfs.to(device)        
        rules_expand = self.rule_idx.unsqueeze(0).expand(batch_size, -1, -1)  # [B, d, R]
        rule_mfs = torch.gather(mfs, 2, rules_expand)

        

        
        firing_s = rule_mfs.prod(dim=1)
        
        
        
        #norm_fs = firing_strengths / (firing_strengths.sum(dim=1, keepdim=True) + 1e-9)
        norm_fs = firing_s / (firing_s.sum(dim=1, keepdim=True) + 1e-9)
        #print(norm_fs.shape)
        #print(torch.count_nonzero(norm_fs))
        
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
        print("Phi shape:", Phi.shape)
        
        
        with torch.no_grad():
            Phi = Phi.cpu()
            print("Phi shape on cpu:", Phi.shape)
            Phi_T_Phi = Phi.t().matmul(Phi)
            print("Phi_T_Phi shape:", Phi_T_Phi.shape)
            I = torch.eye(Phi_T_Phi.size(0))
            print("I shape:", I.shape)
            Y = Y.cpu()
            Phi_T_Y = Phi.t().matmul(Y)
            print("Phi_T_Y shape:", Phi_T_Y.shape)

            lam = 1e-3
            B = torch.linalg.solve(Phi_T_Phi + lam * I, Phi_T_Y)     # (R*(d+1), C)
            print("B shape:", B.shape)

            # --- hier fehlt das view! -----------------------------
            B = B.view(self.num_rules, self.input_dim + 1, self.num_classes)

            # jetzt passt es
            self.consequents.copy_(B.to(self.consequents.device))


        # Reshape in die Form der consequent Parameter: [num_rules, input_dim+1, num_classes]
        #self.consequents.data = B.view(self.num_rules, self.input_dim + 1, self.num_classes)
        
    @torch.no_grad()
    def _forward_mf_only(self, x):
        """
        Liefert reine Regel‑Firing‑Stärken (vor Normalisierung).
        x : Tensor [B, input_dim]  (bereits auf dem gleichen Gerät wie die MFs)
        Rückgabe: Tensor [B, num_rules]
        """
        # -- 1. MF‑Grade je Feature & MF --
        x_exp   = x.unsqueeze(2)                  # [B, d, 1]
        centers = self.centers.unsqueeze(0)       # [1, d, m]
        widths  = self.widths.unsqueeze(0)
        mfs     = torch.exp(-((x_exp - centers) ** 2) /
                            (2 * widths ** 2) + 1e-9)           # [B, d, m]

        # -- 2. Relevante MF pro Regel herausziehen --
        rules_expand = self.rule_idx.unsqueeze(0).expand(x.size(0), -1, -1)  # [B, d, R]
        rule_mfs     = torch.gather(mfs, 2, rules_expand)                    # [B, d, R]

        # -- 3. Firing‑Stärke je Regel (Produkt über Features) --
        firing = rule_mfs.prod(dim=1)                                        # [B, R]
        return firing



