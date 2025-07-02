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
        self.M = num_mfs  
        self.num_rules = num_mfs ** input_dim
        self.max_rules = max_rules
        
        self.centers = nn.Parameter(torch.ones(input_dim, num_mfs), requires_grad=True)
        self.widths = nn.Parameter(torch.ones(input_dim, num_mfs), requires_grad=True)  
        
        if self.num_rules <= max_rules:
            self.rules = torch.cartesian_prod(*[torch.arange(self.M) for _ in range(self.input_dim)])
        else:
            self.rules = torch.randint(low=0,
                    high=self.M,
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
        #device = x.device
        batch_size = x.size(0)

        x_exp      = x.unsqueeze(2)                      # [B, d, 1]
        centers    = self.centers.unsqueeze(0)           # [1, d, m]
        widths     = self.widths.unsqueeze(0)            # [1, d, m]
        mfs  = torch.exp(-((x_exp - centers) ** 2) /
                            (2 * widths ** 2) + 1e-9)
            
        #mfs = mfs.to(device)        
        rules_expand = self.rule_idx.unsqueeze(0).expand(batch_size, -1, -1)  # [B, d, R]
        rule_mfs = torch.gather(mfs, 2, rules_expand)

        

        
        firing_s = rule_mfs.prod(dim=1)
        
        
        
        #norm_fs = firing_strengths / (firing_strengths.sum(dim=1, keepdim=True) + 1e-9)
        norm_fs = firing_s / (firing_s.sum(dim=1, keepdim=True) + 1e-9)

        
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
        #print("Phi shape:", Phi.shape)

        Phi = Phi.view(batch_size, self.num_rules * (self.input_dim + 1))  # Flattened design matrix
        #print("Phi flattened shape:", Phi.shape)

        #B = torch.linalg.lstsq(Phi, Y).solution
        
        
        
        with torch.no_grad():
            #Phi = Phi.cpu()
            Phi_T_Phi = Phi.t().matmul(Phi)
            I = torch.eye(Phi_T_Phi.size(0)).to(Phi.device)  # Identity matrix
            #Y = Y.cpu()
            Phi_T_Y = Phi.t().matmul(Y)
            #print("Phi_T_Y shape:", Phi_T_Y.shape)

            lam = 1e-3
            B = torch.linalg.solve(Phi_T_Phi + lam * I, Phi_T_Y)     # (R*(d+1), C)
            #print("B shape:", B.shape)

            # --- hier fehlt das view! -----------------------------
            B = B.view(self.num_rules, self.input_dim + 1, self.num_classes)

            # jetzt passt es
            self.consequents.copy_(B.to(self.consequents.device))


        # Reshape in die Form der consequent Parameter: [num_rules, input_dim+1, num_classes]
        #self.consequents.data = B.view(self.num_rules, self.input_dim + 1, self.num_classes)
    
    # In HybridANFIS class:
    def update_consequents_gem(self, normalized_firing_strengths: torch.Tensor, 
                           x_ext: torch.Tensor, Y_onehot: torch.Tensor, 
                           lambda_lse: float = 1e-3): # Added lambda_lse as a parameter
        """
        Update consequent parameters using Least Squares Estimation (LSE)
        with Tikhonov regularization.
        """
        # normalized_firing_strengths: [batch_size, num_rules]
        # x_ext: [batch_size, input_dim + 1] (already detached in training loop)
        # Y_onehot: [batch_size, num_classes] (target outputs)

        batch_size = normalized_firing_strengths.size(0)
        
        # Ensure data types are consistent, float32 is usually fine for LSE.
        # If using AMP, inputs might be FP16, ensure they are FP32 before LSE if necessary.
        norm_fs_f32 = normalized_firing_strengths.float()
        x_ext_f32 = x_ext.float()
        Y_onehot_f32 = Y_onehot.float()

        # Phi: [batch_size, num_rules, input_dim + 1]
        Phi_expanded = norm_fs_f32.unsqueeze(2) * x_ext_f32.unsqueeze(1)
        
        # Phi_flat: [batch_size, num_rules * (input_dim + 1)]
        Phi_flat = Phi_expanded.view(batch_size, self.num_rules * (self.input_dim + 1))

        # Perform LSE with Tikhonov regularization
        # Note: Phi_flat and Y_onehot_f32 must be on the same device
        # This block is already under torch.no_grad() in the training loop
        
        # A = Phi_flat.T @ Phi_flat
        A = torch.matmul(Phi_flat.T, Phi_flat)
        
        # b = Phi_flat.T @ Y_onehot_f32
        b = torch.matmul(Phi_flat.T, Y_onehot_f32)

        # Add regularization: A + lambda * I
        I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        A_reg = A + lambda_lse * I
        
        try:
            # Solve (A + lambda*I) * Beta = b for Beta
            Beta_flat = torch.linalg.solve(A_reg, b) # Shape: [num_rules * (input_dim + 1), num_classes]
        except torch.linalg.LinAlgError as e:
            print(f"LSE failed: {e}. Using pseudo-inverse.")
            # Fallback to pseudo-inverse if solve fails (e.g. singular matrix even with regularization)
            A_reg_pinv = torch.linalg.pinv(A_reg)
            Beta_flat = torch.matmul(A_reg_pinv, b)


        # Reshape Beta into the consequents' shape
        # Beta shape: [num_rules, input_dim + 1, num_classes]
        Beta = Beta_flat.view(self.num_rules, self.input_dim + 1, self.num_classes)

        # Update model consequents
        self.consequents.data.copy_(Beta)
        
    @torch.no_grad()
    def _forward_mf_only(self, x):
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
    
    def _fuzzify(self, x):
        """
        Fuzzify the input data x using the membership functions.
        Returns:
            μ: Tensor of shape [B, d, M] where B is batch size, d is input dimension, M is number of MFs.
        """
        x_exp = x.unsqueeze(2)
        centers = self.centers.unsqueeze(0)
        widths = self.widths.unsqueeze(0)
        μ = torch.exp(-((x_exp - centers) ** 2) / (2 * widths ** 2) + 1e-9)
        return μ
