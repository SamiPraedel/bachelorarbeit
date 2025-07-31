import torch
import torch.nn as nn
import numpy as np
import random

class NoHybridANFIS(nn.Module):
    def __init__(self, input_dim, num_classes, num_mfs, max_rules, seed, zeroG=False, topk_r=0.2):
       # num_mfs_tensor = torch.tensor[3,3,3,2,2,2]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        super(NoHybridANFIS, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.M = num_mfs
        self.num_rules = num_mfs ** input_dim if num_mfs**input_dim <= max_rules else max_rules
        self.zeroG = zeroG

        self.centers = nn.Parameter(torch.ones(input_dim, num_mfs))  # Centers
        self.widths = nn.Parameter(torch.ones(input_dim, num_mfs))  # Widths


        if ((num_mfs**input_dim) <= max_rules):
            self.rules = torch.cartesian_prod(*[torch.arange(self.M) for _ in range(self.input_dim)])
        else:
            self.rules = torch.randint(low=0,
                    high=self.M,
                    size=(max_rules, self.input_dim))
            self.num_rules = max_rules

        if self.zeroG:
            self.consequents = nn.Parameter(torch.rand(self.num_rules, num_classes))
        else:
            self.consequents = nn.Parameter(torch.rand(self.num_rules, input_dim + 1, num_classes))
        
        self.register_buffer("rule_idx", self.rules.t())  # shape [d, R]
        self.topk_r = topk_r
        
        # self.fc1 = nn.Linear(self.num_rules, 16)
        # self.act = nn.ReLU()
        # self.fc2 = nn.Linear(16, num_classes)

        

    def gaussian_mf(self, x, center, width):

        return torch.exp(-((x - center) ** 2) / (2 * width ** 2))
    

    def forward(self, x):
        batch_size = x.size(0)
        
        x_exp      = x.unsqueeze(2)                      # [B, d, 1]
        centers    = self.centers.unsqueeze(0)           # [1, d, m]
        widths     = self.widths.unsqueeze(0)            # [1, d, m]
        mfs  = torch.exp(-((x_exp - centers) ** 2) /
                            (2 * widths ** 2) + 1e-9)
        

            
        # rules.shape => [num_rules, input_dim]
        self.rules_idx = self.rules.unsqueeze(0).expand(batch_size, -1, -1).permute(0, 2, 1)
        # rules_idx.shape => [batch_size, input_dim, num_rules]
        
        rules_idx_on_device = self.rules_idx.to(mfs.device)
        rule_mfs = torch.gather(mfs, dim=2, index=rules_idx_on_device)  #rule_mfs => [batch_size, input_dim, num_rules]

        #rule_mfs = torch.gather(mfs, dim=2, index=rules_idx)  #rule_mfs => [batch_size, input_dim, num_rules]

        fiering_strengths = torch.prod(rule_mfs, dim=1)  #[batch_size, num_rules]        
        
        topk_p = self.topk_r

        K = max(1, int(topk_p * self.num_rules))
        vals, idx = torch.topk(fiering_strengths, k=K, dim=1)
        mask = torch.zeros_like(fiering_strengths).scatter_(1, idx, 1.)
        firing = fiering_strengths * mask
        normalized_firing_strengths = firing / (firing.sum(1, keepdim=True)+1e-9)
        
        
        #normalized_firing_strengths = fiering_strengths / (fiering_strengths.sum(dim=1, keepdim=True) + eps) 
        #print(normalized_firing_strengths)

        #x_ext = torch.cat([x, torch.ones(batch_size, 1)], dim=1)  # Add bias term, shape: [batch_size, input_dim + 1]
        x_ext = torch.cat([x, torch.ones(batch_size, 1, device=x.device)], dim=1)  # Add bias term, shape: [batch_size, input_dim + 1]
        
        # Schritt 1: Berechne die Regel-MF-Werte [B, R, C]
        if self.zeroG:
            outputs = self.consequents
            rule_outputs = torch.einsum('br,rc->bc', normalized_firing_strengths, outputs)
        else:
            rule_mfs = torch.einsum('bi,rjc->brc', x_ext, self.consequents)  # [B, R, C]
            # Schritt 2: Gewichtete Summe der Regel-MF-Werte [B, C]
            rule_outputs = torch.einsum('br, brc->bc', normalized_firing_strengths, rule_mfs)  # [B, C]  
            

        # h = self.act(self.fc1(rule_outputs))  # [B, hidden_size]
        # rule_outputs = self.fc2(h)         
        
        # Shape: [batch_size, num_classes]
        return rule_outputs, normalized_firing_strengths, mask


    def load_balance_loss(self, router_probs, mask, alpha=0.01):
        """
        router_probs : [B, R]   = p_i(x)
        mask         : [B, R]   = 0/1 indicator (top‑K selection)
        """
        T, R = router_probs.shape

        f = mask.float().mean(0)                 # [R]

        P = router_probs.mean(0)                 # [R]
        lb = alpha * R * (f * P).sum()
        return lb
    
    def _fuzzify(self, x):
        """
        Fuzzify the input data x using the membership functions defined by centers and widths.
        Returns the membership values for each input dimension and each membership function.
        """
        x_exp = x.unsqueeze(2)
        centers = self.centers.unsqueeze(0)
        widths = self.widths.unsqueeze(0)
        mfs = torch.exp(-((x_exp - centers) ** 2) / (2 * widths ** 2) + 1e-9)
        return mfs
    
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
    