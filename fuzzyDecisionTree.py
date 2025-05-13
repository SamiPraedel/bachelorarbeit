# soft_decision_tree.py
import torch, torch.nn as nn, numpy as np, random

# class SoftGate(nn.Module):
#     def __init__(self, dim, data_col, init_s=0.25):
#         super().__init__()
#         m = np.median(data_col.cpu().numpy())
#         self.c      = nn.Parameter(torch.tensor(float(m)))
#         self.log_s  = nn.Parameter(torch.log(torch.tensor(init_s)))
#         self.dim    = dim                     # nur zum Merken

#     def forward(self, x):                     # x:[B,D]
#         z  = (self.c - x[:, self.dim]) / self.log_s.exp()
#         return torch.sigmoid(5*z)             # [B]

# fuzzyDecisionTree.py  ---------------------------

class SoftGate(nn.Module):
    def __init__(self, in_dim, init_s=0.3):
        super().__init__()
        # 1) weiche Feature-Wahl (logits über Features)
        self.log_alpha = nn.Parameter(torch.zeros(in_dim))
        # 2) Zentrum / Spread
        self.c   = nn.Parameter(torch.zeros(()))          # wird nach init gesetzt
        self.log_s = nn.Parameter(torch.log(torch.tensor(init_s)))
        self.in_dim = in_dim

    def forward(self, x, tau=0.5):
        # -----  Soft Feature-Attention  -----------------
        pi = torch.softmax(self.log_alpha / tau, dim=0)     # [F]  Σ pi =1
        z  = (x @ pi - self.c) / (self.log_s.exp() + 1e-3)
        # steilere Sigmoid gibt klarere Splits, 6≈≈ slope
        return torch.sigmoid(-6*z)          # μ ∈ (0,1)



# ----------------------------------------------------------
class SDTNode(nn.Module):
    def __init__(self, depth, max_depth, in_dim, n_out):
        super().__init__()
        self.is_leaf = depth == max_depth
        if self.is_leaf:
            self.logits = nn.Parameter(torch.zeros(n_out))
        else:
            self.gate  = SoftGate(in_dim)
            self.left  = SDTNode(depth+1, max_depth, in_dim, n_out)
            self.right = SDTNode(depth+1, max_depth, in_dim, n_out)

    def forward(self,x,tau):
        if self.is_leaf:
            return self.logits.expand(x.size(0),-1)
        μ = self.gate(x,tau)                       # [B]
        print(μ)
        return μ[:,None]*self.left(x,tau) + (1-μ)[:,None]*self.right(x,tau)


class SoftDecisionTree(nn.Module):
    def __init__(self, in_dim, n_out, depth):
        super().__init__()
        self.root = SDTNode(0, depth, in_dim, n_out)


    def forward(self, x, tau):
        return self.root(x, tau)
