# dfpt_iris_main.py
import torch, torch.nn as nn, torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# ---------- DFPT-Modell ----------------------------------------------------
class SoftGate(nn.Module):
    """Ein Gate mit Gauß–ähnlicher MF."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.c   = nn.Parameter(torch.rand(()))        # Zentrum
        self.s   = nn.Parameter(torch.rand(()))        # Spread (>0)
    def forward(self, x):
        # μ_left, μ_right   (2 Werte pro Sample)
        z = (x[:, self.dim]-self.c)/self.s.abs()
        mu = torch.sigmoid(-z*4.0)          # smooth step ≈ fuzzy ≤ c
        return torch.stack((mu, 1-mu), dim=1)  # [B,2]

class DFPT_Node(nn.Module):
    def __init__(self, depth, max_depth, in_dim, n_class):
        super().__init__()
        self.is_leaf = depth==max_depth
        if self.is_leaf:
            self.logits = nn.Parameter(torch.zeros(n_class))
        else:
            self.dim   = nn.Parameter(torch.randint(0,in_dim,(1,)), requires_grad=False)
            self.gate  = SoftGate(self.dim.item())
            self.left  = DFPT_Node(depth+1,max_depth,in_dim,n_class)
            self.right = DFPT_Node(depth+1,max_depth,in_dim,n_class)
    def forward(self,x):
        if self.is_leaf:
            B = x.size(0)
            return self.logits.expand(B,-1)
        mu = self.gate(x)            # [B,2]
        out_L = self.left(x)
        out_R = self.right(x)
        return mu[:,[0]]*out_L + mu[:,[1]]*out_R

class DFPT(nn.Module):
    def __init__(self, in_dim, n_class, depth=2):
        super().__init__()
        self.root = DFPT_Node(0, depth, in_dim, n_class)
        self.feat = [f"x{i}" for i in range(in_dim)]
        self.classes = [f"c{i}" for i in range(n_class)]
    def forward(self,x): return self.root(x)
    
def extract_rules(tree, feature_names, class_names, tau=0.1):
    """
    Traversiert den DFPT rekursiv und sammelt alle Regeln,
    deren Pfad-Gewicht >= tau (Schwellwert für Lesbarkeit).
    """
    rules = []

    def _recurse(node, path, weight):
        if node.is_leaf:
            cls_idx = node.logits.argmax().item()
            conf    = weight * torch.softmax(node.logits,0)[cls_idx].item()
            if conf >= tau:
                rules.append((
                    " AND ".join(path) if path else "TRUE",
                    class_names[cls_idx],
                    conf
                ))
            return
        # linker & rechter Zweig
        m = node.mf   # (μ_left, μ_right) callables
        for side,μ in zip(("≤","≥"), m):
            # verbalisiere MF als Intervall/Zentrum+Breite
            desc = f"{feature_names[node.dim]} {side} {round(node.c.item(),3)}±{round(node.s.item()/2,3)}"
            _recurse( node.left if side=="≤" else node.right,
                      path+[desc],
                      weight * node.gate_weight(side) )
    _recurse(tree.root, [], 1.0)
    return sorted(rules, key=lambda r: -r[2])


# ---------- Training -------------------------------------------------------
def train_dfpt():
    X,y = load_iris(return_X_y=True)
    scaler = MinMaxScaler().fit(X); X=scaler.transform(X)
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
    Xtr = torch.tensor(Xtr,dtype=torch.float32)
    ytr = torch.tensor(ytr)
    Xte = torch.tensor(Xte,dtype=torch.float32)
    yte = torch.tensor(yte)

    model = DFPT(in_dim=4, n_class=3, depth=2)
    opt   = optim.Adam(model.parameters(), lr=0.05)
    crit  = nn.CrossEntropyLoss()

    for epoch in range(400):
        opt.zero_grad()
        loss = crit(model(Xtr), ytr)
        loss.backward()
        opt.step()
        if (epoch+1)%100==0:
            pred = model(Xte).argmax(1)
            acc  = (pred==yte).float().mean().item()*100
            print(f"Epoch {epoch+1}: loss={loss.item():.3f}  acc={acc:.1f}%")

    print("\nRule base ≥0.2 conf.:")
    rules = extract_rules(model, ["SepLen","SepWid","PetLen","PetWid"],
                          ["Setosa","Versi","Virginica"], tau=0.2)
    for cond,cls,conf in rules:
        print(f"IF {cond}  THEN {cls}  [{conf:.2f}]")
    return model

# --------------- run -------------------------------------------------------
if __name__=="__main__":
    train_dfpt()
