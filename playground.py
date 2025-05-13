# fmnn_minimal.py
import torch, math
from torch import Tensor
from typing import Optional

class FMNN_Min:
    def __init__(self, theta: float = 1.0, device="cpu"):
        self.th = theta; self.dev = torch.device(device)
        self.V = None  # [B, D]
        self.W = None  # [B, D]
        self.C = None  # [B]
        self.xmin = None; self.xmax = None          # für 0-1-Norm

    # ---------- Hilfsfunktionen ----------
    def _norm(self, X: Tensor, fit=False) -> Tensor:
        if fit:
            self.xmin, self.xmax = X.min(0).values, X.max(0).values
        return (X - self.xmin) / (self.xmax - self.xmin + 1e-12)

    def _membership(self, x: Tensor) -> Tensor:
        left  = 1 - torch.clamp(self.V - x, min=0)
        right = 1 - torch.clamp(x - self.W, min=0)
        return torch.minimum(left, right).amin(dim=1)          # [B]

    # ---------- Public API ----------
    def fit(self, X: Tensor, y: Tensor):
        X, y = X.to(self.dev), y.to(self.dev)
        X = self._norm(X, fit=True)
        for xi, yi in zip(X, y):
            if self.V is None:             # 1. Box
                self.V = xi.unsqueeze(0);  self.W = xi.unsqueeze(0)
                self.C = yi.unsqueeze(0);  continue

            # --- Box derselben Klasse suchen + evtl. expandieren
            same = (self.C == yi)
            if same.any():
                m = self._membership(xi)[same]
                idx_local = int(m.argmax())
                idx = same.nonzero(as_tuple=True)[0][idx_local]
                v_new = torch.minimum(self.V[idx], xi)
                w_new = torch.maximum(self.W[idx], xi)
                
                if (w_new - v_new).max() <= self.th:           # θ-Kriterium
                    self.V[idx], self.W[idx] = v_new, w_new
                    self._contract(idx);           continue

            # --- sonst neue Box
            self.V = torch.cat([self.V, xi.unsqueeze(0)])
            self.W = torch.cat([self.W, xi.unsqueeze(0)])
            self.C = torch.cat([self.C, yi.unsqueeze(0)])

    def _contract(self, j: int):
        for k in range(len(self.V)):
            if k == j or self.C[k] == self.C[j]: continue
            # 1-D-Overlap?
            low  = torch.maximum(self.V[j], self.V[k])
            high = torch.minimum(self.W[j], self.W[k])
            inter = (high - low) > 0
            if inter.sum() == 1:
                i = int(inter.nonzero())
                # 4 Fälle
                vj, wj, vk, wk = self.V[j,i], self.W[j,i], self.V[k,i], self.W[k,i]
                if   vj < vk < wj < wk: vk = wj = (vk+wj)/2
                elif vk < vj < wk < wj: vj = wk = (vj+wk)/2
                elif vj < vk < wk < wj:
                    if (wj-vk) > (wk-vj): vj = wk
                    else:                 wj = vk
                else:  # vk < vj < wj < wk
                    if (wk-vj) > (wj-vk): vk = wj
                    else:                 wk = vj
                self.V[j,i], self.W[j,i] = vj, wj
                self.V[k,i], self.W[k,i] = vk, wk

    def predict(self, X: Tensor) -> Tensor:
        X = self._norm(X.to(self.dev))
        preds = []
        for xi in X:
            idx = int(self._membership(xi).argmax())
            preds.append(int(self.C[idx]))
        return torch.tensor(preds, device=self.dev)

    def score(self, X: Tensor, y: Tensor) -> float:
        return (self.predict(X)==y.to(self.dev)).float().mean().item()


if __name__ == "__main__":
    from data_utils import load_iris_data, load_heart_data, load_Kp_chess_data       # nimmt NumPy → Torch
    #Xtr,ytr,Xte,yte = load_iris_data()
    #Xtr,ytr,Xte,yte = load_heart_data()
    Xtr,ytr,Xte,yte = load_Kp_chess_data()
    clf = FMNN_Min(theta=0.5)
    clf.fit(Xtr,ytr)
    print("ACC =", clf.score(Xte,yte))       # ~0.93 – 0.96
