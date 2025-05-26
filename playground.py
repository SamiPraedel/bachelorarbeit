# fmnc_continuous.py
from __future__ import annotations
import torch
from torch import Tensor
from typing import Optional, Literal



# ❶ ---------- FMNC (Batch-Version) ---------------------------------
class FMNC:
    def __init__(self,
                 gamma: float = 0.6,
                 theta0: float = 1.0,
                 theta_min: float = 0.6,
                 theta_decay: float = 0.97,
                 bound_mode: Literal["sum", "max"] = "sum",
                 aggr: Literal["min", "mean"] = "min",
                 device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 256):
        self.g, self.th = gamma, theta0
        self.th_min, self.th_decay = theta_min, theta_decay
        self.bound_mode, self.aggr = bound_mode, aggr
        self.dev = torch.device(device)
        self.bs = batch_size                      # <-- NEU

        self.V = self.W = self.cls = None         # Box-Container

    # -------- Batch-Membership  [N,D] -> [N,B] -----------------------
    def _memb_batch(self, X: Tensor) -> Tensor:
        V, W = self.V, self.W                     # [B,D]
        left  = 1 - self.g * torch.clamp(V.unsqueeze(0) - X.unsqueeze(1), min=0)
        right = 1 - self.g * torch.clamp(X.unsqueeze(1) - W.unsqueeze(0), min=0)
        m = torch.minimum(left, right)            # [N,B,D]
        return m.amin(2) if self.aggr == "min" else m.mean(2)   # [N,B]

    def _span(self, v: Tensor, w: Tensor) -> float:
        side = w - v
        return side.sum().item() if self.bound_mode == "sum" else side.max().item()

    def _add_box(self, x: Tensor, y: int):
        self.V  = x.unsqueeze(0) if self.V  is None else torch.cat([self.V,  x.unsqueeze(0)])
        self.W  = x.unsqueeze(0) if self.W  is None else torch.cat([self.W,  x.unsqueeze(0)])
        lbl     = torch.tensor([y], device=self.dev)
        self.cls = lbl            if self.cls is None else torch.cat([self.cls, lbl])

    # -------- vektorisiertes Simpson-Contract (nur 1. Überlapp-Dim) --
    def _contract(self, j: int):
        mask = self.cls != self.cls[j]
        if not mask.any(): return
        V_oth, W_oth = self.V[mask], self.W[mask]

        inter_low  = torch.maximum(V_oth, self.V[j])     # [K,D]
        inter_high = torch.minimum(W_oth, self.W[j])
        inter_len  = inter_high - inter_low
        overlap    = (inter_len > 0).all(1)              # [K]

        if not overlap.any(): return
        for k_sub in overlap.nonzero(as_tuple=False).flatten():
            k = torch.arange(len(self.V), device=self.dev)[mask][k_sub].item()
            vk, wk = self.V[k].clone(), self.W[k].clone()
            pos = (inter_len[k_sub] > 0).nonzero()[0]    # erste Dim mit Overlap
            i = int(pos)
            vj, wj = self.V[j, i], self.W[j, i]

            if   vj < vk[i] < wj < wk[i]:
                vk[i] = wj = (vk[i] + wj) / 2
            elif vk[i] < vj < wk[i] < wj:
                vj = wk[i] = (vj + wk[i]) / 2
            elif vj < vk[i] < wk[i] < wj:
                (vj, wj) = (wk[i], wj) if (wj - vk[i]) > (wk[i] - vj) else (vj, vk[i])
            else:  # vk < vj < wj < wk
                (vk[i], wk[i]) = (wj, wk[i]) if (wk[i] - vj) > (wj - vk[i]) else (vk[i], vj)

            self.V[k, i], self.W[k, i] = vk[i], wk[i]
            self.V[j, i], self.W[j, i] = vj, wj

    # -------- Batch-Online-Fit --------------------------------------
    def fit(self, X: Tensor, y: Tensor, epochs: int = 3, shuffle: bool = True):
        X, y = X.to(self.dev), y.to(self.dev)
        for ep in range(epochs):
            idx = torch.randperm(len(X)) if shuffle else torch.arange(len(X))
            Xs, ys = X[idx], y[idx]

            for s in range(0, len(X), self.bs):
                xb, yb = Xs[s:s+self.bs], ys[s:s+self.bs]

                # zuerst evtl. neue Klassen-Box anlegen
                if self.V is None:
                    for x_i, y_i in zip(xb, yb):
                        self._add_box(x_i, int(y_i))
                    continue

                # At this point, self.V is not None (guaranteed by the block above for the first run).
                # B_snapshot is the number of boxes when `memb` is calculated.
                B_snapshot = self.V.shape[0]
                memb = self._memb_batch(xb) # Resulting shape [len(xb), B_snapshot]

                for i in range(len(xb)):
                    x_i, y_i = xb[i], int(yb[i])

                    # Case 1: No box for this class y_i exists yet (even after previous samples in this batch). Add new box.
                    if (self.cls == y_i).sum() == 0:
                        self._add_box(x_i, y_i); continue

                    # Case 2: Boxes for class y_i exist. Use the pre-calculated memberships.
                    m = memb[i].clone()

                    # If B_snapshot was 0, m will be empty. No boxes from snapshot to consider for expansion.
                    if m.numel() == 0: # Equivalent to B_snapshot == 0
                        self._add_box(x_i, y_i); continue # Add new box as no candidates from snapshot

                    # Apply mask using cls state corresponding to the snapshot.
                    # self.cls might be longer than B_snapshot if boxes were added by previous samples in this batch.
                    cls_at_snapshot = self.cls[:B_snapshot]
                    m[cls_at_snapshot != y_i] = -1
                    j = int(m.argmax())

                    v_new = torch.minimum(self.V[j], x_i)
                    w_new = torch.maximum(self.W[j], x_i)
                    if self._span(v_new, w_new) <= self.th:
                        self.V[j], self.W[j] = v_new, w_new
                        self._contract(j)
                    else:
                        self._add_box(x_i, y_i)

            self.th = max(self.th * self.th_decay, self.th_min)
            print(f"epoch {ep+1}/{epochs} – θ={self.th:.3f}  #boxes={len(self.V)}")

    # -------- Inferenz (vektorisiert) -------------------------------
    @torch.no_grad()
    def predict(self, X: Tensor) -> Tensor:
        X = X.to(self.dev)
        memb = self._memb_batch(X)                   # [N,B]
        return self.cls[memb.argmax(1)]

    def score(self, X: Tensor, y: Tensor) -> float:
        return (self.predict(X) == y.to(self.dev)).float().mean().item()

if __name__ == "__main__":
    from data_utils import load_K_chess_data_splitted, load_Kp_chess_data, load_Kp_chess_data_ord
    Xtr, ytr, Xte, yte = load_K_chess_data_splitted()   # → Tensoren
    #Xtr, ytr, Xte, yte = load_Kp_chess_data()   # → Tensoren

    clf = FMNC(
            gamma      = 1.5,    
            theta0     = 1,    
            theta_min  = 0.2,
            theta_decay= 0.9,
            bound_mode = "max", 
            aggr       = "min",
    )

    clf.fit(Xtr, ytr, epochs=3, shuffle=True)
    print("Test-Acc :", clf.score(Xte, yte))
    # nach dem Training
