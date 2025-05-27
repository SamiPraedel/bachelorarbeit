# fmnc_continuous.py
from __future__ import annotations
import torch, numpy as np
from torch import Tensor
from typing import Optional, Literal
from sklearn.metrics import matthews_corrcoef


class FMNC:
    def __init__(self,
                 gamma: float = 0.6,
                 theta0: float = 1.0,
                 theta_min: float = 0.6,
                 theta_decay: float = 0.97,
                 bound_mode: Literal["sum", "max"] = "sum",
                 aggr: Literal["min", "mean"] = "min",
                 m_min: float = 0.8,
                 device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 256):
        self.g, self.th = gamma, theta0
        self.th_min, self.th_decay = theta_min, theta_decay
        self.bound_mode, self.aggr = bound_mode, aggr
        self.dev = torch.device(device)
        self.bs = batch_size
        self.m_min = m_min
        self.V = self.W = self.cls = None   # Box-Container

    # ---------- NEU: k-Means-Seeding -------------------------------
    def seed_boxes_kmeans(self, X: Tensor, y: Tensor, k: int = 3, random_state: int = 42):
        """Erzeuge pro Klasse k Start-Boxen (V = W = Cluster-Center)."""
        from sklearn.cluster import KMeans

        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()

        for c in np.unique(y_np):
            X_c = X_np[y_np == c]
            k_eff = min(k, len(X_c))          # nicht mehr Cluster als Punkte
            km = KMeans(n_clusters=k_eff, n_init="auto", random_state=random_state)
            km.fit(X_c)

            centers = torch.tensor(km.cluster_centers_,
                                   dtype=X.dtype, device=self.dev)
            for center in centers:
                self._add_box(center, int(c))

    # ---------------------------------------------------------------
    def _memb_batch(self, X: Tensor) -> Tensor:
        V, W = self.V, self.W
        left  = 1 - self.g * torch.clamp(V.unsqueeze(0) - X.unsqueeze(1), min=0)
        right = 1 - self.g * torch.clamp(X.unsqueeze(1) - W.unsqueeze(0), min=0)
        m = torch.minimum(left, right)
        return m.amin(2) if self.aggr == "min" else m.mean(2)

    def _span(self, v: Tensor, w: Tensor) -> float:
        side = w - v
        return side.sum().item() if self.bound_mode == "sum" else side.max().item()

    def _add_box(self, x: Tensor, y: int):
        self.V  = x.unsqueeze(0) if self.V  is None else torch.cat([self.V,  x.unsqueeze(0)])
        self.W  = x.unsqueeze(0) if self.W  is None else torch.cat([self.W,  x.unsqueeze(0)])
        lbl     = torch.tensor([y], device=self.dev)
        self.cls = lbl            if self.cls is None else torch.cat([self.cls, lbl])

    # --------- Simpson-Contract (1. Overlap-Dim) -------------------
    def _contract_simpson(self, j: int):
        mask = self.cls != self.cls[j]
        if not mask.any(): return
        V_oth, W_oth = self.V[mask], self.W[mask]

        inter_low  = torch.maximum(V_oth, self.V[j])
        inter_high = torch.minimum(W_oth, self.W[j])
        inter_len  = inter_high - inter_low
        overlap = (inter_len > 0).all(1)

        for k_sub in overlap.nonzero(as_tuple=False).flatten():
            k = torch.arange(len(self.V), device=self.dev)[mask][k_sub].item()
            vk, wk = self.V[k].clone(), self.W[k].clone()
            i = int((inter_len[k_sub] > 0).nonzero()[0])   # erste Dim
            vj, wj = self.V[j, i], self.W[j, i]

            if   vj < vk[i] < wj < wk[i]:
                vk[i] = wj = (vk[i] + wj) / 2
            elif vk[i] < vj < wk[i] < wj:
                vj = wk[i] = (vj + wk[i]) / 2
            elif vj < vk[i] < wk[i] < wj:
                if (wj - vk[i]) > (wk[i] - vj): vj = wk[i]
                else:                           wj = vk[i]
            else:  # vk < vj < wj < wk
                if (wk[i] - vj) > (wj - vk[i]): vk[i] = wj
                else:                           wk[i] = vj

            self.V[k, i], self.W[k, i] = vk[i], wk[i]
            self.V[j, i], self.W[j, i] = vj, wj
            
    def _contract(self, j: int):
        mask = self.cls != self.cls[j]
        if not mask.any(): return
        V_oth, W_oth = self.V[mask], self.W[mask]          # [K,D]

        inter_low  = torch.maximum(V_oth, self.V[j])
        inter_high = torch.minimum(W_oth, self.W[j])
        inter_len  = inter_high - inter_low                # [K,D]
        overlap    = (inter_len > 0).all(1)                # [K]

        for k_sub in overlap.nonzero(as_tuple=False).flatten():
            k = torch.arange(len(self.V), device=self.dev)[mask][k_sub].item()
            for i in (inter_len[k_sub] > 0).nonzero(as_tuple=False).flatten():
                vj, wj = self.V[j, i], self.W[j, i]
                vk, wk = self.V[k, i], self.W[k, i]

                if   vj < vk < wj < wk:
                    mid = (vk + wj) / 2
                    vk, wj = mid, mid
                elif vk < vj < wk < wj:
                    mid = (vj + wk) / 2
                    vj, wk = mid, mid
                elif vj < vk < wk < wj:
                    if (wj - vk) > (wk - vj): vj = wk
                    else:                     wj = vk
                else:  # vk < vj < wj < wk
                    if (wk - vj) > (wj - vk): vk = wj
                    else:                     wk = vj

                self.V[j, i], self.W[j, i] = vj, wj
                self.V[k, i], self.W[k, i] = vk, wk
    
    def fit(self, X: Tensor, y: Tensor, epochs: int = 3, shuffle: bool = True):
        X, y = X.to(self.dev), y.to(self.dev)
        for ep in range(epochs):
            idx = torch.randperm(len(X)) if shuffle else torch.arange(len(X))
            Xs, ys = X[idx], y[idx]

            for s in range(0, len(X), self.bs):
                xb, yb = Xs[s:s+self.bs], ys[s:s+self.bs]

                if self.V is None:                       # erste Batch
                    for x_i, y_i in zip(xb, yb):
                        self._add_box(x_i, int(y_i))
                    continue

                B = self.V.shape[0]
                memb = self._memb_batch(xb)              # [Bsz,B]

                for i in range(len(xb)):
                    x_i, y_i = xb[i], int(yb[i])

                    # -------- MEMBERSHIP-SCHWELLE -----------------
                    m = memb[i][:B].clone()
                    m[self.cls[:B] != y_i] = -1
                    j = int(m.argmax())
                    if m[j] < self.m_min:                # <<< NEU
                        self._add_box(x_i, y_i)
                        continue
                    # ------------------------------------------------

                    v_new = torch.minimum(self.V[j], x_i)
                    w_new = torch.maximum(self.W[j], x_i)

                    if self._span(v_new, w_new) <= self.th:
                        self.V[j], self.W[j] = v_new, w_new
                        self._contract(j)
                    else:
                        self._add_box(x_i, y_i)
            


            print(f"epoch {ep+1}/{epochs} – θ={self.th:.3f}  #boxes={len(self.V)}")
            self.th = max(self.th * self.th_decay, self.th_min)

    # -------- Inferenz --------------------------------------------
    @torch.no_grad()
    def predict(self, X: Tensor) -> Tensor:
        X = X.to(self.dev)
        memb = self._memb_batch(X)
        return self.cls[memb.argmax(1)]

    def score(self, X: Tensor, y: Tensor) -> float:
        return (self.predict(X) == y.to(self.dev)).float().mean().item()

    @torch.no_grad()
    def mcc_score(self, X: Tensor, y_true: Tensor) -> float:
        """Calculates the Matthews Correlation Coefficient for the given data."""
        X = X.to(self.dev)
        y_true_cpu = y_true.cpu().numpy() # Ensure y_true is on CPU for sklearn

        # self.predict will raise a TypeError if self.cls is None (untrained model)
        y_pred = self.predict(X)
        y_pred_cpu = y_pred.cpu().numpy()

        # sklearn's matthews_corrcoef handles cases where MCC is undefined by returning 0.0
        mcc = matthews_corrcoef(y_true_cpu, y_pred_cpu)
        return float(mcc)

# ---------------------------------------------------------------
if __name__ == "__main__":
    from data_utils import load_K_chess_data_splitted, load_Kp_chess_data_ord

    #Xtr, ytr, Xte, yte = load_K_chess_data_splitted()
    Xtr, ytr, Xte, yte = load_Kp_chess_data_ord()
    print(Xtr.shape, ytr.shape)
    #1.2, 1.2, 0.4, 0.9, sum, mean k=300 -> 58%, 5400 boxen

    clf = FMNC(
        gamma      = 1.7, 
        theta0     = 2.5,
        theta_min  = 0.1,
        theta_decay= 0.1,
        bound_mode = "sum",
        aggr       = "mean",
        m_min     = 0.8,
    )

    clf.seed_boxes_kmeans(Xtr, ytr, k=3)   # 3 Start-Boxen je Klasse

-
    clf.fit(Xtr, ytr, epochs=1, shuffle=True)

    print("Test-MCC:", clf.mcc_score(Xte, yte))
    print("Test-Acc:", clf.score(Xte, yte))
    print("#Boxes :", len(clf.V))
