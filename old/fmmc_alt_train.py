#!/usr/bin/env python3
# ---------------------------------------------------------------
# fmnc_alt_train.py
# ---------------------------------------------------------------
"""
Fuzzy-Min-Max-Classifier  (continuous features)
 * klassische Expand/Contract-Logik
 * γ (slope)   & θ (box-limit) werden per Autograd gelernt
 * Alternating-Training:  Box-Rebuild ↔ Parameter-Update
"""

from __future__ import annotations
import math, torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Optional, Literal, Tuple

# ---------------------------------------------------------------
class FMNC(nn.Module):
    def __init__(
        self,
        gamma0: float = 0.5,
        theta0: float = 1.0,
        bound_mode: Literal["sum", "max"] = "sum",
        aggr: Literal["min", "mean"] = "min",
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.dev = torch.device(device)

        # --- lernbare Roh-Parameter (Softplus > 0) ----------------
        self.log_g  = nn.Parameter(torch.tensor(_inv_softplus(gamma0),
                                                dtype=torch.float32,
                                                device=self.dev))
        self.log_th = nn.Parameter(torch.tensor(_inv_softplus(theta0),
                                                dtype=torch.float32,
                                                device=self.dev))

        # --- Box-Speicher ----------------------------------------
        self.register_buffer("V",   None)   # [B,D] lower
        self.register_buffer("W",   None)   # [B,D] upper
        self.register_buffer("cls", None)   # [B]

        self.bound_mode, self.aggr = bound_mode, aggr

    # --------- abgeleitete positiven Parameter -------------------
    @property
    def g(self)  -> Tensor:  return F.softplus(self.log_g)
    @property
    def theta(self) -> Tensor:  return F.softplus(self.log_th)

    # ------------------------------------------------------------
    def reset_boxes(self):
        self.V = self.W = self.cls = None

    # ------------------------------------------------------------
    def _memb(self, x: Tensor) -> Tensor:
        a = 1 - self.g * torch.clamp(self.V - x, min=0)
        b = 1 - self.g * torch.clamp(x - self.W, min=0)
        m = torch.minimum(a, b)
        return m.amin(1) if self.aggr == "min" else m.mean(1)

    def _span(self, v_new: Tensor, w_new: Tensor) -> float:
        side = w_new - v_new
        return side.sum().item() if self.bound_mode == "sum" else side.max().item()

    def _add_box(self, x: Tensor, y: int):
        self.V  = x.clone().unsqueeze(0) if self.V  is None else torch.cat([self.V,  x.unsqueeze(0)])
        self.W  = x.clone().unsqueeze(0) if self.W  is None else torch.cat([self.W,  x.unsqueeze(0)])
        lbl     = torch.tensor([y], device=self.dev)
        self.cls = lbl                      if self.cls is None else torch.cat([self.cls, lbl])

    def _contract(self, j: int):
        vj, wj = self.V[j], self.W[j]
        for k in range(len(self.V)):
            if self.cls[k] == self.cls[j]:
                continue
            vk, wk = self.V[k].clone(), self.W[k].clone()
            inter_low  = torch.maximum(vj, vk)
            inter_high = torch.minimum(wj, wk)
            inter_len  = inter_high - inter_low
            if not (inter_len > 0).all():
                continue
            i = int((inter_len > 0).nonzero()[0])
            if   vj[i] < vk[i] < wj[i] < wk[i]:
                vk[i] = wj[i] = (vk[i] + wj[i]) / 2
            elif vk[i] < vj[i] < wk[i] < wj[i]:
                vj[i] = wk[i] = (vj[i] + wk[i]) / 2
            elif vj[i] < vk[i] < wk[i] < wj[i]:
                if (wj[i]-vk[i]) > (wk[i]-vj[i]): vj[i] = wk[i]
                else:                           wj[i] = vk[i]
            else:  # vk < vj < wj < wk
                if (wk[i]-vj[i]) > (wj[i]-vk[i]): vk[i] = wj[i]
                else:                            wk[i] = vj[i]
            self.V[k], self.W[k] = vk, wk
        self.V[j], self.W[j] = vj, wj

    def _learn_one(self, x: Tensor, y: int):
        if self.V is None or (self.cls == y).sum() == 0:
            self._add_box(x, y)
            return
        m = self._memb(x)
        m[self.cls != y] = -1
        j = int(m.argmax())
        v_new = torch.minimum(self.V[j], x)
        w_new = torch.maximum(self.W[j], x)
        if self._span(v_new, w_new) <= self.theta.item():
            self.V[j], self.W[j] = v_new, w_new
            self._contract(j)
        else:
            self._add_box(x, y)

    # ------------------------------------------------------------
    def fit_boxes(self,
                  X: Tensor, y: Tensor,
                  shuffle: bool = True) -> None:
        """ein einziger Online-Pass über die Daten"""
        X, y = X.to(self.dev), y.to(self.dev)
        idx = torch.randperm(len(X)) if shuffle else torch.arange(len(X))
        for i in idx:
            self._learn_one(X[i], int(y[i]))

    # ------------------------------------------------------------
    def forward(self, X: Tensor) -> Tensor:
        """Membership-Matrix  [N,B]  –  benutzt γ, V, W"""
        return torch.stack([self._memb(x) for x in X.to(self.dev)])

    # ------------------------------------------------------------
    def predict(self, X: Tensor) -> Tensor:
        mem = self(X)                     # [N,B]
        idx = mem.argmax(1)
        return self.cls[idx]

    def score(self, X: Tensor, y: Tensor) -> float:
        return (self.predict(X) == y.to(self.dev)).float().mean().item()

# ===============================================================
def alternating_train(model: FMNC,
                      Xtr: Tensor, ytr: Tensor,
                      Xte: Tensor, yte: Tensor,
                      outer_epochs: int = 30,
                      lr: float = 3e-3) -> None:
    opt = torch.optim.Adam([model.log_g, model.log_th], lr=lr)
    ce  = nn.CrossEntropyLoss()

    Xtr, ytr = Xtr.to(model.dev), ytr.to(model.dev)
    for ep in range(outer_epochs):
        # 1) Boxen neu erstellen
        model.reset_boxes()
        model.fit_boxes(Xtr, ytr, shuffle=True)

        # 2) γ & θ-Gradient-Step (Boxen fix)
        opt.zero_grad()
        mem = model(Xtr)                                 # [N,B]
        C   = int(model.cls.max().item() + 1)
        logits = torch.full((len(Xtr), C), -1e9, device=model.dev)
        for c in range(C):
            logits[:, c] = torch.logsumexp(mem[:, model.cls == c], dim=1)
        loss = ce(logits, ytr.long())
        loss.backward()
        opt.step()

        print(f"EP {ep+1:02d} | γ={model.g.item():.3f}  "
              f"θ={model.theta.item():.3f} | "
              f"train-loss={loss.item():.4f} | "
              f"val-acc={model.score(Xte, yte):.3%} | "
              f"#boxes={len(model.V)}")

# ---------------------------------------------------------------
def _inv_softplus(x: float) -> float:
    """x ↦ y  s.t.  softplus(y) = x   (numerisch stabil)"""
    return math.log(math.exp(x) - 1.0 + 1e-8)


# ---------------------------------------------------------------
if __name__ == "__main__":
    # ----  Daten laden  -----------------------------------------
    # Ersetze load_dummy_data() durch deinen Chess-Loader
    from data_utils import load_iris_data, load_Kp_chess_data_ord
    #Xtr, ytr, Xte, yte = load_iris_data()
    Xtr, ytr, Xte, yte = load_Kp_chess_data_ord()

    # ----  Modell & Training  -----------------------------------
    clf = FMNC(gamma0=1.2, theta0=4.0,
               bound_mode="max", aggr="mean",
               device="cuda" if torch.cuda.is_available() else "cpu")

    alternating_train(clf, Xtr, ytr, Xte, yte,
                      outer_epochs=30, lr=3e-3)

    print("\nFinal-Val-Accuracy:",
          f"{clf.score(Xte, yte):.3%}",
          "|  #Boxes:", len(clf.V))
