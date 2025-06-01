# fmnc_continuous.py
from __future__ import annotations
import torch
from torch import Tensor
from typing import Optional, Literal



class FMNC:
    """Fuzzy-Min-Max-Classifier  –  reine kontinuierliche Features.
       * Online-Lernen (Simpson 1992)
       * kompletter Overlap-Test + Kontraktion
       * θ-Annealing
    """
    # ------------------------------------------------------------
    def __init__(
        self,
        gamma: float        = 0.5,       # Slope γ  (0.2-0.8 üblich)
        theta0: float       = 1.0,       # Start-θ   (max. Box-Kantenlänge)
        theta_min: float    = 0.3,
        theta_decay: float  = 0.95,
        bound_mode: Literal["sum", "max"] = "sum",
        aggr: Literal["min", "mean"]      = "min",
        device: str | torch.device = "cpu" if torch.cuda.is_available() else "cpu",
    ):
        self.V: Optional[Tensor] = None   # [B, D]  lower corners
        self.W: Optional[Tensor] = None   # [B, D]  upper corners
        self.cls: Optional[Tensor] = None # [B]
        
        self.g, self.th = gamma, theta0  # [D]
        self.th_min, self.th_decay = min(self.th,theta_min), theta_decay
        self.bound_mode, self.aggr = bound_mode, aggr
        self.dev = torch.device(device)
#
    def _memb(self, x: Tensor) -> Tensor:
        """Membership jeder Box für x   →  [B]"""
        a = 1 - self.g * torch.clamp(self.V - x, min=0)   # left
        b = 1 - self.g * torch.clamp(x - self.W, min=0)   # right
        m = torch.minimum(a, b)
        return m.amin(1) if self.aggr == "min" else m.mean(1)

    def _span(self, v_new: Tensor, w_new: Tensor) -> float:
        side = w_new - v_new
        return side.sum().item() if self.bound_mode == "sum" else side.max().item()


    def _add_box(self, x: Tensor, y: int):
        self.V = x.clone().unsqueeze(0) if self.V is None else torch.cat([self.V, x.unsqueeze(0)])
        self.W = x.clone().unsqueeze(0) if self.W is None else torch.cat([self.W, x.unsqueeze(0)])
        lbl    = torch.tensor([y], device=self.dev)
        self.cls = lbl if self.cls is None else torch.cat([self.cls, lbl])


    def _contract(self, j: int):
        vj, wj = self.V[j], self.W[j]
        #print(vj.shape, wj.shape)
        for k in range(len(self.V)):
            if self.cls[k] == self.cls[j]:
                continue
            vk, wk = self.V[k].clone(), self.W[k].clone()
            # Überlappung in jeder Dimension?
            inter_low  = torch.maximum(vj, vk)
            inter_high = torch.minimum(wj, wk)
            inter_len  = inter_high - inter_low
            pos        = inter_len > 0
            if pos.sum() < 1:               # exakt 1 Dim?
                #print("no overlap")
                continue
            i = int(pos.nonzero()[0])
            print("i:", i)

            if   vj[i] < vk[i] < wj[i] < wk[i]:
                #print("vj < vk < wj < wk")
                vk[i] = wj[i] = (vk[i] + wj[i]) / 2
            elif vk[i] < vj[i] < wk[i] < wj[i]:
                #print("vk < vj < wk < wj")
                vj[i] = wk[i] = (vj[i] + wk[i]) / 2
            elif vj[i] < vk[i] < wk[i] < wj[i]:
                #print("vj < vk < wk < wj")
                if (wj[i]-vk[i]) > (wk[i]-vj[i]): vj[i] = wk[i]
                else:                           wj[i] = vk[i]
            else:  # vk < vj < wj < wk
                if (wk[i]-vj[i]) > (wj[i]-vk[i]): vk[i] = wj[i]
                else:                            wk[i] = vj[i]

            self.V[k], self.W[k] = vk, wk
            self.V[j], self.W[j] = vj, wj   # (vj/wj wurden evtl. verändert)


    def _learn_one(self, x: Tensor, y: int):
        if self.V is None or (self.cls == y).sum() == 0:
            self._add_box(x, y)
            #print(f"new box: {x}  {x}")
            return
            
        

        m      = self._memb(x)
       # print(f"m = {m}")
        m[self.cls != y] = -1
        j      = int(m.argmax())

        v_new  = torch.minimum(self.V[j], x)
        w_new  = torch.maximum(self.W[j], x)
        
        
        if self._span(v_new, w_new) <= self.th:
            # expand erlaubt
            #print(f"expand: {v_new}  {w_new}")
            self.V[j], self.W[j] = v_new, w_new
            self._contract(j)
        else:
            # neue Box
            #print(f"new box: {v_new}  {w_new}")
            self._add_box(x, y)


    def fit(self, X: Tensor, y: Tensor, epochs: int = 1, shuffle: bool = True):
        X, y = X.to(self.dev), y.to(self.dev)

        for ep in range(epochs):
            idx = torch.randperm(len(X)) if shuffle else torch.arange(len(X))
            print(idx.shape)
            for i in idx:
                self._learn_one(X[i], int(y[i]))
                print(i)
                print(self.V.shape)


            self.th = max(self.th * self.th_decay, self.th_min)
            print(f"epoch {ep+1}/{epochs} – θ={self.th:.3f}  #boxes={len(self.V)}")
            #print(self.V.shape, self.W.shape, self.cls.shape)
            print("Test-Acc :", clf.score(Xte, yte))

    def predict(self, X: Tensor) -> Tensor:
        X = X.to(self.dev)
        out = []
        for x in X:
            m = self._memb(x)
            out.append(int(self.cls[int(m.argmax())]))
        return torch.tensor(out, device=self.dev)

    def score(self, X: Tensor, y: Tensor) -> float:
        return (self.predict(X) == y.to(self.dev)).float().mean().item()
    
 
if __name__ == "__main__":
    from data_utils import load_K_chess_data_splitted, load_Kp_chess_data, load_Kp_chess_data_ord
    #Xtr, ytr, Xte, yte = load_K_chess_data_splitted()   # → Tensoren
    Xtr, ytr, Xte, yte = load_Kp_chess_data_ord()   # → Tensoren
    print(yte.shape)
    #Xtr, Xte = Xtr / 7.0, Xte / 7.0                     # ordinale Skalierung [0,1]
    print(Xtr.shape)
    clf = FMNC(
        gamma        = 0.2,      # weich
        theta0       = 1,     # passt zur Normierung u. max-Mode
        theta_min    = 0.6,      # nicht unter 0.6 fallen lassen
        theta_decay  = 0.97,     # gaaanz langsam
        bound_mode   = "sum",
        aggr         = "min",   # statt "min"
    )
    clf.fit(Xtr, ytr, epochs=1, shuffle=True)
    print("Test-Acc :", clf.score(Xte, yte))
    # nach dem Training





