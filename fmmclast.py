# -----------------------------------------------
# fuzzy_mmc_torch.py  (rev-2025-05-12)
# -----------------------------------------------
from __future__ import annotations
import torch, math
from torch import Tensor
from typing import Optional, Literal
import matplotlib.pyplot as _plt
from matplotlib import patches as _patches
from typing import Sequence, Tuple


class FuzzyMMC_Torch:
    """
    Fuzzy Min-Max Classifier  (Simpson 1992, diskrete Erweiterung)
    – optionales 0-1-Normieren
    – θ-Annealing  (Summe oder Maximum der Kantenlängen)
    – kontrahiert *alle* überlappenden Box-Paare
    – Sonderregeln für One-Hot-/diskrete Attribute
    """

    # ---------- Konstruktor --------------------------------------
    def __init__(
        self,
        gamma: float = .6,               # Steilheit γ   (0.2 … 0.8)
        theta_start: float = .8,         # Anfangs-θ  (≈ 80 % Brettkante)
        theta_min:   float = .4,         # nicht weiter schrumpfen
        theta_decay: float = .95,        # jedes Epoch: θ ← θ·decay
        bound_mode: Literal["sum", "max"] = "sum",
        onehot: bool = False,            # False = echte 0-7 Koordinaten
        groups: int = 3, group_size: int = 8,
        max_boxes_per_hot: int = 4,      # Lockerung für seltene Felder
        membership_aggr: Literal["min", "mean"] = "min",
        normalize: bool = True,
        device: str | torch.device = "cpu",
    ):
        self.g      = gamma
        self.th     = theta_start
        self.th_min = theta_min
        self.decay  = theta_decay
        self.bound_mode = bound_mode

        self.onehot = onehot
        self.G, self.GS = groups, group_size
        self.max_boxes_per_hot = max_boxes_per_hot
        self.aggr = membership_aggr

        self.device = torch.device(device)
        self.norm   = normalize
        self.xmin: Optional[Tensor] = None
        self.xmax: Optional[Tensor] = None

        self.boxes:  Optional[Tensor] = None   # [B,2,D]
        self.labels: Optional[Tensor] = None   # [B]

    # ---------- Hilfen -------------------------------------------
    def _maybe_norm(self, X: Tensor) -> Tensor:
        if not self.norm:
            return X
        if self.xmin is None:                 # fit-Phase einmalig merken
            self.xmin = X.min(0).values
            self.xmax = X.max(0).values
        return (X - self.xmin) / (self.xmax - self.xmin + 1e-9)

    def _membership(self, p: Tensor) -> Tensor:
        v, w = self.boxes[:, 0], self.boxes[:, 1]
        left  = 1 - self.g * torch.clamp(v - p, min=0)
        right = 1 - self.g * torch.clamp(p - w, min=0)
        m = torch.minimum(left, right)
        return m.amin(1) if self.aggr == "min" else m.mean(1)

    # ---------- One-Hot-Erkennung --------------------------------
    def _hot_groups(self, x: Tensor) -> list[int]:
        if not self.onehot:
            return []
        hot = []
        for g in range(self.G):
            seg = x[g*self.GS:(g+1)*self.GS]
            if (seg == 1).sum() != 1:
                return []                    # keine echte One-Hot-Probe
            hot.append(int((seg == 1).nonzero()))
        return hot

    # ---------- Box hinzufügen -----------------------------------
    def _add_box(self, x: Tensor, y: int):
        box = x.repeat(2,1).unsqueeze(0)     # [1,2,D]  (min=max=x)
        self.boxes  = box if self.boxes is None else torch.cat([self.boxes, box])
        lab  = torch.tensor([y], device=self.device)
        self.labels = lab if self.labels is None else torch.cat([self.labels, lab])

    # ---------- Kontraktion (alle 1-D-Überlappungen) --------------
    def _contract(self, idx_new: int):
        vj, wj = self.boxes[idx_new]
        mask = self.labels != self.labels[idx_new]
        if not mask.any():
            return
        vk, wk = self.boxes[mask,0], self.boxes[mask,1]

        ol_low  = torch.maximum(vj, vk)
        ol_high = torch.minimum(wj, wk)
        inter   = ol_high - ol_low
        pos     = inter > 0
        oned    = pos.sum(1) == 1

        for k_loc in oned.nonzero(as_tuple=True)[0]:
            d = pos[k_loc].nonzero(as_tuple=True)[0].item()
            k_glob = mask.nonzero(as_tuple=True)[0][k_loc]
            vk_i, wk_i = vk[k_loc,d], wk[k_loc,d]

            # 4 Fälle (Simpson 92)
            if   vj[d] < vk_i < wj[d] < wk_i:
                vk_i = wj[d] = (vk_i + wj[d]) / 2
            elif vk_i < vj[d] < wk_i < wj[d]:
                vj[d] = wk_i = (vj[d] + wk_i) / 2
            elif vj[d] < vk_i < wk_i < wj[d]:
                if (wj[d]-vk_i) > (wk_i-vj[d]): vj[d] = wk_i
                else:                           wj[d] = vk_i
            else:  # vk_i < vj < wj < wk_i
                if (wk_i-vj[d]) > (wj[d]-vk_i): vk_i = wj[d]
                else:                           wk_i = vj[d]

            self.boxes[idx_new,0,d] = vj[d]
            self.boxes[idx_new,1,d] = wj[d]
            self.boxes[k_glob,0,d]  = vk_i
            self.boxes[k_glob,1,d]  = wk_i

    # ---------- Online-Training eines Samples --------------------
    def _train_one(self, x: Tensor, y: int):
        x = self._maybe_norm(x)
        hot = self._hot_groups(x)

        # Hot-Limit → schon genug Boxen für dieses Feld?
        if hot and self.boxes is not None:
            same = self.labels == y
            for g,pos in enumerate(hot):
                d = g*self.GS + pos
                mask = (same &
                        (self.boxes[:,0,d]==1) &
                        (self.boxes[:,1,d]==1) &
                        (self.boxes.sum((1,2))==1))
                if mask.sum() >= self.max_boxes_per_hot:
                    return                      # Sample abgedeckt

        # neue Klasse?
        if self.labels is None or (self.labels==y).sum()==0:
            self._add_box(x,y); return

        # Gewinner-Box
        mem = self._membership(x)
        mem[self.labels!=y] = -1
        idx = int(mem.argmax())
        v_old, w_old = self.boxes[idx]

        # One-Hot-Check
        if hot:
            for g,pos in enumerate(hot):
                d = g*self.GS + pos
                if not (v_old[d]==1 and w_old[d]==1):
                    idx = None; break

        # Expansion möglich?
        if idx is not None:
            v_new = torch.minimum(v_old, x)
            w_new = torch.maximum(w_old, x)
            span = (w_new - v_new).sum() if self.bound_mode=="sum" else (w_new-v_new).amax()
            if span <= self.th:
                self.boxes[idx] = torch.stack([v_new, w_new])
                if not hot:
                    self._contract(idx)
                return

        # sonst neue Box
        self._add_box(x,y)
        if not hot:
            self._contract(len(self.boxes)-1)

    # ---------- Public API  --------------------------------------
    def fit(self, X: Tensor, y: Tensor, epochs:int=1, shuffle:bool=True):
        X, y = X.to(self.device), y.to(self.device)
        for ep in range(epochs):
            idx = torch.randperm(len(X)) if shuffle else torch.arange(len(X))
            for i in idx:
                self._train_one(X[i], int(y[i]))
            self.th = max(self.th*self.decay, self.th_min)
            
            
            print(f"epoch {ep+1}/{epochs} – θ={self.th:.3f}  #boxes={len(self.boxes)}")
            print("Test-Acc :", clf.score(Xte, yte))
            print("größte Box-Kantenlänge:", (clf.boxes[:,1]-clf.boxes[:,0]).max().item())

    def predict(self, X: Tensor) -> Tensor:
        X = self._maybe_norm(X.to(self.device))
        preds = []
        for x in X:
            m = self._membership(x)
            preds.append(int(self.labels[int(m.argmax())]))
        return torch.tensor(preds, device=self.device)

    def score(self, X: Tensor, y: Tensor) -> float:
        return (self.predict(X)==y.to(self.device)).float().mean().item()
    
    def plot_boxes(
        self,
        dims: Tuple[int,int] = (0,1),
        sample_points: Optional[Tuple[Tensor,Tensor]] = None,
        figsize: Tuple[int,int] = (6,5)
    ):

        if self.boxes is None:
            raise RuntimeError("Model not fitted – no boxes to draw.")

        i,j = dims
        _plt.figure(figsize=figsize)
        ax = _plt.gca()
        
        # optionale Punkte
        if sample_points is not None:
            X, y = sample_points
            X = self._maybe_norm(X.to(self.device)).cpu()
            y = y.cpu().numpy()
            ax.scatter(X[:,i], X[:,j], c=y, s=8, alpha=.3, cmap="tab20")

        # Box-Umrisse
        for (v,w), lbl in zip(self.boxes.cpu(), self.labels.cpu()):
            rect = _patches.Rectangle(
                (v[i], v[j]), (w[i]-v[i]), (w[j]-v[j]),
                fill=False, lw=1.2, edgecolor=_plt.cm.tab20(lbl % 20)
            )
            ax.add_patch(rect)
        
        ax.set_xlabel(f"feature {i}")
        ax.set_ylabel(f"feature {j}")
        ax.set_title("Fuzzy-MMC – Hyperbox-Umrisse")
        _plt.tight_layout()
        _plt.show()


    def plot_membership_heat(
        self,
        dims: Tuple[int,int] = (0,1),
        res: int = 200,
        cmap: str = "viridis"
    ):
        """
        Decision-Surface: zeigt für jedes Gitter-Pixel die maximale Membership-
        Stärke (0…1) der *gewinnenden* Box.
        """
        if self.boxes is None:
            raise RuntimeError("Model not fitted.")

        i,j = dims
        # Gitterbereich aus Box-Extrema ableiten
        lo = self.boxes[:,0,[i,j]].min(0).values.cpu()
        hi = self.boxes[:,1,[i,j]].max(0).values.cpu()
        xi = torch.linspace(lo[0], hi[0], res)
        yi = torch.linspace(lo[1], hi[1], res)
        XX, YY = torch.meshgrid(xi, yi, indexing="ij")
        grid = torch.stack([XX.flatten(), YY.flatten()], dim=1).to(self.device)

        # Dummy-Sample mit allen D-Dimensionen (andere dims = 0.5)
        D = self.boxes.shape[2]
        full = torch.full((grid.shape[0], D), 0.5, device=self.device)
        full[:,i] = grid[:,0]; full[:,j] = grid[:,1]
        
        with torch.no_grad():
            m = self._membership_batch(full,           # [N,D]
                                self.boxes,     # [B,2,D]
                                self.labels,
                                self.g,
                                self.aggr)      # →  [B, N]
            best = m.max(0).values.reshape(res,res).cpu()


        # with torch.no_grad():
        #     m = self._membership(full)           # [B, N]
        #     best = m.max(0).values.reshape(res,res).cpu()

        _plt.figure(figsize=(6,5))
        _plt.imshow(best.T, extent=(lo[0],hi[0],lo[1],hi[1]),
                    origin="lower", cmap=cmap, aspect="auto", vmin=0, vmax=1)
        _plt.colorbar(label="max. Membership")
        _plt.xlabel(f"feature {i}"); _plt.ylabel(f"feature {j}")
        _plt.title("Decision-Surface (max Membership)")
        _plt.tight_layout(); _plt.show()


    def explain_sample(
        self,
        x: Tensor,
        k: int = 10
    ):
        """
        Lokale Erklärung – zeigt die k stärksten Boxen und deren Membership.
        Gibt zugleich ihre min/max-Koordinaten in tabellarischer Form aus.
        """
        x = self._maybe_norm(x.to(self.device))
        m = self._membership(x)
        if len(m)==0:
            print("Keine Boxen im Modell.")
            return
        topk = torch.topk(m, k=min(k,len(m)))
        idxs, vals = topk.indices.cpu(), topk.values.cpu()

        # Balkenplot
        _plt.figure(figsize=(4,2.5))
        _plt.barh(range(len(idxs)), vals, color="steelblue")
        _plt.gca().invert_yaxis()
        _plt.xlabel("Membership")
        _plt.yticks(range(len(idxs)), [f"B{int(i)}" for i in idxs])
        _plt.title("Top-k Memberships für Muster x")
        _plt.tight_layout(); _plt.show()

        # Tabelle der Box-Grenzen
        print("\nTop-k Hyperbox-Grenzen:")
        for rank,(i,val) in enumerate(zip(idxs, vals), 1):
            v,w = self.boxes[i]; lab = int(self.labels[i])
            print(f"{rank}. Box {i}  (class {lab})  m={val:.3f}")
            print(f"   min={v.cpu().numpy()}\n   max={w.cpu().numpy()}")
        
    



# playground.py
import torch
from data_utils import load_K_chess_data_splitted, load_iris_data, load_Kp_chess_data, load_heart_data, load_abalon_data, load_Poker_data, load_Kp_chess_data_ord
# Xtr: 6-D float (0…7)  -> /7   |   ytr: 0…17
if __name__ == "__main__":
    #Xtr, ytr, Xte, yte = load_K_chess_data_splitted()
    #Xtr, ytr, Xte, yte = load_iris_data()
    #Xtr, ytr, Xte, yte = load_heart_data()
    #Xtr, ytr, Xte, yte,_ = load_abalon_data()
    Xtr, ytr, Xte, yte = load_Kp_chess_data()
    #Xtr, ytr, Xte, yte = load_Poker_data()
    
    
    #Xtr, Xte = Xtr/7.0, Xte/7.0     # *geordnete* Normierung!

    clf = FuzzyMMC_Torch(
            gamma=1.5, theta_start=.4, theta_decay=.97, theta_min=.3,
            bound_mode="max", normalize=False, onehot=True)
    clf.fit(Xtr, ytr, epochs=1)
    print("größte Box-Kantenlänge:",
      (clf.boxes[:,1]-clf.boxes[:,0]).max().item())
    print("Test-Acc :", clf.score(Xte, yte))
    clf.plot_boxes(dims=(3,4), sample_points=(Xtr[:200], ytr[:200]))
    #clf.plot_membership_heat(dims=(0,1), res=300)
    clf.explain_sample(Xte[0], k=5)
