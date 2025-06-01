# fmnn_sklearn.py
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
from jkll import FuzzyMMC_Torch      # ← dein FMNN-Code

class FMNN_SK(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 gamma=0.8, theta_start=0.6, theta_min=0.3, theta_decay=0.95,
                 bound_mode="max", epochs=5, shuffle=True,
                 onehot=False, device="cpu"):
        self.gamma, self.theta_start, self.theta_min = gamma, theta_start, theta_min
        self.theta_decay, self.bound_mode = theta_decay, bound_mode
        self.epochs, self.shuffle = epochs, shuffle
        self.onehot, self.device  = onehot, device
        self._fmnn = None                         # wird im fit() erzeugt

    # ---------- sklearn API ----------
    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long,   device=self.device)

        self._fmnn = FuzzyMMC_Torch(
            gamma=self.gamma,
            theta_start=self.theta_start,
            theta_min=self.theta_min,
            theta_decay=self.theta_decay,
            bound_mode=self.bound_mode,
            onehot=self.onehot,
            device=self.device
        )
        self._fmnn.fit(X_t, y_t, epochs=self.epochs, shuffle=self.shuffle)
        return self

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        return self._fmnn.predict(X_t).cpu().numpy()

    def score(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long,   device=self.device)
        return self._fmnn.score(X_t, y_t)
