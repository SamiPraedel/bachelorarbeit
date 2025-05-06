# data_utils.py

import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
from ucimlrepo import fetch_ucirepo
import pandas as pd
from scipy import stats
from data.openD import loadK, loadKP, loadPoker
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.cluster import KMeans
import umap.umap_ as umap

# data_utils.py  (Ergänzungen / Ersatz vorhandener Funktionen)

import numpy as np, pandas as pd, torch
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, LabelEncoder
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import load_iris

# ---------------------------------------------------------
def _minmax_np(X_df: pd.DataFrame) -> np.ndarray:
    return MinMaxScaler().fit_transform(X_df).astype(np.float32)

# ---------- 1) Chess ‑ King & Rook vs King  ----------------
def get_chessK_full():
    ds  = fetch_ucirepo(id=23)                         # UCI ID 23
    Xdf = ds.data.features.copy()
    y   = ds.data.targets.squeeze()

    enc = OrdinalEncoder()
    Xdf[["white-king-file",
         "white-rook-file",
         "black-king-file"]] = enc.fit_transform(
             Xdf[["white-king-file",
                  "white-rook-file",
                  "black-king-file"]])

    X_np = _minmax_np(Xdf)
    y_np = LabelEncoder().fit_transform(y.values.ravel())
    return X_np, y_np

# ---------- 2) Chess ‑ King & Rook vs King + Pawn ----------
def get_chessKp_full():
    ds  = fetch_ucirepo(id=22)                         # UCI ID 22
    Xdf = ds.data.features.copy()
    y   = ds.data.targets.squeeze()

    Xdf[:] = OrdinalEncoder().fit_transform(Xdf)
    X_np   = _minmax_np(Xdf)
    y_np   = LabelEncoder().fit_transform(y)
    return X_np, y_np

# ---------------- 3) Abalone (age groups) ------------------
def get_abalone_full():
    ds  = fetch_ucirepo(id=1)                          # UCI ID 1
    Xdf = ds.data.features.copy()
    rings = ds.data.targets.squeeze()

    # Target – 3 Klassen: 0–8 ▶ 0, 9–11 ▶ 1, >=12 ▶ 2
    age_grp = pd.cut(rings, bins=[-1, 8, 11, np.inf], labels=[0,1,2]).astype(int)

    # Nominal‑Spalte "Sex" → One‑Hot
    Xdf = pd.get_dummies(Xdf, columns=["Sex"])
    X_np = _minmax_np(Xdf)
    y_np = age_grp.to_numpy(dtype=int)
    return X_np, y_np

# -------------------- 4) Heart Disease ---------------------
def get_heart_full():
    ds  = fetch_ucirepo(id=45)                         # Cleveland heart
    Xdf = ds.data.features.copy()
    y   = ds.data.targets.squeeze().astype(int)

    # Kategoriale nach One‑Hot
    cat_cols = Xdf.select_dtypes(include="object").columns
    if len(cat_cols):
        Xdf = pd.get_dummies(Xdf, columns=cat_cols)

    X_np = _minmax_np(Xdf)
    y_np = y.to_numpy() if isinstance(y, pd.Series) else y
    return X_np, y_np

# -------------------- 5) Iris  (Baseline) ------------------
def get_iris_full():
    iris = load_iris()
    X_np = _minmax_np(pd.DataFrame(iris.data))
    y_np = iris.target.astype(int)
    return X_np, y_np
