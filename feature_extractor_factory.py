"""
feature_extractor_factory.py
----------------------------
Helpers that (1) pick a neuro-fuzzy model, (2) train it once on a small
labelled subset, and (3) return ready-to-use rule- and MF-feature
matrices for the SSL pipeline.

Returned dict keys
------------------
model            – the trained torch model  
rule_train       – ℓ2-normalised rule activations for X_train  [Nₜ, R]  
rule_test        –               …            for X_test   [Nₑ, R]  
mf_train         – ℓ2-normalised MF features   for X_train  [Nₜ, d·M]  
mf_test          –               …            for X_test   [Nₑ, d·M]  
labeled_indices  – np.array of indices that were truly labelled
"""

from typing import Dict, Any
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------
#  model classes & trainers you already have in the repo
# ---------------------------------------------------------------------
from anfis_nonHyb import NoHybridANFIS
from anfis_hybrid import HybridANFIS
from PopFnn       import POPFNN

from trainAnfis   import train_anfis_noHyb, train_anfis_hybrid
from anfisHelper  import initialize_mfs_with_kmeans
# ---------------------------------------------------------------------
from trainPF       import train_popfnn

# FMNC: Fuzzy Min-Max Classifier (kmFmmc)
from kmFmmc import FMNC
# ---------------------------------------------------------------------


# ---------- individual trainer wrappers --------------------------------
def _train_nohyb(
    X_l, y_l, *, X_train_full,
    input_dim, num_classes,
    device="cpu", num_mfs=4, max_rules=1000,
    seed=42, lr=1e-2, epochs=400
):
    model = NoHybridANFIS(input_dim, num_classes,
                          num_mfs, max_rules, seed).to(device)

    train_anfis_noHyb(model,
                      X_l.to(device), y_l.to(device),
                      X_train_full,
                      num_epochs=epochs, lr=lr)
    return model


def _train_hybrid(
    X_l, y_l, *, X_train_full,
    input_dim, num_classes,
    device="cpu", num_mfs=4, max_rules=1000,
    seed=42, lr=1e-2, epochs=400
):
    model = HybridANFIS(input_dim, num_classes,
                        num_mfs, max_rules, seed=seed).to(device)
    initialize_mfs_with_kmeans(model, X_train_full)
    train_anfis_hybrid(model,
                    X_l.to(device), y_l.to(device),
                    X_train_full,
                    num_epochs=epochs, lr=lr)
    return model


def _train_popfnn(
    X_l, y_l, *, X_train_full,
    input_dim, num_classes, num_mfs=4,
    device="cpu",
    epochs=400, lr=1e-2, seed=42
):
    torch.manual_seed(seed)
    model = POPFNN(d=input_dim,
                   C=num_classes, num_mfs=num_mfs).to(device)
    
    model = train_popfnn(model, X_l.to(device), y_l.to(device),
                 epochs=epochs, lr=lr)

    return model


# ---------- FMNC trainer wrapper --------------------------------------
def _train_fmmc(
    X_l, y_l, *, X_train_full,
    input_dim, num_classes,
    device="cpu",
    gamma=1.7, theta0=2.5,
    theta_min=0.1, theta_decay=0.1,
    bound_mode="sum", aggr="mean",
    m_min=0.8, max_epochs=100
):
    """
    Train a Fuzzy‑Min‑Max classifier (FMNC) on the labelled subset.
    Only labelled data is used for hyper‑box creation.
    """
    # FMNC works purely on NumPy ‑ keep data on CPU
    model = FMNC(gamma=gamma, theta0=theta0,
                 theta_min=theta_min, theta_decay=theta_decay,
                 bound_mode=bound_mode, aggr=aggr,
                 m_min=m_min)
    model.seed_boxes_kmeans(X_l.cpu().numpy(), y_l.cpu().numpy(),
                            k=3)  # k=3 for Kp-chess
    model.fit(X_l, y_l, shuffle=True,)
    return model


# ---------- helper: get rule-/MF-views ---------------------------------
def _extract_views(model, X, device):
    """
    Return (rule_view, mf_view) for arbitrary extractor.
    For ANFIS/POPFNN we use rule activations + MF grades.
    For FMNC we use the membership grades of every hyper‑box
    as a surrogate ‘rule’ view; MF view is duplicated.
    """
    if isinstance(model, (NoHybridANFIS, HybridANFIS, POPFNN)):
        model.eval()
        with torch.no_grad():
            _, rule, _ = model(X.to(device))
            mf = model._fuzzify(X.to(device))          # [N,d,M]
        rule = F.normalize(rule, p=2, dim=1).cpu()
        mf   = F.normalize(mf.reshape(len(mf), -1), p=2, dim=1).cpu()
        return rule, mf
    elif isinstance(model, FMNC):
        # FMNC → membership degree matrix [N, num_boxes]
        mem = torch.tensor(model.membership(X.cpu().numpy()),
                           dtype=torch.float32)
        mem = F.normalize(mem, p=2, dim=1)
        return mem, mem          # no separate MF view
    else:
        raise TypeError(f"Unsupported extractor type: {type(model)}")


# ---------- public factory --------------------------------------------
MODEL_REGISTRY = {
    "NoHybridANFIS": _train_nohyb,
    "HybridANFIS"  : _train_hybrid,
    "POPFNN"       : _train_popfnn,
    "FMNC"         : _train_fmmc,
}


def prepare_feature_extractor(
    model_key: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test:  torch.Tensor,
    *,
    frac_labeled: float = 0.1,
    device: str = "cpu",
    model_kwargs: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Initialise + train one extractor and return all feature views."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_key {model_key!r}")

    model_fn = MODEL_REGISTRY[model_key]
    model_kwargs = model_kwargs or {}

    # pick labelled subset
    n_lab = int(frac_labeled * len(y_train))
    rng   = np.random.default_rng(42)
    idx_l = rng.choice(len(y_train), size=n_lab, replace=False)
    X_l, y_l = X_train[idx_l], y_train[idx_l]

    model = model_fn(
        X_l, y_l,
        X_train_full=X_train,
        input_dim=X_train.shape[1],
        num_classes=len(torch.unique(y_train)),
        device=device,
        **model_kwargs
    )

    rule_tr, mf_tr = _extract_views(model, X_train, device)
    rule_te, mf_te = _extract_views(model, X_test,  device)

    return {
        "model"          : model,
        "rule_train"     : rule_tr,
        "rule_test"      : rule_te,
        "mf_train"       : mf_tr,
        "mf_test"        : mf_te,
        "labeled_indices": idx_l,
    }