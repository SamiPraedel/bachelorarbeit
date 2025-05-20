# pop_ssl_utils.py
# ------------------------------------------------------------
import torch, torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def semi_split(y_tr, frac, seed=0):
    """gibt Indizes für Labeled ℒ und Unlabeled 𝕌 zurück"""
    idx_L, idx_U = train_test_split(
        torch.arange(len(y_tr)), train_size=frac,
        stratify=y_tr, random_state=seed)
    return idx_L, idx_U


@torch.no_grad()
def rf_teacher_pseudo(X, y, idx_L, idx_U, thr=0.9, seed=0):
    """RF trainieren → Pseudo-Label für 𝕌 mit confidence ≥ thr"""
    rf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
    rf.fit(X[idx_L], y[idx_L])
    proba = rf.predict_proba(X[idx_U])
    conf  = proba.max(1)
    mask  = conf >= thr
    X_pl  = X[idx_U][mask]
    y_pl  = proba.argmax(1)[mask]
    w_pl  = conf[mask]                         # Gewichte = Vertrauen
    return X_pl, y_pl, w_pl, rf



def rule_class_mapping(pop, top_k=5):
    """
    Ermittelt pro Regel die Klasse mit größtem Gewicht.
    Gibt optional die `top_k` wichtigsten Regeln pro Klasse zurück.
    """
    with torch.no_grad():
        W = (pop.W * pop.label_cent).view(pop.R, pop.C, pop.M).sum(2)  # [R,C]
        strength, cls = W.abs().max(1)                                 # [R]
        mapping = [(int(cls[i]), float(strength[i])) for i in range(pop.R)]

    if top_k > 0:
        print("\nTop-Regeln je Klasse:")
        for c in range(pop.C):
            top = [(i,s) for i,(cl,s) in enumerate(mapping) if cl==c]
            top_sorted = sorted(top, key=lambda t: -t[1])[:top_k]
            for rid, s in top_sorted:
                antecedent = " AND ".join(
                    f"x{j}∈MF{pop.rules[rid,j].item()}"
                    for j in range(pop.d))
                print(f"Cl {c}:  IF {antecedent}  THEN vote {c}  "
                      f"(strength={s:.3f})")
    return mapping
