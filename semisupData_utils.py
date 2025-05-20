from data_utils import *
from sklearn.model_selection import train_test_split

def make_semi_split(X_tr, y_tr, frac, seed=42):
    """
    Zerlegt das Trainings-Set in:
      • labeled   – Anteil `frac`
      • unlabeled – Rest  (y = -1 für scikit-learn)
    """
    
    idx_lab, idx_unlab = train_test_split(
        torch.arange(len(X_tr)),
        train_size=frac, stratify=y_tr, random_state=seed)

    y_part = torch.full_like(y_tr, -1)
    y_part[idx_lab] = y_tr[idx_lab]

    return idx_lab, idx_unlab, y_part        # Indizes + y mit -1

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import accuracy_score
import numpy as np, torch

def rf_with_pseudo(X_tr, y_tr, X_te, y_te, frac=0.1, thr=0.0, seed=42):
    idx_lab, idx_unlab, y_part = make_semi_split(X_tr, y_tr, frac, seed)

    # ---------------- Lehrer ----------------
    rf_teacher = RandomForestClassifier(n_estimators=300, random_state=seed)
    rf_teacher.fit(X_tr[idx_lab], y_tr[idx_lab])

    proba   = rf_teacher.predict_proba(X_tr[idx_unlab])
    conf    = proba.max(1)
    mask    = conf >= thr                       # nur sichere Pseudolabels
    pseudoX = X_tr[idx_unlab][mask]
    pseudoy = proba.argmax(1)[mask]

    # combined training set
    X_comb  = torch.vstack([X_tr[idx_lab], pseudoX])
    y_comb  = torch.hstack([y_tr[idx_lab], torch.tensor(pseudoy)])

    # ---------------- Schüler ----------------
    rf_student = RandomForestClassifier(n_estimators=300, random_state=seed)
    rf_student.fit(X_comb, y_comb)

    acc_lab  = accuracy_score(y_te, rf_teacher.predict(X_te))
    acc_pl   = accuracy_score(y_te, rf_student.predict(X_te))
    return acc_lab, acc_pl

if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = load_K_chess_data_splitted()
    for p in [0.1, 0.2, 0.3]:
        acc_rf, acc_pl = rf_with_pseudo(X_tr, y_tr, X_te, y_te, frac=p)
        print(f"{int(p*100):>2}% labels | RF: {acc_rf*100:4.1f}% | RF+PL: {acc_pl*100:4.1f}%")
