# ssl_nohybrid_anfis_pipeline.py
# ------------------------------------------------------------
import numpy as np, torch, random, warnings, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import accuracy_score
from sklearn.model_selection import train_test_split
from data_utils  import load_K_chess_data_splitted, load_iris_data, load_Kp_chess_data_ord
from anfis_nonHyb   import NoHybridANFIS      
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED);  random.seed(SEED);  torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Helper ---------------------------------------------------------
def semi_split(y, frac, seed=0):
    idx_L, idx_U = train_test_split(
        torch.arange(len(y)), train_size=frac,
        stratify=y, random_state=seed)
    return idx_L, idx_U


def rf_teacher(X, y, idx_L, idx_U, thr=0.9, seed=0):
    rf = RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                random_state=seed)
    rf.fit(X[idx_L], y[idx_L])
    proba = rf.predict_proba(X[idx_U])
    conf  = proba.max(1)
    mask  = conf >= thr
    X_p   = X[idx_U][mask]
    y_p   = proba.argmax(1)[mask]
    w_p   = conf[mask]          # Gewichte = Vertrauen
    return X_p, y_p, w_p, rf, conf.mean()


# ---------- Training -------------------------------------------------------
def train_nohybrid_ssl(X_L, y_L, X_P, y_P, w_P,
                       input_dim, num_classes,
                       num_mfs=7, max_rules=2000, zeroG=False,
                       epochs=100, lr=5e-3):

    # --- Daten   -----------------------------------------------------------
    X_all = torch.cat([X_L, X_P]).to(device)
    y_all = torch.cat([y_L, y_P]).to(device)
    w_all = torch.cat([torch.ones(len(y_L)), w_P]).to(device) ** 2  # dämpfen

    model = NoHybridANFIS(input_dim, num_classes,
                          num_mfs, max_rules,
                          seed=SEED, zeroG=zeroG).to(device)

    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    ce    = torch.nn.CrossEntropyLoss(reduction='none')

    for _ in range(epochs):
        model.train();  opt.zero_grad()
        logits, norm_fs, mask = model(X_all)

        loss_main = (w_all * ce(logits, y_all)).mean()
        loss_aux  = model.load_balance_loss(norm_fs.detach(), mask)  # alpha=0.01
        loss = loss_main + loss_aux

        loss.backward();  opt.step()

    return model


# ---------- Full experiment loop ------------------------------------------
def run(label_fracs=(0.1, 0.2, 0.3), thr=0.9,
        num_mfs=4, max_rules=1000, zeroG=False):

    X_tr, y_tr, X_te, y_te = load_K_chess_data_splitted()
    X_tr, y_tr, X_te, y_te = load_iris_data()
    d, C = X_tr.shape[1], int(y_tr.max() + 1)

    for frac in label_fracs:
        idx_L, idx_U = semi_split(y_tr, frac, SEED)

        X_p, y_p, w_p, rf, cbar = rf_teacher(
            X_tr.numpy(), y_tr.numpy(),
            idx_L.numpy(), idx_U.numpy(),
            thr=thr, seed=SEED)

        # torch-Tensors -----------------------------------------------------
        X_L, y_L = X_tr[idx_L], y_tr[idx_L]
        X_P = torch.from_numpy(X_p).float()
        y_P = torch.from_numpy(y_p).long()
        w_P = torch.from_numpy(w_p).float()

        model = train_nohybrid_ssl(X_L, y_L, X_P, y_P, w_P,
                                   d, C, num_mfs, max_rules, zeroG)

        # ---------- Evaluation -------------------------------------------
        rf_acc = rf.score(X_te.numpy(), y_te.numpy())

        model.eval()
        with torch.no_grad():
            pred = model(X_te.to(device))[0].argmax(1).cpu()
        nh_acc = accuracy_score(y_te, pred)

        print(f"{int(frac*100):>2}% labels | RF: {rf_acc*100:5.2f}% "
              f"| NoHybrid SSL: {nh_acc*100:5.2f}%  "
              f"(avg conf {cbar:.2f})")


if __name__ == "__main__":
    run()
