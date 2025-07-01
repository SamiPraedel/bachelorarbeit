# semi_sup_rule_lp.py
# ------------------------------------------------------------
import math, time, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset 
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from anfis_nonHyb  import NoHybridANFIS           # dein Modell
from anfisHelper   import initialize_mfs_with_kmeans
from data_utils    import load_K_chess_data_splitted   # Beispiel-Datensatz
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics


# ------------------------------------------------------------
def _val_split(X, y, frac=.1, seed=42):
    """kleiner Valid-Split auf GPU‐Tensors"""
    torch.manual_seed(seed)
    n_val = max(1, int(len(X)*frac))
    idx   = torch.randperm(len(X), device=X.device)
    return (X[idx[n_val:]], y[idx[n_val:]]), (X[idx[:n_val]], y[idx[:n_val]])
# 
#  RULE-SPACE  SSL  mit  iterativer  Label-Propagation
# 
def _rule_embeddings(model, X):
    """liefert Firing-Stärken (nach Normalisierung) als numpy  [N, R]"""
    with torch.no_grad():
        _, fs, _ = model(X.to(next(model.parameters()).device))
    return fs.cpu().numpy()


def ssl_inputLP_nohyb(model, X, y, labeled_amount, num_firstEpochs):
    trueLabels = y

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    label_prop_model = LabelPropagation()
    
    rng = np.random.RandomState(42)
    random_unlabeled_points = rng.rand(len(y)) < (1-labeled_amount)
    
    labels = np.copy(y)
    labels[random_unlabeled_points] = -1
    
    
    
    
    
    X = X.cpu().numpy()

    label_prop_model.fit(X, labels)

    
    y = label_prop_model.transduction_
    
    true_np = np.asarray(trueLabels)
    pred_np = np.asarray(y)

    A = (true_np == pred_np)  
    
    accuracy = A.mean() 
    print(f"Pseudo-Label Accuracy: {accuracy*100:.2f}%")
    

    
    X = torch.tensor(X).clone().detach()
    y = torch.tensor(y).clone().detach()
    
    X = X.to(device)
    y = y.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # warmstart
    for _ in range(num_firstEpochs):
        model.train()
        for xb, yb in DataLoader(TensorDataset(X_l, y_l), batch_size=256, shuffle=True):
            xb, yb = xb.to(device), yb.to(device)
            out, fs, _ = model(xb)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            
    

    """for _ in range(num_firstEpochs):
        model.train()
        for xb, yb in DataLoader(TensorDataset(X, y), batch_size=256, shuffle=True):
            xb, yb = xb.to(device), yb.to(device)
            out, fs, _ = model(xb)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()"""
            
    
   
    return model



def ssl_rule_labelprop_nohyb(
        model,                     # NoHybridANFIS-Instanz
        X_l,  y_l,                 # labeled data  (Tensor)
        X_u,                       # unlabeled pool
        *,
        warm_epochs          = 200, # supervised warm-up
        rounds               = 40,  # max LP-Iterationen
        epochs_per_round     = 20,
        per_round_budget     = 0.60,  # höchstens x-Prozent des Pools pro Runde
        conf_thr             = 0.70,  # Mindest-Konfidenz
        lp_gamma             = 20,    # RBF-gamma   (oder kernel='knn')
        batch_train          = 512,
        lr_all               = 5e-3,
        device               = "cuda",
        kernel              = "knn"):

    C     = model.num_classes
    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr_all)

    # ------------- Warm-up 
    (X_tr, y_tr), _ = _val_split(X_l, y_l)
    initialize_mfs_with_kmeans(model, X_u)
    for _ in range(warm_epochs):
        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_train,
                            shuffle=True, drop_last=False)
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            model.train(); opt.zero_grad()
            out, fs, _ = model(xb)
            F.cross_entropy(out, yb).backward(); opt.step()
            
    print(f"Warm-up done: {warm_epochs} epochs")
    # ------------- Iterative LP-Loop 
    pool = X_u.clone()
    for r in range(rounds):
        if len(pool) == 0: break

        # --- 1) k-NN / RBF Label-Propagation im RULE-SPACE ----
        Z = _rule_embeddings(model, torch.cat([X_l, pool]))  # [N_all, R]
        y_lp_init = np.full(len(Z), -1, dtype=np.int64)
        y_lp_init[:len(y_l)] = y_l.cpu().numpy()

        lp = LabelPropagation(kernel=kernel, gamma=lp_gamma, max_iter=1000)
        lp.fit(Z, y_lp_init)

        prob_all = lp.label_distributions_
        conf_all = prob_all.max(1)
        pseudo_all = lp.transduction_

        conf_pool = conf_all[len(y_l):]             # nur Pool-Samples
        pseudo_pool = pseudo_all[len(y_l):]

        # --- 2) Auswahl neuer Pseudo-Labels  
        good_mask = conf_pool > conf_thr
        if not good_mask.any(): break                # nichts Neues

        # Budget: höchstens x % des Pools
        k_budget  = int(per_round_budget * len(pool))
        if k_budget > 0 and good_mask.sum() > k_budget:
            # wähle die k-besten nach Konfidenz
            topk = np.argpartition(-conf_pool, k_budget-1)[:k_budget]
            sel  = np.zeros_like(good_mask); sel[topk] = True
            good_mask = good_mask & sel

        idx_new = torch.where(torch.from_numpy(good_mask))[0]
        if idx_new.numel() == 0: break

        X_new = pool[idx_new]
        y_new = torch.from_numpy(pseudo_pool[good_mask]).long()

        # Anhängen
        X_l = torch.cat([X_l, X_new])
        y_l = torch.cat([y_l, y_new])

        # Pool schrumpfen
        keep = torch.ones(len(pool), dtype=torch.bool)
        keep[idx_new.cpu()] = False
        pool = pool[keep]

        # --- 3) Finetuning auf erweitertem Labeled-Set --------
        for _ in range(epochs_per_round):
            loader = DataLoader(TensorDataset(X_l, y_l), batch_size=batch_train,
                                shuffle=True, drop_last=False)
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                model.train(); opt.zero_grad()
                out, fs, _ = model(xb)
                F.cross_entropy(out, yb).backward(); opt.step()

        print(f"[Round {r+1}] added {len(idx_new)} pseudo-labels | "
              f"Pool left: {len(pool)}")

    return model






def label_propa_lame(model, num_firstEpochs, X, y, labeled_amount, x_test, y_test):
    
    trueLabels = y
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    label_prop_model = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=1000)
    
    rng = np.random.RandomState(42)
    random_unlabeled_points = rng.rand(len(y)) < (1-labeled_amount)
    print(random_unlabeled_points)
    
    labels = np.copy(y)
    labels[random_unlabeled_points] = -1
    
    mask_labeled = ~random_unlabeled_points
    X_l = X[mask_labeled]
    y_l = y[mask_labeled]

    
    label_prop_model.fit(X, labels)
    y_lb = label_prop_model.transduction_
    
    
    # X = X.to(device)
    # y = y.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=0.001)
    
    
    
    # warmstart

            
    clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
    clf.fit(X_l, y_l.cpu().numpy())
    y_pred = clf.predict(x_test)
    print('Accuracy: ', metrics.accuracy_score(y_test,y_pred))
    
    clf2 = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
    clf2.fit(X, y_lb)
    y_pred = clf2.predict(x_test)
    print('Accuracy: ', metrics.accuracy_score(y_test,y_pred))
    
    #rule_net = _rule_embeddings(model,X)

    #label_prop_model.fit(rule_net, labels)
    
    #y = label_prop_model.transduction_
    
   

    


    
    X = torch.tensor(X).clone().detach()
    y = torch.tensor(y).clone().detach()
    

    true_np = np.asarray(trueLabels)            
    pseudo = y_lb                            
    mask_unlabeled = random_unlabeled_points  
    pseudo_acc = np.mean(true_np[mask_unlabeled] == pseudo[mask_unlabeled])
    print(f"Pseudo-Label Accuracy: {pseudo_acc*100:.2f}%")
    

    
    
    return model


if __name__ == "__main__":
    # 1. Load a real dataset
    X_tr, y_tr, X_te, y_te = load_K_chess_data_splitted()
    input_dim = X_tr.shape[1]
    num_classes = len(torch.unique(y_tr))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    

    # 2. Create a semi-supervised split
    frac_labeled = 0.1
    idx_l, idx_u = train_test_split(
        torch.arange(len(y_tr)),
        train_size=frac_labeled,
        stratify=y_tr,
        random_state=42
    )
    X_l, y_l = X_tr[idx_l], y_tr[idx_l]
    X_u, y_u = X_tr[idx_u], y_tr[idx_u]

    


    
    

    print(f"Using device: {device}")
    print(f"Dataset: K-Chess | Labeled: {len(X_l)} | Unlabeled: {len(X_u)} | Test: {len(X_te)}")

    # 3. Initialize model
    model = NoHybridANFIS(
        input_dim=input_dim,
        num_classes=num_classes,
        num_mfs=4,
        max_rules=1000,
        seed=42,
        zeroG=True
    )
    model.to(device)

    # 4. Run the training
    print("\nTraining with ssl_rule_labelprop_nohyb...")
    model = ssl_rule_labelprop_nohyb(
        model, X_l, y_l, X_u, device=device, kernel='knn',
    )
    
    #model = label_propa_lame(model, 400, X_tr, y_tr, 0.1, X_te, y_te)

    # 5. Evaluate the model
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(X_te.to(device))
        preds = logits.argmax(1).cpu()
        acc = accuracy_score(y_te, preds)
        print(f"\nFinal Test Accuracy (ssl_rule_labelprop_nohyb): {acc*100:.2f}%")

    # --- Second Experiment: ssl_inputLP_nohyb ---
    print("\n" + "="*50)

    # # 3b. Re-initialize model for the second experiment
    """model_input_lp = NoHybridANFIS(
        input_dim=input_dim,
        num_classes=num_classes,
        num_mfs=4,
        max_rules=1000,
        seed=42,
        zeroG=True
    )

    # # 4b. Run the training for ssl_inputLP_nohyb
    print("\nTraining with ssl_inputLP_nohyb (rbf kernel)...")
    model_input_lp = ssl_inputLP_nohyb(
        model_input_lp, X_tr, y_tr, labeled_amount=0.1, num_firstEpochs=100
    )

    # 5b. Evaluate the second model
    model_input_lp.eval()
    with torch.no_grad():
        logits, _, _ = model_input_lp(X_te.to(device))
        preds = logits.argmax(1).cpu()
        acc = accuracy_score(y_te, preds)
        print(f"\nFinal Test Accuracy (ssl_inputLP_nohyb): {acc*100:.2f}%")"""