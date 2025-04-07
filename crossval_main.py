# crossval_main.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torchmetrics.functional import matthews_corrcoef

# Import your fuzzy models & training code
from anfis_nonHyb import NoHybridANFIS, train_anfis
from anfis_hybrid import HybridANFIS, train_hybrid_anfis
# from mlp_iris import FullyConnected, fit_mlp

# Import your "full dataset" loaders
from data_utils import (
    get_chessK_full,
    get_chessKp_full,
    get_iris_full
    # etc. 
)

def get_full_dataset(dataset_name):
    """
    Return (X_np, y_np) *without splitting*.
    """
    if dataset_name == "ChessK":
        return get_chessK_full()  
    elif dataset_name == "ChessKp":
        return get_chessKp_full()
    elif dataset_name == "iris":
        return get_iris_full()
    elif dataset_name == "abalone":
        return 
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# -------------------------------------------------------------
#  Helper: train_no_hybrid_anfis
# -------------------------------------------------------------
def train_no_hybrid_anfis(X_train, y_train, X_val, y_val,
                          input_dim, num_classes, num_mfs, max_rules,
                          seed=42, lr=1e-3, num_epochs=50):
    # Convert to Torch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)

    # WeightedRandomSampler
    classes = np.unique(y_train)
    class_counts = [np.sum(y_train == c) for c in classes]
    weights_per_class = 1.0 / np.array(class_counts, dtype=float)
    sample_weights    = [weights_per_class[label] for label in y_train]
    sampler           = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_ds          = TensorDataset(X_train_t, y_train_t)
    train_loader      = DataLoader(train_ds, batch_size=64, sampler=sampler)

    # Initialize model
    model = NoHybridANFIS(input_dim=input_dim,
                          num_classes=num_classes,
                          num_mfs=num_mfs,
                          max_rules=max_rules,
                          seed=seed, zeroG=False)

    # K-Means init
    centers_k, widths_k = model.initialize_mfs_with_kmeans(X_train)  # X_train as np array
    with torch.no_grad():
        model.centers[:] = torch.tensor(centers_k, dtype=torch.float32)
        model.widths[:]  = torch.tensor(widths_k,  dtype=torch.float32)

    # Train
    train_anfis(model, X_train_t, y_train_t, num_epochs=num_epochs, lr=lr, dataloader=train_loader)

    # Predict
    with torch.no_grad():
        model.eval()
        outputs, _, _ = model(X_val_t)
        preds_val = torch.argmax(outputs, dim=1).cpu().numpy()
    return preds_val

# -------------------------------------------------------------
#  RandomForest or other models
# -------------------------------------------------------------
def train_random_forest(X_train, y_train, X_val, y_val):
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    preds_val = rf.predict(X_val)
    return preds_val


#--------------------
# Hybrid anfis
#--------------------
def train_hybrid_anfis(X_train, y_train, X_val, y_val,
                          input_dim, num_classes, num_mfs, max_rules,
                          seed=42, lr=1e-3, num_epochs=100):
    # Convert to Torch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)

    # WeightedRandomSampler
    classes = np.unique(y_train)
    class_counts = [np.sum(y_train == c) for c in classes]
    weights_per_class = 1.0 / np.array(class_counts, dtype=float)
    sample_weights    = [weights_per_class[label] for label in y_train]
    sampler           = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_ds          = TensorDataset(X_train_t, y_train_t)
    train_loader      = DataLoader(train_ds, batch_size=64, sampler=sampler)

    # Initialize model
    model = HybridANFIS(input_dim=input_dim,
                          num_classes=num_classes,
                          num_mfs=num_mfs,
                          max_rules=max_rules,
                          seed=seed)

    # K-Means init
    centers_k, widths_k = model.initialize_mfs_with_kmeans(X_train)  # X_train as np array
    with torch.no_grad():
        model.centers[:] = torch.tensor(centers_k, dtype=torch.float32)
        model.widths[:]  = torch.tensor(widths_k,  dtype=torch.float32)

    # Train
    train_anfis(model, X_train_t, y_train_t, num_epochs=num_epochs, lr=lr, dataloader=train_loader)

    # Predict
    with torch.no_grad():
        model.eval()
        outputs, _, _ = model(X_val_t)
        preds_val = torch.argmax(outputs, dim=1).cpu().numpy()
    return preds_val






# -------------------------------------------------------------
#  CROSS-VALIDATION
# -------------------------------------------------------------
def cross_val_experiment(dataset_name, model_type,
                         num_mfs=2, max_rules=50, seed=42,
                         lr=1e-3, num_epochs=50, n_splits=1):
    from sklearn.model_selection import StratifiedKFold

    # 1) Get the full dataset
    X_np, y_np = get_full_dataset(dataset_name)
    n_classes = len(np.unique(y_np))
    input_dim = X_np.shape[1]

    # 2) StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    mccs = []

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(X_np, y_np)):
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]

        if model_type == "noHyb":
            preds_val = train_no_hybrid_anfis(
                X_train, y_train, X_val, y_val,
                input_dim, n_classes,
                num_mfs, max_rules,
                seed, lr, num_epochs
            )
        elif model_type == "rf":
            preds_val = train_random_forest(X_train, y_train, X_val, y_val)
        elif model_type =="anfis":
                preds_val = train_hybrid_anfis(
                X_train, y_train, X_val, y_val,
                input_dim, n_classes,
                num_mfs, max_rules,
                seed, lr, num_epochs
            )
        else:
            raise ValueError("Unknown model type: " + model_type)

        # Evaluate
        acc = accuracy_score(y_val, preds_val)
        preds_val_t = torch.tensor(preds_val)
        y_val_t     = torch.tensor(y_val)
        mcc_val = float(matthews_corrcoef(preds_val_t, y_val_t,
                                          task="multiclass",
                                          num_classes=n_classes))
        accs.append(acc)
        mccs.append(mcc_val)

    return {
        "dataset": dataset_name,
        "model": model_type,
        "acc_mean": np.mean(accs),
        "acc_std":  np.std(accs),
        "mcc_mean": np.mean(mccs),
        "mcc_std":  np.std(mccs)
    }

# -------------------------------------------------------------
#  MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    # combos = [
    #     ("ChessK", "noHyb"),
    #     ("ChessK", "anfis"),
    #     ("ChessK", "rf")
    #     ("ChessKp", "noHyb"),
    #     ("ChessKp", "anfis"),
    #     ("ChessKp", "rf"),
    #     ("Poker", "noHyb"),
    #     ("Poker", "anfis"),
    #     ("Poker", "rf"),
    #     ("abalone", "noHyb"),
    #     ("abalone", "anfis"),
    #     ("abalone", "rf"),
    #     ("heart", "noHyb"),
    #     ("heart", "anfis"),
    #     ("heart", "rf"),
    #     ("iris",   "noHyb"),
    #     ("iris",   "anfis"),
    #     ("iris", "rf"),
    # ]
    
    # combos = [("ChessK", "noHyb"),
    #           ("ChessKp", "noHyb"),
    #           ("Poker", "noHyb")
    #           ("Iris", "noHyb"),
    #           ("abalone", "noHyb")
    #           ]

    results = []
    # for (ds, mt) in combos:
    #     res = cross_val_experiment(
    #         dataset_name=ds,
    #         model_type=mt,
    #         num_mfs=2,
    #         max_rules=100,
    #         seed=42,
    #         lr=1e-3,
    #         num_epochs=30,
    #         n_splits=5
    #     )
    res = cross_val_experiment(
        dataset_name="ChessKp",
        model_type="rf",
        num_mfs=2,
        max_rules=500,
        seed=45,
        lr=0.001,
        num_epochs=200,
        n_splits=5)
    results.append(res)

    df_results = pd.DataFrame(results)
    print(df_results)
