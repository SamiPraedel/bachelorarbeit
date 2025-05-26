# unified_ssl_pipeline.py
# ------------------------------------------------------------
import numpy as np
import torch
import torch.nn.functional as F
import random
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Assuming these files are in the same directory or Python path
from data_utils import load_iris_data, load_K_chess_data_splitted, load_Kp_chess_data_ord
from anfis_hybrid import HybridANFIS
from PopFnn import POPFNN
from anfis_nonHyb import NoHybridANFIS
# from pop_ssl_utils import rule_class_mapping # Optional: if you want to use this for POPFNN

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Common Helper Functions ------------------------------------------------
def semi_split(y_train_tensor: torch.Tensor, frac: float, seed: int = SEED):
    """Splits training data indices into labeled and unlabeled."""
    idx_lab, idx_unlab = train_test_split(
        np.arange(len(y_train_tensor)),
        train_size=frac,
        stratify=y_train_tensor.cpu().numpy(), # Stratify expects numpy array
        random_state=seed
    )
    return torch.tensor(idx_lab, dtype=torch.long), torch.tensor(idx_unlab, dtype=torch.long)

def rf_teacher_pseudo_labels(X_train_np: np.ndarray, y_train_np: np.ndarray,
                             idx_lab: np.ndarray, idx_unlab: np.ndarray,
                             thr: float = 0.9, seed: int = SEED):
    """Trains a RandomForest teacher and generates pseudo-labels."""
    rf = RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=seed)
    rf.fit(X_train_np[idx_lab], y_train_np[idx_lab])

    if len(idx_unlab) == 0:
        return np.array([]).reshape(0, X_train_np.shape[1]), np.array([]), np.array([]), rf, 0.0

    proba = rf.predict_proba(X_train_np[idx_unlab])
    conf = proba.max(axis=1)
    mask = conf >= thr

    X_pseudo_np = X_train_np[idx_unlab][mask]
    y_pseudo_np = proba.argmax(axis=1)[mask]
    w_pseudo_np = conf[mask]
    
    avg_confidence = conf.mean() if len(conf) > 0 else 0.0

    return X_pseudo_np, y_pseudo_np, w_pseudo_np, rf, avg_confidence

# ---------- Model Specific Training Functions ------------------------------------

def train_hybrid_anfis_ssl(X_l, y_l, X_p, y_p, w_p,
                           input_dim, num_classes,
                           num_mfs=4, max_rules=1000,
                           epochs=50, lr=5e-3, seed=SEED):
    X_all = torch.cat([X_l, X_p]).to(device)
    y_all = torch.cat([y_l, y_p]).to(device)
    w_all = torch.cat([torch.ones(len(y_l), device=device), w_p.to(device)])
    y_onehot = F.one_hot(y_all, num_classes).float()

    model = HybridANFIS(input_dim, num_classes, num_mfs, max_rules, seed=seed).to(device)
    opt = torch.optim.Adam([
        {'params': model.centers, 'lr': lr},
        {'params': model.widths, 'lr': lr},
    ])

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits, norm_fs, x_ext = model(X_all)
        loss = (w_all * F.cross_entropy(logits, y_all, reduction='none')).mean()
        loss.backward()
        opt.step()
        model.update_consequents(norm_fs.detach(), x_ext.detach(), y_onehot)
    return model

def train_popfnn_ssl(X_l, y_l, X_p, y_p, w_p,
                     input_dim, num_classes,
                     num_mfs=4, epochs=50, lr=5e-4, seed=SEED):
    X_all = torch.cat([X_l, X_p]).to(device)
    y_all = torch.cat([y_l, y_p]).to(device)
    w_all = torch.cat([torch.ones(len(y_l), device=device), w_p.to(device)])

    model = POPFNN(input_dim, num_classes, num_mfs=num_mfs).to(device)
    model.pop_init(X_l.to(device), y_l.to(device))  # POPFNN specific initialization

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_all)
        loss = (loss_fn(logits, y_all) * w_all).mean()
        loss.backward()
        opt.step()
    return model

def train_nohybrid_anfis_ssl(X_l, y_l, X_p, y_p, w_p,
                             input_dim, num_classes,
                             num_mfs=7, max_rules=2000, zeroG=False,
                             epochs=100, lr=5e-3, seed=SEED):
    X_all = torch.cat([X_l, X_p]).to(device)
    y_all = torch.cat([y_l, y_p]).to(device)
    # Original NoHybrid pipeline squared the pseudo-label weights
    w_all = torch.cat([torch.ones(len(y_l), device=device), w_p.to(device)**2])

    model = NoHybridANFIS(input_dim, num_classes, num_mfs, max_rules, seed=seed, zeroG=zeroG).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits, norm_fs, mask = model(X_all)
        loss_main = (w_all * ce_loss_fn(logits, y_all)).mean()
        loss_aux = model.load_balance_loss(norm_fs.detach(), mask) # Assumes alpha is handled in model
        loss = loss_main + loss_aux
        loss.backward()
        opt.step()
    return model

# ---------- Main Experiment Loop ------------------------------------------
def run_experiments(datasets_config, models_config, label_fractions, teacher_threshold):
    print(f"Using device: {device}")
    for dataset_name, data_loader_fn in datasets_config.items():
        print(f"\n{'='*20} DATASET: {dataset_name.upper()} {'='*20}")
        X_tr_tensor, y_tr_tensor, X_te_tensor, y_te_tensor = data_loader_fn()
        
        # Ensure data is on CPU for initial numpy conversions if needed by helpers
        X_tr_np = X_tr_tensor.cpu().numpy()
        y_tr_np = y_tr_tensor.cpu().numpy()

        input_dim = X_tr_tensor.shape[1]
        num_classes = int(y_tr_tensor.max().item() + 1)

        for model_name, model_trainer_fn, model_specific_params in models_config:
            print(f"\n  --- Model: {model_name} ---")
            for frac in label_fractions:
                print(f"\n    Label Fraction: {frac*100:.0f}%")
                idx_l, idx_u = semi_split(y_tr_tensor, frac, seed=SEED)

                X_p_np, y_p_np, w_p_np, rf_teacher, avg_conf = rf_teacher_pseudo_labels(
                    X_tr_np, y_tr_np, idx_l.cpu().numpy(), idx_u.cpu().numpy(),
                    thr=teacher_threshold, seed=SEED
                )

                # Convert to Tensors for model training
                X_l_tensor = X_tr_tensor[idx_l].to(device)
                y_l_tensor = y_tr_tensor[idx_l].to(device)
                X_p_tensor = torch.from_numpy(X_p_np).float().to(device)
                y_p_tensor = torch.from_numpy(y_p_np).long().to(device)
                w_p_tensor = torch.from_numpy(w_p_np).float().to(device)

                print(f"      Labeled: {len(X_l_tensor)}, Pseudo-labeled: {len(X_p_tensor)} (Avg. Teacher Conf: {avg_conf:.2f})")

                if len(X_l_tensor) == 0:
                    print("      Skipping training: No labeled data.")
                    continue
                if len(X_p_tensor) == 0 and model_name != "SupervisedBaseline": # Allow supervised baseline to run if no pseudo labels
                     print("      Warning: No pseudo-labels generated. SSL model might behave like supervised on labeled set.")


                # Train the SSL model
                trained_model = model_trainer_fn(
                    X_l_tensor, y_l_tensor, X_p_tensor, y_p_tensor, w_p_tensor,
                    input_dim, num_classes, seed=SEED, **model_specific_params
                )

                # Evaluation
                rf_acc = accuracy_score(y_te_tensor.cpu().numpy(), rf_teacher.predict(X_te_tensor.cpu().numpy()))
                
                trained_model.eval()
                with torch.no_grad():
                    # Ensure X_te_tensor is on the correct device for the model
                    predictions = trained_model(X_te_tensor.to(device))
                    # Handle different output formats (logits vs. direct class predictions)
                    if isinstance(predictions, tuple): # ANFIS models return (logits, norm_fs, ...)
                        y_pred_ssl = predictions[0].argmax(dim=1).cpu()
                    else: # POPFNN returns logits
                        y_pred_ssl = predictions.argmax(dim=1).cpu()
                
                ssl_model_acc = accuracy_score(y_te_tensor.cpu().numpy(), y_pred_ssl.numpy())

                print(f"      RF Teacher Acc: {rf_acc*100:5.2f}% | "
                      f"{model_name} SSL Acc: {ssl_model_acc*100:5.2f}%")
                
                # Optional: POPFNN specific output
                # if model_name == "POPFNN" and hasattr(trained_model, 'get_rules_with_classes'):
                #     print("      POPFNN Rule Class Mapping (Top 3):")
                #     rule_class_mapping(trained_model, top_k=3) # Requires pop_ssl_utils

if __name__ == "__main__":
    # --- Configuration ---
    DATASETS = {
        "Iris": load_iris_data,
        "K_Chess": load_K_chess_data_splitted,
        "Kp_Chess": load_Kp_chess_data_ord
    }

    # Define model trainers and their specific parameters
    # Common params like epochs, lr can be set here or passed if they vary more.
    MODELS = [
        ("HybridANFIS", train_hybrid_anfis_ssl, {
            "num_mfs": 4, "max_rules": 1000, "epochs": 50, "lr": 5e-3
        }),
        ("POPFNN", train_popfnn_ssl, {
            "num_mfs": 4, "epochs": 50, "lr": 5e-4
        }),
        ("NoHybridANFIS", train_nohybrid_anfis_ssl, {
            "num_mfs": 7, "max_rules": 2000, "zeroG": False, "epochs": 100, "lr": 5e-3
        }),
        # Example: Add a supervised baseline using only labeled data for one of the models
        # ("HybridANFIS_Supervised", train_hybrid_anfis_ssl, { # Use same trainer, but X_p, y_p, w_p will be empty
        #     "num_mfs": 4, "max_rules": 1000, "epochs": 50, "lr": 5e-3
        # }),
    ]

    LABEL_FRACTIONS = [0.1, 0.2, 0.3, 0.5] # Example fractions
    TEACHER_CONF_THRESHOLD = 0.90

    # --- Run All Experiments ---
    # To run for a specific dataset or model, filter DATASETS or MODELS dict/list before passing
    # e.g., run_experiments({"Iris": load_iris_data}, [MODELS[0]], LABEL_FRACTIONS, TEACHER_CONF_THRESHOLD)
    
    run_experiments(DATASETS, MODELS, LABEL_FRACTIONS, TEACHER_CONF_THRESHOLD)

    # Example of running a specific configuration:
    # print("\n\nRunning a specific configuration for Iris and HybridANFIS:")
    # specific_datasets = {"Iris": DATASETS["Iris"]}
    # specific_models = [model_config for model_config in MODELS if model_config[0] == "HybridANFIS"]
    # run_experiments(specific_datasets, specific_models, [0.1], 0.95)

