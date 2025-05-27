# unified_ssl_pipeline.py
# ------------------------------------------------------------
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import warnings
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from anfis_hybrid import HybridANFIS
from anfis_nonHyb import NoHybridANFIS
from PopFnn import POPFNN

from data_utils import load_iris_data, load_K_chess_data_splitted, load_heart_data, load_wine_data, load_abalon_data

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
        stratify=y_train_tensor.cpu().numpy(),
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

# ---------- Teacher-Student SSL Training Functions ----------------------------------
def train_hybrid_anfis_ssl(X_l, y_l, X_p, y_p, w_p,
                           input_dim, num_classes,
                           num_mfs=4, max_rules=1000,
                           epochs=50, lr=5e-3, seed=42, **kwargs):
    X_all = torch.cat([X_l, X_p])
    y_all = torch.cat([y_l, y_p])
    w_all = torch.cat([torch.ones(len(y_l), device=device), w_p])
    y_onehot = F.one_hot(y_all, num_classes).float()

    model = HybridANFIS(input_dim, num_classes, num_mfs, max_rules, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': model.centers, 'lr': lr}, {'params': model.widths, 'lr': lr}])

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits, norm_fs, x_ext = model(X_all)
        loss = (w_all * F.cross_entropy(logits, y_all, reduction='none')).mean()
        loss.backward()
        opt.step()
        model.update_consequents(norm_fs.detach(), x_ext.detach(), y_onehot)
    return model

def train_nohybrid_anfis_ssl(X_l, y_l, X_p, y_p, w_p,
                             input_dim, num_classes,
                             num_mfs=7, max_rules=2000, zeroG=False,
                             epochs=100, lr=5e-3, seed=42, **kwargs):
    X_all = torch.cat([X_l, X_p])
    y_all = torch.cat([y_l, y_p])
    w_all = torch.cat([torch.ones(len(y_l), device=device), w_p**2])

    model = NoHybridANFIS(input_dim, num_classes, num_mfs, max_rules, seed=seed, zeroG=zeroG).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits, norm_fs, mask = model(X_all)
        loss_main = (w_all * ce_loss_fn(logits, y_all)).mean()
        loss_aux = model.load_balance_loss(norm_fs.detach(), mask)
        loss = loss_main + loss_aux
        loss.backward()
        opt.step()
    return model

def train_popfnn_ssl(X_l, y_l, X_p, y_p, w_p,
                     input_dim, num_classes,
                     num_mfs=4, epochs=50, lr=5e-4, seed=42, **kwargs):
    X_all = torch.cat([X_l, X_p])
    y_all = torch.cat([y_l, y_p])
    w_all = torch.cat([torch.ones(len(y_l), device=device), w_p])

    model = POPFNN(input_dim, num_classes, num_mfs=num_mfs).to(device)
    model.pop_init(X_l, y_l)

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

# =========== NEW: Rule-Based Self-Training SSL Functions ==========================
def train_hybrid_anfis_rule_ssl(X_l, y_l, X_u, input_dim, num_classes,
                                num_mfs=4, max_rules=1000, epochs=200, lr=5e-3, seed=42,
                                initial_train_ratio=0.2, rule_conf_thresh=0.9, firing_thresh=0.5, **kwargs):
    """Trains HybridANFIS using rule-based self-training."""
    model = HybridANFIS(input_dim, num_classes, num_mfs, max_rules, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': model.centers, 'lr': lr}, {'params': model.widths, 'lr': lr}])
    
    initial_epochs = int(epochs * initial_train_ratio)
    y_l_onehot = F.one_hot(y_l, num_classes).float()
    
    for _ in range(initial_epochs):
        model.train()
        opt.zero_grad()
        logits, norm_fs, x_ext = model(X_l)
        loss = F.cross_entropy(logits, y_l)
        loss.backward()
        opt.step()
        model.update_consequents(norm_fs.detach(), x_ext.detach(), y_l_onehot)

    X_p, y_p, w_p = torch.empty(0, X_l.shape[1], device=device), torch.empty(0, device=device, dtype=torch.long), torch.empty(0, device=device)
    
    with torch.no_grad():
        model.eval()
        _, norm_fs_l, _ = model(X_l)
        rule_class_weights = norm_fs_l.t() @ y_l_onehot
        rule_class_probs = F.normalize(rule_class_weights, p=1, dim=1)
        rule_confidence, confident_class = torch.max(rule_class_probs, dim=1)
        confident_rules_mask = rule_confidence > rule_conf_thresh
        
        if confident_rules_mask.sum().item() > 0 and len(X_u) > 0:
            _, norm_fs_u, _ = model(X_u)
            sample_max_firing, sample_best_rule_idx = torch.max(norm_fs_u, dim=1)
            best_rule_is_confident = confident_rules_mask[sample_best_rule_idx]
            firing_is_strong = sample_max_firing > firing_thresh
            pseudo_label_mask = best_rule_is_confident & firing_is_strong
            idx_p = torch.where(pseudo_label_mask)[0]
            
            if len(idx_p) > 0:
                X_p = X_u[idx_p]
                best_rules_for_pseudo_samples = sample_best_rule_idx[idx_p]
                y_p = confident_class[best_rules_for_pseudo_samples]
                w_p = rule_confidence[best_rules_for_pseudo_samples]
    
    X_all = torch.cat([X_l, X_p])
    y_all = torch.cat([y_l, y_p])
    w_all = torch.cat([torch.ones(len(y_l), device=device), w_p])
    y_all_onehot = F.one_hot(y_all, num_classes).float()

    for _ in range(epochs - initial_epochs):
        model.train()
        opt.zero_grad()
        logits, norm_fs, x_ext = model(X_all)
        loss = (w_all * F.cross_entropy(logits, y_all, reduction='none')).mean()
        loss.backward()
        opt.step()
        model.update_consequents(norm_fs.detach(), x_ext.detach(), y_all_onehot)

    return model, len(X_p)


def train_popfnn_rule_ssl(X_l, y_l, X_u, input_dim, num_classes,
                          num_mfs=4, epochs=50, lr=5e-4, seed=42, 
                          rule_conf_thresh=0.9, **kwargs):
    """Trains POPFNN using its inherent rule structure for SSL."""
    torch.manual_seed(seed)
    model = POPFNN(input_dim, num_classes, num_mfs=num_mfs).to(device)
    model.pop_init(X_l, y_l)
    
    if model.R == 0: # No rules were generated, train supervised only
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.cross_entropy(model(X_l), y_l)
            loss.backward(); opt.step()
        return model, 0

    X_p, y_p, w_p = torch.empty(0, X_l.shape[1], device=device), torch.empty(0, device=device, dtype=torch.long), torch.empty(0, device=device)
    
    with torch.no_grad():
        rule_class_weights = model.W.view(model.R, model.C, model.M).sum(dim=2)
        rule_confidence, confident_class = torch.max(rule_class_weights, dim=1)
        confident_rules_mask = rule_confidence > rule_conf_thresh

        if confident_rules_mask.sum().item() > 0 and len(X_u) > 0:
            fire_u = model._fire(X_u)
            sample_max_firing, sample_best_rule_idx = torch.max(fire_u, dim=1)
            best_rule_is_confident = confident_rules_mask[sample_best_rule_idx]
            idx_p = torch.where(best_rule_is_confident)[0]

            if len(idx_p) > 0:
                X_p = X_u[idx_p]
                best_rules_for_pseudo_samples = sample_best_rule_idx[idx_p]
                y_p = confident_class[best_rules_for_pseudo_samples]
                w_p = rule_confidence[best_rules_for_pseudo_samples]

    X_all = torch.cat([X_l, X_p])
    y_all = torch.cat([y_l, y_p])
    w_all = torch.cat([torch.ones(len(y_l), device=device), w_p])
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_all)
        loss = (loss_fn(logits, y_all) * w_all).mean()
        loss.backward()
        opt.step()
        
    return model, len(X_p)

# ---------- CSV Logging Helper ---------------------------------------------
CSV_FILE_PATH = "ssl_experiment_results_v2.csv"
CSV_HEADERS = [
    "Dataset", "Model", "SSL_Method", "Label_Fraction_Percent", "SSL_Threshold",
    "Seed", "Epochs", "Learning_Rate", "Num_MFs", "Max_Rules", "ZeroG",
    "Num_Labeled", "Num_Pseudo_Labeled", "Avg_Pseudo_Confidence",
    "Teacher_Accuracy_Percent", "Student_Accuracy_Percent"
]
COLUMN_WIDTHS = [15, 15, 12, 24, 15, 6, 8, 15, 10, 10, 7, 14, 20, 24, 28, 28]

def format_for_csv_row(data_list, widths):
    return [str(item).ljust(widths[i]) for i, item in enumerate(data_list)]

def initialize_csv():
    with open(CSV_FILE_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(format_for_csv_row(CSV_HEADERS, COLUMN_WIDTHS))

def append_to_csv(data_row):
    with open(CSV_FILE_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(format_for_csv_row(data_row, COLUMN_WIDTHS))

# ---------- Main Experiment Loop ------------------------------------------
def run_experiments(datasets_config, models_config, label_fractions, teacher_conf_threshold):
    print(f"Using device: {device}")
    initialize_csv()

    for dataset_name, data_loader_fn in datasets_config.items():
        print(f"\n{'='*20} DATASET: {dataset_name.upper()} {'='*20}")
        X_tr_tensor, y_tr_tensor, X_te_tensor, y_te_tensor = data_loader_fn()
        
        X_tr_np, y_tr_np = X_tr_tensor.cpu().numpy(), y_tr_tensor.cpu().numpy()
        input_dim = X_tr_tensor.shape[1]
        num_classes = int(y_tr_tensor.max().item() + 1)

        for model_name, model_trainer_fn, model_params in models_config:
            ssl_method = model_params.get("ssl_method", "teacher")
            model_display_name = f"{model_name}_{ssl_method.upper()}"
            
            for frac in label_fractions:
                print(f"\n  --- Model: {model_display_name}, Label Fraction: {frac*100:.0f}% ---")
                idx_l, idx_u = semi_split(y_tr_tensor, frac, seed=SEED)
                X_l, y_l = X_tr_tensor[idx_l].to(device), y_tr_tensor[idx_l].to(device)

                num_pseudo_labels, avg_pseudo_conf, teacher_acc, ssl_threshold_log = 0, "N/A", "N/A", "N/A"
                
                # Create a mutable copy of model_params to pass to trainers
                current_params = model_params.copy()

                if ssl_method == "teacher":
                    ssl_threshold_log = f"{teacher_conf_threshold:.2f}"
                    X_p_np, y_p_np, w_p_np, rf, avg_conf = rf_teacher_pseudo_labels(
                        X_tr_np, y_tr_np, idx_l.cpu().numpy(), idx_u.cpu().numpy(),
                        thr=teacher_conf_threshold, seed=SEED)
                    
                    X_p = torch.from_numpy(X_p_np).float().to(device)
                    y_p = torch.from_numpy(y_p_np).long().to(device)
                    w_p = torch.from_numpy(w_p_np).float().to(device)
                    
                    num_pseudo_labels = len(X_p)
                    avg_pseudo_conf = f"{avg_conf:.4f}"
                    teacher_acc = f"{accuracy_score(y_te_tensor.cpu().numpy(), rf.predict(X_te_tensor.cpu().numpy()))*100:.2f}"
                    
                    trained_model = model_trainer_fn(
                        X_l, y_l, X_p, y_p, w_p,
                        input_dim, num_classes, seed=SEED, **current_params)

                elif ssl_method == "rule_based":
                    X_u = X_tr_tensor[idx_u].to(device)
                    rule_thr = current_params.get('rule_conf_thresh', 'N/A')
                    fire_thr = current_params.get('firing_thresh', 'N/A')
                    ssl_threshold_log = f"R:{rule_thr}|F:{fire_thr}"
                    
                    trained_model, num_pseudo_labels = model_trainer_fn(
                        X_l, y_l, X_u,
                        input_dim, num_classes, seed=SEED, **current_params)
                
                print(f"      Labeled: {len(X_l)}, Pseudo-labeled: {num_pseudo_labels}")
                
                # Evaluation
                trained_model.eval()
                with torch.no_grad():
                    predictions = trained_model(X_te_tensor.to(device))
                    y_pred_ssl = predictions[0].argmax(dim=1).cpu() if isinstance(predictions, tuple) else predictions.argmax(dim=1).cpu()
                
                ssl_model_acc = accuracy_score(y_te_tensor.cpu().numpy(), y_pred_ssl.numpy())
                print(f"      Teacher Acc: {teacher_acc}% | Student Acc: {ssl_model_acc*100:5.2f}%")
                
                # Logging
                result_row = [
                    dataset_name, model_name, ssl_method, f"{frac*100:.0f}", ssl_threshold_log, SEED,
                    current_params.get("epochs", "N/A"),
                    f"{current_params.get('lr', 'N/A'):.1e}",
                    current_params.get("num_mfs", "N/A"),
                    current_params.get("max_rules", "N/A"),
                    current_params.get("zeroG", "N/A"),
                    len(X_l), num_pseudo_labels, avg_pseudo_conf, teacher_acc,
                    f"{ssl_model_acc*100:.2f}"
                ]
                append_to_csv(result_row)

if __name__ == "__main__":
    DATASETS = {
        "Iris": load_iris_data, "K_Chess": load_K_chess_data_splitted,
        "Heart": load_heart_data, "Wine": load_wine_data, "Abalon": load_abalon_data,
    }

    MODELS = [
        # --- Teacher-Student SSL Models ---
        ("HybridANFIS", train_hybrid_anfis_ssl, {
            "ssl_method": "teacher", "num_mfs": 4, "max_rules": 1000, "epochs": 200, "lr": 5e-3
        }),
        ("POPFNN", train_popfnn_ssl, {
            "ssl_method": "teacher", "num_mfs": 4, "epochs": 300, "lr": 5e-4
        }),
        # --- Rule-Based Self-Training SSL Models ---
        ("HybridANFIS", train_hybrid_anfis_rule_ssl, {
            "ssl_method": "rule_based", "num_mfs": 4, "max_rules": 1000, "epochs": 200, "lr": 5e-3,
            "initial_train_ratio": 0.2, "rule_conf_thresh": 0.9, "firing_thresh": 0.5
        }),
        ("POPFNN", train_popfnn_rule_ssl, {
            "ssl_method": "rule_based", "num_mfs": 4, "epochs": 300, "lr": 5e-4,
            "rule_conf_thresh": 0.9
        }),
    ]

    LABEL_FRACTIONS = [0.1, 0.2, 0.5]
    TEACHER_CONF_THRESHOLD = 0.90

    run_experiments(DATASETS, MODELS, LABEL_FRACTIONS, TEACHER_CONF_THRESHOLD)