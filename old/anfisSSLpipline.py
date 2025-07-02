# ssl_anfis_pipeline.py
# ----------------------------------------------------------
import numpy as np, torch, random, warnings
import csv
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import accuracy_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from data_utils import load_K_chess_data_splitted, load_iris_data, load_Kp_chess_data_ord
from anfis_hybrid     import HybridANFIS                 # deine Klasse

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED);  random.seed(SEED);  torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
def make_semi_split(y_tr, frac, seed=0):
    idx_lab, idx_unlab = train_test_split(
        np.arange(len(y_tr)), train_size=frac,
        stratify=y_tr, random_state=seed)
    return torch.tensor(idx_lab), torch.tensor(idx_unlab)

def teacher_pseudo_labels(X_train, y_train, idx_lab, idx_unlab,
                          thr=0.90, seed=SEED):
    rf = RandomForestClassifier(n_estimators=400, max_depth=None,
                                n_jobs=-1, random_state=seed)
    rf.fit(X_train[idx_lab], y_train[idx_lab])

    proba  = rf.predict_proba(X_train[idx_unlab])
    conf   = proba.max(1)
    mask   = conf >= thr

    X_pseudo = X_train[idx_unlab][mask]
    y_pseudo = proba.argmax(1)[mask]
    w_pseudo = conf[mask]                      # Gewichte = Vertrauen

    return X_pseudo, y_pseudo, w_pseudo, rf

# ---------------------------------------------------------------------------
def train_anfis_ssl(X_l, y_l, X_p, y_p, w_p,
                    input_dim, num_classes, global_device,
                    num_mfs=4, max_rules=1000,
                    epochs=50, lr=5e-3, seed=SEED):
    X_all = torch.cat([X_l, X_p]).to(global_device)          # [N, d]
    y_all = torch.cat([y_l, y_p]).to(global_device)          # [N]
    w_all = torch.cat([torch.ones(len(y_l), device=global_device), w_p.to(global_device)])

    y_onehot = F.one_hot(y_all, num_classes).float()

    model = HybridANFIS(input_dim, num_classes,
                        num_mfs, max_rules, seed).to(global_device)

    opt   = torch.optim.Adam([
                {'params': model.centers, 'lr': lr},
                {'params': model.widths , 'lr': lr},
            ])

    for epoch in range(epochs):
        model.train();   opt.zero_grad()
        logits, norm_fs, x_ext = model(X_all)

        # gewichtete CE-Loss
        loss = (w_all *
                F.cross_entropy(logits, y_all, reduction='none')).mean()
        loss.backward();  opt.step()

        # Consequents als closed-form LS anpassen
        model.update_consequents(norm_fs.detach(), x_ext.detach(), y_onehot)

    return model
# ---------------------------------------------------------------------------
RESULTS_FILE_PATH = "anfis_ssl_experiment_results.csv"
ERROR_LOG_PATH = "anfis_ssl_experiment_errors.log"
CSV_HEADER = ["Dataset", "LabelFraction", "RF_Teacher_Accuracy", "ANFIS_SSL_Accuracy"]

def load_completed_experiments(filepath):
    completed = set()
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        # Create file with header if it doesn't exist or is empty
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)
        except Exception as e:
            print(f"Warning: Could not create or write header to results file {filepath}: {e}")
        return completed
    try:
        with open(filepath, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            for row in reader:
                if len(row) >= 2: # Ensure at least dataset and fraction are present
                    dataset, frac_str = row[0], row[1]
                    try:
                        completed.add((dataset, float(frac_str)))
                    except ValueError:
                        print(f"Warning: Skipping malformed row in results file: {row}")
    except Exception as e:
        print(f"Warning: Could not read existing results file {filepath}: {e}")
    return completed

def append_result_to_csv(filepath, dataset_name, frac, rf_acc, ssl_acc):
    file_exists = os.path.exists(filepath)
    try:
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(filepath) == 0:
                writer.writerow(CSV_HEADER)
            writer.writerow([dataset_name, f"{frac:.2f}", f"{rf_acc:.4f}", f"{ssl_acc:.4f}"])
    except Exception as e:
        print(f"Error: Could not write to results file {filepath}: {e}")

def log_error(filepath, dataset_name, frac, error_message):
    try:
        with open(filepath, 'a') as f:
            f.write(f"Dataset: {dataset_name}, Fraction: {frac:.2f} - ERROR: {error_message}\n")
    except Exception as e:
        print(f"Critical: Could not write to error log file {filepath}: {e}")

def run_pipeline(label_fracs=(0.1, 0.2, 0.3), thr=0.90):
    print(f"Using device: {device}")
    completed_experiments = load_completed_experiments(RESULTS_FILE_PATH)

    # --- Configuration for datasets ---
    # This makes it easier to manage multiple datasets if you uncomment them later
    datasets_to_run = {
        # "K_Chess": load_K_chess_data_splitted,
        # "Iris": load_iris_data,
        "Kp_Chess": load_Kp_chess_data_ord,
    }

    for dataset_name, data_loader_fn in datasets_to_run.items():
        print(f"\n{'='*10} Processing Dataset: {dataset_name} {'='*10}")
        X_tr, y_tr, X_te, y_te = data_loader_fn()
        # Ensure tensors are on CPU initially if data_loader_fn returns them on GPU
        X_tr, y_tr, X_te, y_te = X_tr.cpu(), y_tr.cpu(), X_te.cpu(), y_te.cpu()

    print(f"\n{'='*10} Running ANFIS SSL Pipeline for Dataset: {dataset_name} {'='*10}")

    input_dim   = X_tr.shape[1]
    num_classes = int(y_tr.max().item() + 1)

    for frac in label_fracs:
        experiment_key = (dataset_name, round(frac, 2))
        if experiment_key in completed_experiments:
            print(f"Skipping: {dataset_name}, {frac*100:.0f}% labels (already completed).")
            continue

        print(f"\n--- Processing: {dataset_name}, {frac*100:.0f}% labels ---")
        try:
            idx_l, idx_u = make_semi_split(y_tr, frac, seed=SEED)
            X_p_np, y_p_np, w_p_np, rf_teacher = teacher_pseudo_labels(
                X_tr.numpy(), y_tr.numpy(), idx_l.numpy(), idx_u.numpy(),
                thr=thr)

            # Torch-Tensors
            X_l, y_l = X_tr[idx_l], y_tr[idx_l]
            X_p = torch.from_numpy(X_p_np).float()
            y_p = torch.from_numpy(y_p_np).long()
            w_p = torch.from_numpy(w_p_np).float()

            model = train_anfis_ssl(X_l, y_l, X_p, y_p, w_p,
                                    input_dim, num_classes, global_device=device,
                                    num_mfs=4, max_rules=1000)

            # Evaluation
            model.eval()
            with torch.no_grad():
                y_pred = model(X_te.to(device))[0].argmax(1).cpu()
            acc_rf  = accuracy_score(y_te.cpu().numpy(), rf_teacher.predict(X_te.cpu().numpy()))
            acc_ssl = accuracy_score(y_te.cpu().numpy(), y_pred.numpy())
            print(f"  Results: {int(frac*100):>2}% labels | RF-sup: {acc_rf*100:5.2f}%"
                  f" | ANFIS SSL: {acc_ssl*100:5.2f}%")
            append_result_to_csv(RESULTS_FILE_PATH, dataset_name, frac, acc_rf, acc_ssl)
        except Exception as e:
            error_msg = f"Error during experiment for {dataset_name}, {frac*100:.0f}% labels: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            log_error(ERROR_LOG_PATH, dataset_name, frac, str(e) + "\n" + traceback.format_exc())


if __name__ == "__main__":
    run_pipeline()
