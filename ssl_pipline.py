# unified_ssl_pipeline.py
# ------------------------------------------------------------
import numpy as np
import torch
import torch.nn.functional as F
import random
import warnings
import csv # Added import for CSV writing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Assuming these files are in the same directory or Python path
from data_utils import load_iris_data, load_K_chess_data_splitted, load_Kp_chess_data_ord, load_heart_data, load_wine_data, load_abalon_data, load_K_chess_data_OneHot, load_Poker_data
# from pop_ssl_utils import rule_class_mapping # Optional: if you want to use this for POPFNN
from trainAnfis import train_hybrid_anfis_ssl, train_nohybrid_anfis_ssl
from trainPF import train_popfnn_ssl

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

# ---------- CSV Logging Helper ---------------------------------------------
CSV_FILE_PATH = "ssl_experiment_results.csv"

# Define column headers and their desired widths for alignment
CSV_HEADERS = [
    "Dataset", "Model", "Label_Fraction_Percent", "Teacher_Threshold", "Seed",
    "Epochs", "Learning_Rate", "Num_MFs", "Max_Rules", "ZeroG",
    "Num_Labeled", "Num_Pseudo_Labeled", "Avg_Teacher_Confidence",
    "RF_Teacher_Accuracy_Percent", "SSL_Model_Accuracy_Percent"
]

COLUMN_WIDTHS = [
    15,  # Dataset
    18,  # Model
    24,  # Label_Fraction_Percent
    18,  # Teacher_Threshold
    6,   # Seed
    8,   # Epochs
    15,  # Learning_Rate
    10,  # Num_MFs
    10,  # Max_Rules
    7,   # ZeroG
    14,  # Num_Labeled
    20,  # Num_Pseudo_Labeled
    24,  # Avg_Teacher_Confidence
    28,  # RF_Teacher_Accuracy_Percent
    28   # SSL_Model_Accuracy_Percent
]

def format_for_csv_row(data_list, widths):
    """Pads each item in the list to the specified width."""
    return [str(item).ljust(widths[i]) for i, item in enumerate(data_list)]

def initialize_csv():
    """Initializes the CSV file with headers."""
    with open(CSV_FILE_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(format_for_csv_row(CSV_HEADERS, COLUMN_WIDTHS))

def append_to_csv(data_row):
    """Appends a row of data to the CSV file."""
    with open(CSV_FILE_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(format_for_csv_row(data_row, COLUMN_WIDTHS))

# ---------- Main Experiment Loop ------------------------------------------
def run_experiments(datasets_config, models_config, label_fractions, teacher_threshold):
    print(f"Using device: {device}")
    initialize_csv() # Initialize CSV file at the beginning of experiments

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
                
                # Extract model-specific hyperparameters for logging
                epochs_log = model_specific_params.get("epochs", "N/A")
                lr_log = model_specific_params.get("lr", "N/A")
                num_mfs_log = model_specific_params.get("num_mfs", "N/A")
                max_rules_log = model_specific_params.get("max_rules", "N/A")
                zero_g_log = model_specific_params.get("zeroG", "N/A")

                # Log results to CSV
                result_row = [
                    dataset_name,
                    model_name,
                    f"{frac*100:.0f}",
                    f"{teacher_threshold:.2f}",
                    SEED,
                    epochs_log,
                    f"{lr_log:.1e}" if isinstance(lr_log, float) else lr_log,
                    num_mfs_log,
                    max_rules_log,
                    zero_g_log,
                    len(X_l_tensor),
                    len(X_p_tensor),
                    f"{avg_conf:.4f}",
                    f"{rf_acc*100:.2f}",
                    f"{ssl_model_acc*100:.2f}"
                ]
                append_to_csv(result_row)
                # Optional: POPFNN specific output
                # if model_name == "POPFNN" and hasattr(trained_model, 'get_rules_with_classes'):
                #     print("      POPFNN Rule Class Mapping (Top 3):")
                #     rule_class_mapping(trained_model, top_k=3) # Requires pop_ssl_utils

if __name__ == "__main__":

    DATASETS = {
        "Iris": load_iris_data,
        "K_Chess": load_K_chess_data_splitted,
        "Heart": load_heart_data,
        "Wine": load_wine_data,
        "Abalon": load_abalon_data,
        #"Kp_Chess": load_Kp_chess_data_ord
    }



    MODELS = [
        ("HybridANFIS", train_hybrid_anfis_ssl, {
            "num_mfs": 4, "max_rules": 1000, "epochs": 200, "lr": 5e-3
        }),
        ("POPFNN", train_popfnn_ssl, {
            "num_mfs": 4, "epochs": 300, "lr": 5e-4
        }),
        ("NoHybridANFIS", train_nohybrid_anfis_ssl, {
            "num_mfs": 4, "max_rules": 1000, "zeroG": False, "epochs": 200, "lr": 5e-3
        }),
    ]

    LABEL_FRACTIONS = [0.1, 0.2, 0.3, 0.5] # Example fractions
    TEACHER_CONF_THRESHOLD = 0.90

    run_experiments(DATASETS, MODELS, LABEL_FRACTIONS, TEACHER_CONF_THRESHOLD)
