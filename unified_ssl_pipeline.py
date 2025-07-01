# unified_ssl_pipeline.py
# ------------------------------------------------------------
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import warnings
import csv
import os # Added for path operations for saving visualizations
import matplotlib.pyplot as plt # Added for plotting



import experiment_config as config
from csv_logger import initialize_csv, append_to_csv # Import specific functions
from visualizers import plot_firing_strengths # Import the new plotting function




from teacher_student_trainers import (
    train_hybrid_anfis_ssl, train_nohybrid_anfis_ssl, train_popfnn_ssl
)
from rule_based_trainers import (
    train_hybrid_anfis_rule_ssl, train_popfnn_rule_ssl
)
from lb_scratch import GraphSSL


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, silhouette_score # Added silhouette_score
from sklearn.semi_supervised import LabelPropagation # Added for Label Propagation baseline
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
np.random.seed(config.SEED)
random.seed(config.SEED)
torch.manual_seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def semi_split(y_train_tensor: torch.Tensor, frac: float, seed: int = config.SEED): # Use config.SEED as default
    """Splits training data indices into labeled and unlabeled."""
    idx_lab, idx_unlab = train_test_split(
        np.arange(len(y_train_tensor.cpu())), 
        train_size=frac,
        stratify=y_train_tensor.cpu().numpy(),
        random_state=seed
    )
    return torch.tensor(idx_lab, dtype=torch.long), torch.tensor(idx_unlab, dtype=torch.long)

def rf_teacher_pseudo_labels(X_train_np: np.ndarray, y_train_np: np.ndarray,
                             idx_lab: np.ndarray, idx_unlab: np.ndarray,
                             thr: float = 0.9, seed: int = config.SEED): # Use config.SEED as default
    """Trains a RandomForest teacher and generates pseudo-labels."""
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, n_jobs=-1, random_state=seed)
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

# ---------- Trainer Function Mapping --------------------------------------
# Used to resolve trainer function names from the configuration
TRAINER_MAPPING = {
    "train_hybrid_anfis_ssl": train_hybrid_anfis_ssl,
    "train_popfnn_ssl": train_popfnn_ssl,
    "train_hybrid_anfis_rule_ssl": train_hybrid_anfis_rule_ssl,
    "train_popfnn_rule_ssl": train_popfnn_rule_ssl,
    "train_nohybrid_anfis_ssl": train_nohybrid_anfis_ssl
    "train_nohybrid_anfis_lp_"
}
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
        X_te_np = X_te_tensor.cpu().numpy() # Moved here for broader access
        y_te_np = y_te_tensor.cpu().numpy() # Moved here for broader access

        for model_name, model_config_key, model_params_from_config in models_config:
            
            for frac in label_fractions:
                idx_l, idx_u = semi_split(y_tr_tensor, frac, seed=config.SEED)
                X_l, y_l = X_tr_tensor[idx_l].to(device), y_tr_tensor[idx_l].to(device)
                X_l_np, y_l_np = X_l.cpu().numpy(), y_l.cpu().numpy() # For sklearn models

                # Initialize common variables
                trained_model_obj = None
                y_pred_ssl_np = None
                predictions_output = None # For PyTorch model outputs (potentially with firing strengths)
                teacher_acc_str = "N/A"
                num_pseudo_labels_for_print = 0 # For the print statement
                ssl_model_acc = 0.0
                
                current_params = model_params_from_config.copy()
                current_params['seed'] = config.SEED # Pass seed to trainers
                
                # Determine ssl_method_tag for display and logging (comes from config)
                # This tag also helps determine how the model is handled (PyTorch SSL, RF Baseline, LP Baseline)
                ssl_method_tag = current_params.get("ssl_method", "UnknownSSL")
                model_display_name = f"{model_name}_{ssl_method_tag.upper()}"
                print(f"\n  --- Model: {model_display_name}, Label Fraction: {frac*100:.0f}% ---")

                if model_config_key == "SKLEARN_RF_BASELINE":
                    print(f"    Training RF Baseline...")
                    # rf_baseline = RandomForestClassifier(n_estimators=current_params.get('n_estimators', 100), 
                    #                                      max_depth=current_params.get('max_depth', None),
                    #                                      random_state=config.SEED, n_jobs=-1)
                    rf_baseline = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=config.SEED)
                    rf_baseline.fit(X_l_np, y_l_np)
                    y_pred_ssl_np = rf_baseline.predict(X_te_np)
                    trained_model_obj = rf_baseline # Store for consistency, though not used further in this path
                    num_pseudo_labels_for_print = 0 # No pseudo-labels in this context
                    print("hasenhüttle")

                elif model_config_key == "SKLEARN_LABEL_PROP":
                    print(f"    Training Label Propagation Baseline...")
                    # Prepare labels for LabelPropagation: known labels and -1 for unlabeled
                    y_lp_input = np.full_like(y_tr_np, fill_value=-1, dtype=np.int64)
                    y_lp_input[idx_l.cpu().numpy()] = y_l_np
                    
                    lp_params = {k: v for k, v in current_params.items() if k not in ['ssl_method', 'seed']}
                    lp_model = LabelPropagation(**lp_params)
                    
                    lp_model.fit(X_tr_np, y_lp_input)
                    y_pred_ssl_np = lp_model.predict(X_te_np)
                    trained_model_obj = lp_model
                    num_pseudo_labels_for_print = len(idx_u) # All unlabeled data is used by LP
                   # print('Accuracy: ', metrics.accuracy_score(y_test,y_pred))
                    
                    
                    #true_np = np.asarray(y_l_np)                                      
                    #mask_unlabeled = random_unlabeled_points  
                   # pseudo_acc = np.mean(true_np[mask_unlabeled] == pseudo[mask_unlabeled])
                    #pseudo_acc = np.mean(true_np == y_lp_input) # Overall accuracy for pseudo-labels
                    #print(f"Pseudo-Label Accuracy: {pseudo_acc*100:.2f}%")
      

                elif model_config_key in TRAINER_MAPPING: # Existing PyTorch SSL models
                    model_trainer_fn = TRAINER_MAPPING[model_config_key]
                    
                    ssl_threshold_log = f"{teacher_conf_threshold:.2f}"

                    if ssl_method_tag.lower() == "teacher": # Check against the tag from config
                        X_p_np, y_p_np, w_p_np, rf_teacher, avg_conf = rf_teacher_pseudo_labels(
                            X_tr_np, y_tr_np, idx_l.cpu().numpy(), idx_u.cpu().numpy(),
                            thr=teacher_conf_threshold, seed=config.SEED)
                        
                        X_p = torch.from_numpy(X_p_np).float().to(device)
                        y_p = torch.from_numpy(y_p_np).long().to(device)
                        w_p = torch.from_numpy(w_p_np).float().to(device)
                        
                        num_pseudo_labels_for_print = len(X_p)
                        # avg_pseudo_conf_str = f"{avg_conf:.4f}" # If needed for logging
                        teacher_acc_str = f"{accuracy_score(y_te_np, rf_teacher.predict(X_te_np))*100:.2f}"
                        
                        trained_model_obj = model_trainer_fn(
                            X_l, y_l, X_p, y_p, w_p,
                            input_dim, num_classes, device, **current_params)

                    elif ssl_method_tag.lower() == "rule_based": # Check against the tag from config
                        X_u = X_tr_tensor[idx_u].to(device)
                        # rule_thr = current_params.get('rule_conf_thresh', 'N/A') # If needed for logging
                        # fire_thr = current_params.get('firing_thresh', 'N/A') # If needed for logging
                        # ssl_threshold_log = f"R:{rule_thr}|F:{fire_thr}" # If needed for logging
                        
                        trained_model_obj, num_pseudo_from_trainer = model_trainer_fn(
                            X_l, y_l, X_u,
                            input_dim, num_classes, device, **current_params)
                        num_pseudo_labels_for_print = num_pseudo_from_trainer
                   
                    else:
                        print(f"      Warning: Unknown ssl_method '{ssl_method_tag}' for PyTorch model {model_name}. Skipping.")
                        continue # Skip to next fraction or model

                    # Evaluation for PyTorch models
                    if trained_model_obj:
                        trained_model_obj.eval()
                        with torch.no_grad():
                            predictions_output = trained_model_obj(X_te_tensor.to(device))
                        
                        # --- Graph-based SSL integration ---
                        if ssl_method_tag.lower() in ["grf", "iterative"]:
                            # Ensure rule-based models return firing strengths
                            if not (isinstance(predictions_output, tuple) and len(predictions_output) > 1):
                                print(f"      Warning: {model_name} with '{ssl_method_tag}' SSL requires firing strengths but model output is not in the expected format. Skipping.")
                                continue  # Skip if firing strengths are not available
                            
                            firing_strengths_train = predictions_output[1].cpu()

                            # Prepare semi-supervised labels for graph SSL
                            y_semi_sup = np.full(len(y_tr_np), -1, dtype=np.int64)
                            y_semi_sup[idx_l.cpu().numpy()] = y_l_np
                            
                            # Initialize and fit the appropriate GraphSSL model
                            graph_ssl_model = GraphSSL(method=ssl_method_tag.lower(), device="cpu")  # GraphSSL works on CPU
                            graph_ssl_model.fit(firing_strengths_train, y_semi_sup)

                            # Predict and evaluate
                            with torch.no_grad():
                                _, test_firing_strengths, _ = trained_model_obj(X_te_tensor.to(device))
                                y_pred_ssl_np = graph_ssl_model.predict(test_firing_strengths.cpu())

                            # Override model output for accuracy calculation
                            predictions_output = (torch.tensor(y_pred_ssl_np), )

                            print(f"      Applied graph-based SSL ({ssl_method_tag.upper()}).")

                        # --- End Graph-based SSL ---



                        final_model_output = predictions_output[0] if isinstance(predictions_output, tuple) else predictions_output
                        y_pred_ssl_torch = final_model_output.argmax(dim=1).cpu()
                        y_pred_ssl_np = y_pred_ssl_torch.numpy()
                else:
                    print(f"      Warning: Unknown model_config_key '{model_config_key}'. Skipping.")
                    continue # Skip to next fraction or model

                print(f"      Labeled: {len(X_l)}, Pseudo-involved/Unlabeled used: {num_pseudo_labels_for_print}")
                
                # --- Calculate SSL Model Accuracy (common for all paths if y_pred_ssl_np is set) ---
                if y_pred_ssl_np is not None:
                    ssl_model_acc = accuracy_score(y_te_np, y_pred_ssl_np)
                    print(f"      Teacher Acc: {teacher_acc_str}% | Student Acc: {ssl_model_acc*100:5.2f}%")
                else:
                    print(f"      WARNING: y_pred_ssl_np not set for {model_display_name}. Accuracy will be 0.")
                    ssl_model_acc = 0.0
                
                # --- Silhouette Score Calculation ---
                # Sil_Feat_True and Sil_Feat_Pred have been removed as per request.
                # Only Sil_Firing_True remains.
                firing_strengths_silhouette_str = "N/A"


                # --- Firing Strength Analysis and Visualization ---
                # This section is primarily for PyTorch models that return firing strengths
                if predictions_output is not None and isinstance(predictions_output, tuple) and len(predictions_output) > 1:
                    firing_strengths = predictions_output[1].cpu().numpy()
                    print(f"      Firing strengths detected with shape: {firing_strengths.shape}")

                    fs_for_silhouette = firing_strengths.reshape(firing_strengths.shape[0], -1) # Ensure 2D
                    
                    if fs_for_silhouette.shape[0] == len(y_te_np) and fs_for_silhouette.shape[1] > 0 and y_pred_ssl_np is not None:
                        try:
                            if len(np.unique(y_te_np)) > 1 and fs_for_silhouette.shape[0] > 1:
                                score = silhouette_score(fs_for_silhouette, y_te_np)
                                firing_strengths_silhouette_str = f"{score:.4f}"
                                print(f"      Silhouette (Firing Strengths, True Labels): {score:.4f}")
                            else:
                                print("      Silhouette (Firing Strengths, True Labels): N/A (not enough unique labels or samples)")
                        except ValueError as e:
                            print(f"      Error calculating Silhouette on firing strengths: {e}")
                            firing_strengths_silhouette_str = "N/A (ValueError)"

                        # --- Visualization of Firing Strengths ---
                        if hasattr(config, 'VISUALIZE_FIRING_STRENGTHS') and config.VISUALIZE_FIRING_STRENGTHS:
                            plot_firing_strengths(
                                firing_strengths_np=fs_for_silhouette,
                                true_labels_np=y_te_np,
                                dataset_name=dataset_name,
                                model_display_name=model_display_name,
                                label_fraction_percentage=frac*100,
                                base_viz_path="visualizations" # Default, can be configured
                            )
                else:
                    if model_config_key in TRAINER_MAPPING: # Only print for models expected to have them
                        print("      Firing strengths not available or not in expected format from model output.")

                # Logging
                result_row = [
                    dataset_name, model_name, ssl_method_tag, f"{frac*100:.0f}", teacher_acc_str,
                    f"{ssl_model_acc*100:.2f}", firing_strengths_silhouette_str
                ]
                append_to_csv(result_row)

if __name__ == "__main__":
    # Load configurations from the config file
    run_experiments(
        config.DATASETS,
        config.MODELS,
        config.LABEL_FRACTIONS,
        config.TEACHER_CONF_THRESHOLD
    )