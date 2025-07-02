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

# ---------- Trainer Function Mapping --------------------------------------
# Used to resolve trainer function names from the configuration
TRAINER_MAPPING = {
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
                num_pseudo_labels_for_print = 0 # For the print statement
                ssl_model_acc = 0.0
                
                current_params = model_params_from_config.copy()
                current_params['seed'] = config.SEED # Pass seed to trainers
                
                # Determine ssl_method_tag for display and logging (comes from config)
                # This tag also helps determine how the model is handled (PyTorch SSL, RF Baseline, LP Baseline)
                ssl_method_tag = current_params.get("ssl_method", "UnknownSSL")
                model_display_name = f"{model_name}_{ssl_method_tag.upper()}"
                print(f"\n  --- Model: {model_display_name}, Label Fraction: {frac*100:.0f}% ---")



                if model_config_key == "SKLEARN_LABEL_PROP":
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
      

                elif model_config_key in TRAINER_MAPPING: # Existing PyTorch SSL models
                    model_trainer_fn = TRAINER_MAPPING[model_config_key]
                    
                    X_u = X_tr_tensor[idx_u].to(device)
                    
                    trained_model_obj, num_pseudo_from_trainer = model_trainer_fn(
                        X_l, y_l, X_u,
                        input_dim, num_classes, device, **current_params)
                    num_pseudo_labels_for_print = num_pseudo_from_trainer

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

                print(f"      Labeled: {len(X_l)}, Pseudo-involved/Unlabeled used: {num_pseudo_labels_for_print}")
                
                # --- Calculate SSL Model Accuracy 
                ssl_model_acc = accuracy_score(y_te_np, y_pred_ssl_np)
                print(f"      Acc: {ssl_model_acc*100:5.2f}%")

                # --- Firing Strength Analysis and Visualization ---
                if predictions_output is not None and isinstance(predictions_output, tuple) and len(predictions_output) > 1:
                    firing_strengths = predictions_output[1].cpu().numpy()
                    print(f"      Firing strengths detected with shape: {firing_strengths.shape}")
                    if hasattr(config, 'VISUALIZE_FIRING_STRENGTHS') and config.VISUALIZE_FIRING_STRENGTHS:
                        plot_firing_strengths(
                            firing_strengths_np=firing_strengths,
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
                    dataset_name, model_name, ssl_method_tag, f"{frac*100:.0f}", f"{ssl_model_acc*100:.2f}"
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