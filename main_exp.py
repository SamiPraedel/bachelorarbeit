import numpy as np
import torch
import random
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import inspect

# Local imports
import ssl_exp_config as config
from model_factory import get_model_class # <-- The key import!
from csv_logger import initialize_csv, append_to_csv

# --- Setup ---
warnings.filterwarnings("ignore")
np.random.seed(config.SEED)
random.seed(config.SEED)
torch.manual_seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def semi_split(y_train_tensor, frac, seed=config.SEED):
    """Splits training data indices into labeled and unlabeled."""
    idx_lab, idx_unlab = train_test_split(
        np.arange(len(y_train_tensor.cpu())),
        train_size=frac,
        stratify=y_train_tensor.cpu().numpy(),
        random_state=seed
    )
    return torch.tensor(idx_lab, dtype=torch.long), torch.tensor(idx_unlab, dtype=torch.long)

def run_experiments():
    """Main function to run the semi-supervised learning experiments."""
    print(f"Using device: {device}")
    initialize_csv() # Assumes you have this helper

    for dataset_name, data_loader_fn in config.DATASETS.items():
        print(f"\n{'='*20} DATASET: {dataset_name.upper()} {'='*20}")
        X_tr, y_tr, X_te, y_te = data_loader_fn()
        y_te_np = y_te.cpu().numpy()

        for model_config in config.MODELS:
            model_name = model_config["model_name"]
            runner_key = model_config.get("runner_key")
            trainer_fn = model_config.get("trainer_fn")
            params = model_config["params"]

            for frac in config.LABEL_FRACTIONS:
                print(f"\n  --- Model: {model_name}, Label Fraction: {frac*100:.0f}% ---")
                idx_l, idx_u = semi_split(y_tr, frac)
                X_l, y_l = X_tr[idx_l], y_tr[idx_l]

                ModelClass = get_model_class(model_name)
                sig = inspect.signature(ModelClass.__init__)
                constructor_params = set(sig.parameters.keys())

                # 2. Filter config parameters to only include those accepted by the constructor
                init_params = {k: v for k, v in params.items() if k in constructor_params}

                # 3. Instantiate and train the model based on its runner type
                if runner_key == "FuzzySystem":
                    # Add common/required params if they are in the constructor signature
                    if 'input_dim' in constructor_params:
                        init_params['input_dim'] = X_tr.shape[1]
                    if 'd' in constructor_params:  # Alias for POPFNN
                        init_params['d'] = X_tr.shape[1]

                    if 'num_classes' in constructor_params:
                        init_params['num_classes'] = len(torch.unique(y_tr))
                    if 'C' in constructor_params:  # Alias for POPFNN
                        init_params['C'] = len(torch.unique(y_tr))

                    if 'seed' in constructor_params:
                        init_params['seed'] = config.SEED

                    if 'device' in constructor_params: # For models like FMNC
                        init_params['device'] = device

                    model_instance = ModelClass(**init_params).to(device)
                    
                    if trainer_fn:
                        print(f"      Training a {model_name} instance with {trainer_fn.__name__}...")
                        
                        # Start with a copy of all params from the config
                        all_params = params.copy()
                        trainer_sig = inspect.signature(trainer_fn)
                        
                        # Rename 'epochs' to 'num_epochs' if the trainer expects it
                        if 'num_epochs' in trainer_sig.parameters and 'epochs' in all_params:
                            all_params['num_epochs'] = all_params.pop('epochs')
                        
                        # Now, filter the parameters to only those the trainer function accepts
                        trainer_params = {k: v for k, v in all_params.items() if k in trainer_sig.parameters}

                        # Handle different trainer signatures and return types
                        if trainer_fn.__name__ == 'train_rulespace_ssl':
                            # This trainer requires full dataset context and returns predictions directly
                            y_semi_sup = y_tr.clone().cpu().numpy()
                            y_semi_sup[idx_u.cpu().numpy()] = -1

                            y_pred_np = trainer_fn(
                                model_instance, X_l, y_l, X_tr, y_tr, y_semi_sup, X_te, device,
                                **trainer_params
                            )
                        elif trainer_fn.__name__ == 'train_ifgst':
                            # This trainer needs the unlabeled pool and returns predictions directly
                            X_u_data = X_tr[idx_u]
                            y_pred_np = trainer_fn(
                                model_instance, X_l, y_l, X_u_data, X_te, device,
                                **trainer_params
                            )
                        else: # Handle simple supervised trainers that return a model
                            trainer_fn(model_instance, X_l, y_l, **trainer_params)
                            with torch.no_grad():
                                model_instance.eval()
                                output = model_instance(X_te.to(device))
                                logits = output[0] if isinstance(output, tuple) else output
                                y_pred_np = logits.argmax(dim=1).cpu().numpy()
                    else:
                        print("      No trainer function specified. Using placeholder prediction.")
                        y_pred_np = np.zeros_like(y_te_np)

                elif runner_key == "LabelPropagation":
                    print(f"      Training Label Propagation Baseline...")
                    y_lp_input = np.full_like(y_tr.numpy(), fill_value=-1, dtype=np.int64)
                    y_lp_input[idx_l.cpu().numpy()] = y_l.cpu().numpy()
                    
                    lp_model = ModelClass(**init_params)
                    lp_model.fit(X_tr.numpy(), y_lp_input)
                    y_pred_np = lp_model.predict(X_te.numpy())
                
                else:
                    # Fallback for other sklearn models or unhandled runners
                    print(f"      Runner key '{runner_key}' not explicitly handled. Using basic supervised training.")
                    model_instance = ModelClass(**init_params)
                    model_instance.fit(X_l.numpy(), y_l.numpy())
                    y_pred_np = model_instance.predict(X_te.numpy())


                # 4. Evaluate and Log
                accuracy = accuracy_score(y_te_np, y_pred_np)
                ssl_method_tag = params.get("ssl_method", "supervised") # Default tag
                print(f"      Test Accuracy: {accuracy * 100:.2f}%")

                result_row = [
                    dataset_name,
                    model_name,
                    ssl_method_tag,
                    f"{frac*100:.0f}",
                    f"{accuracy*100:.2f}"
                ]
                append_to_csv(result_row)

if __name__ == "__main__":
    run_experiments()