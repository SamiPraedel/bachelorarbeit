# experiment_config.py

from data_utils import (
    load_K_chess_data_splitted,
    # Add other data loaders here if needed
    # load_iris_data, load_heart_data, etc.
)
from trainAnfis import (train_anfis_hybrid, train_anfis_noHyb)
from trainPF import train_popfnn, train_popfnn_ssl
from exp_trainers import train_rulespace_ssl, train_ifgst

# --- General Settings ---
SEED = 42
CSV_FILE_PATH = "ssl_experiment_results_refactored.csv"
VISUALIZE_FIRING_STRENGTHS = True
LABEL_FRACTIONS = [0.1, 0.2, 0.5]

# --- Dataset Configuration ---
DATASETS = {
    "K_Chess": load_K_chess_data_splitted,
    # "Heart": load_heart_data,
}

# --- Model Configuration ---
# This list defines every model and SSL strategy to be tested.
# 'runner_key': Used by the factory to select the correct model handling class.
# 'trainer_fn': The actual function that trains the model.
# 'params': All other parameters for the model and SSL method.
MODELS = [
    # --- Graph-Based SSL with Fuzzy Systems ---
    {
        "model_name": "HybridANFIS",
        "runner_key": "FuzzySystem",
        "trainer_fn": train_rulespace_ssl, # Replace with your actual trainer
        "params": {
            "ssl_method": "iterative",
            "num_mfs": 4,
            "max_rules": 1000,
            "epochs": 20,
            "lr": 5e-3,
        }
    },
    {
        "model_name": "NoHybridANFIS",
        "runner_key": "FuzzySystem",
        "trainer_fn": train_rulespace_ssl, # Replace with your actual trainer
        "params": {
            "ssl_method": "iterative",
            "num_mfs": 4,
            "max_rules": 1000,
            "epochs": 200,
            "lr": 5e-3,
            "zeroG": False, # Add the missing zeroG parameter
        }
    },
    {
        "model_name": "POPFNN",
        "runner_key": "FuzzySystem",
        "trainer_fn": train_rulespace_ssl, # Replace with your actual trainer
        "params": {
            "ssl_method": "iterative",
            "num_mfs": 4,
            "epochs": 200,
            "lr": 5e-4,
            "rule_conf_thresh": 0.9,
        }
    },
        {
        "model_name": "HybridANFIS",
        "runner_key": "FuzzySystem",
        "trainer_fn": train_rulespace_ssl, # Replace with your actual trainer
        "params": {
            "ssl_method": "grf",
            "num_mfs": 4,
            "max_rules": 1000,
            "epochs": 200,
            "lr": 5e-3,
        }
    },
    {
        "model_name": "NoHybridANFIS",
        "runner_key": "FuzzySystem",
        "trainer_fn": train_rulespace_ssl, # Replace with your actual trainer
        "params": {
            "ssl_method": "grf",
            "num_mfs": 4,
            "max_rules": 1000,
            "epochs": 200,
            "lr": 5e-3,
            "zeroG": False, # Add the missing zeroG parameter
        }
    },
    {
        "model_name": "POPFNN",
        "runner_key": "FuzzySystem",
        "trainer_fn": train_rulespace_ssl, # Replace with your actual trainer
        "params": {
            "ssl_method": "grf",
            "num_mfs": 4,
            "epochs": 200,
            "lr": 5e-4,
            "rule_conf_thresh": 0.9,
        }
    },
    

    # --- Scikit-learn Baselines ---
    {
        "model_name": "LabelPropagation",
        "runner_key": "LabelPropagation",
        "trainer_fn": None, # Not needed for sklearn models
        "params": {
            "ssl_method": "LabelProp", # Descriptive tag
            "kernel": "knn",
            "n_neighbors": 7,
            "max_iter": 1000,
        }
    },
    
]