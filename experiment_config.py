from data_utils import load_iris_data, load_K_chess_data_splitted, load_heart_data, load_wine_data, load_abalon_data


SEED = 42
CSV_FILE_PATH = "ssl_experiment_results_v3.csv"
VISUALIZE_FIRING_STRENGTHS = True

DATASETS = {
    #"Iris": load_iris_data,
    "K_Chess": load_K_chess_data_splitted,
    #"Heart": load_heart_data,
    #"Wine": load_wine_data,
    #"Abalon": load_abalon_data,
}


MODELS = [
    # # --- Teacher-Student SSL Models ---
    # ("HybridANFIS", "train_hybrid_anfis_ssl", {
    #     "ssl_method": "teacher", "num_mfs": 4, "max_rules": 3000, "epochs": 300, "lr": 5e-3
    # }),
    # ("POPFNN", "train_popfnn_ssl", {
    #     "ssl_method": "teacher", "num_mfs": 4, "epochs": 300, "lr": 5e-4
    # }),
    
    #     # --- New Baselines ---
    # # Random Forest Baseline (trained on labeled data only)
    # ('RFBaseline', 'SKLEARN_RF_BASELINE', {
    #     'ssl_method': 'RF_Supervised', # Descriptive tag
    #     'n_estimators': 100, # Example parameter for RandomForestClassifier
    #     'max_depth': None    # Example parameter
    # }),

    # # Label Propagation Baseline
    # ('LabelProp', 'SKLEARN_LABEL_PROP', {
    #     'ssl_method': 'LabelProp', # Descriptive tag
    #     'kernel': 'knn',       # Parameter for LabelPropagation
    #     'n_neighbors': 7,      # Parameter for LabelPropagation (if kernel='knn')
    #     'gamma': 20,           # Parameter for LabelPropagation (if kernel='rbf')
    #     'max_iter': 1000,      # Parameter for LabelPropagation
    #     # Add other LabelPropagation parameters as needed
    # }),
    # --- Rule-Based Self-Training SSL Models ---
    ("HybridANFIS", "train_hybrid_anfis_rule_ssl", {
        "ssl_method": "rule_based", "num_mfs": 4, "max_rules": 3000, "epochs": 200, "lr": 5e-3,
        "initial_train_ratio": 0.2, "rule_conf_thresh": 0.9, "firing_thresh": 0.5
    }),
    ("POPFNN", "train_popfnn_rule_ssl", {
        "ssl_method": "rule_based", "num_mfs": 4, "epochs": 300, "lr": 5e-4,
        "rule_conf_thresh": 0.9
    }),
]

LABEL_FRACTIONS = [0.1, 0.2, 0.5]
TEACHER_CONF_THRESHOLD = 0.90