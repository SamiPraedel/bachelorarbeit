from data_utils import load_K_chess_data_splitted


SEED = 42
CSV_FILE_PATH = "ssl_experiment_results_v3.csv"
VISUALIZE_FIRING_STRENGTHS = True

DATASETS = {
    #"Iris": load_iris_data,
    #"Shuttle": load_shuttle_data,
    #"Gamma": load_gamma_data,
    "K_Chess": load_K_chess_data_splitted,
    #"Heart": load_heart_data,
    #"Wine": load_wine_data,
    #"Abalon": load_abalon_data,
}

LP_Methods = ["grf", "iterative"]



# MODELS = {
#     HybridAnfis, NoHybridAnfis, PopFNN, PopFNN, FMMC, RandomForest, LabelPropagation,}


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

    # Label Propagation Baseline
    """('LabelProp', 'SKLEARN_LABEL_PROP', {
        'ssl_method': 'LabelProp', # Descriptive tag
        'kernel': 'knn',       # Parameter for LabelPropagation
        'n_neighbors': 7,      # Parameter for LabelPropagation (if kernel='knn')
        'gamma': 20,           # Parameter for LabelPropagation (if kernel='rbf')
        'max_iter': 1000,      # Parameter for LabelPropagation
        # Add other LabelPropagation parameters as needed
    }),"""
    
    # --- Graph Based  SSL Models ---
    ("HybridANFIS", "train_hybrid_anfis_ssl", {
        "ssl_method": "iterative", "num_mfs": 4, "max_rules": 1000, "epochs": 300, "lr": 5e-3,
        
    }),
    ("NoHybridANFIS", "train_hybrid_anfis_ssl", {
        "ssl_method": "grf", "num_mfs": 4, "max_rules": 1000, "epochs": 300, "lr": 5e-3,
        
    }),
     ("POPFNN", "train_popfnn_rule_ssl", {
        "ssl_method": "grf", "num_mfs": 4, "epochs": 200, "lr": 5e-4,
        "rule_conf_thresh": 0.9
    }),
]

LABEL_FRACTIONS = [0.1, 0.2, 0.5]
TEACHER_CONF_THRESHOLD = 0.90