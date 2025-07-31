################ experiment_config.py ################
"""
Zentrale Hyper-Parameter-Sammlung für alle SSL-Experimente.
Alle Skripte importieren einfach `import experiment_config as cfg`
und greifen dann auf `cfg.SETTINGS[...]` zu.
"""

SETTINGS = {
    # --------------------------------------------------
    #  DATA & LABEL-SPLIT
    # --------------------------------------------------
    "dataset"    : "KChess",   # Name wie in data_utils
    "label_frac" : 0.10,       # Anteil gelabelter Beispiele

    # --------------------------------------------------
    #  WELCHE FEATURE-EXTRAKTOREN TRAINIEREN?
    # --------------------------------------------------
    "models" : [
        "NoHybridANFIS",
        "HybridANFIS",
        "POPFNN",
        "FMNC",                 # Fuzzy-Min-Max-Classifier
    ],

    # --------------------------------------------------
    #  GRID für k-NN Grafs (k, σ)   -> über Schleife durchprobieren
    # --------------------------------------------------
    "graph_grid" : [
        (7,  0.5),
        (15, 0.6),
        (15, 0.3),
        (25, 1.0),
    ],

    # --------------------------------------------------
    #  Hyper-Parameter: MV-GRF
    # --------------------------------------------------
    "grf_params" : {
        "sigma_m" : 1.0,
        "sigma_r" : 0.5,
        "beta"    : 0.5,
    },

    # --------------------------------------------------
    #  Hyper-Parameter: FMV-CLP
    # --------------------------------------------------
    "fmv_params" : {
        "sigma_m" : 1.0,
        "sigma_r" : 0.5,
        "beta"    : 0.5,
        "alpha"   : 0.9,
        "k_thr"   : 1.0,
    },

    # --------------------------------------------------
    #  Hyper-Parameter: Rule-Self-Training
    # --------------------------------------------------
    "rst_params" : {
        "tau_conf"       : 0.90,
        "tau_fire"       : 0.50,
        "max_rounds"     : 10,
        "warm_epochs"    : 200,
        "retrain_epochs" : 40,
        "lr_premise"     : 5e-3,
        "lr_conseq"      : 5e-3,
    },

    # --------------------------------------------------
    #  Default Hyper-Parameter: Model-specific
    # --------------------------------------------------
    "model_params" : {
        # Example default hyper‑params; change per run
        "NoHybridANFIS" : {"num_mfs": 4, "max_rules": 1000, "lr": 1e-2, "epochs": 400},
        "HybridANFIS"   : {"num_mfs": 4, "max_rules": 1000, "lr": 1e-2, "epochs": 400},
        "POPFNN"        : {"num_mfs": 4, "lr": 1e-2, "epochs": 400},
        "FMNC"          : {"gamma":1.7, "theta0":2.5, "theta_min":0.1, "theta_decay":0.1, "bound_mode":"sum", "aggr":"mean", "m_min":0.8},
    },
}