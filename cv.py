# crossval_main.py  (nur Auszug)

from fmnn_sklearn import FMNN_SK      # <-- hinzugefügt

def cross_val_experiment(dataset_name: str,
                         model_type:   str,
                         num_mfs:      int,
                         max_rules:    int,
                         seed:         int,
                         lr:           float,
                         num_epochs:   int,
                         n_splits:     int):

    X_train, y_train, X_test, y_test = load_dataset(dataset_name, seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # ---------------------------------------------------
    # Modell-Auswahl
    # ---------------------------------------------------
    if model_type == "noHyb":
        model = SomeBaseline(...)
    elif model_type == "anfis":
        model = AnfisWrapper(num_mfs, max_rules, lr, num_epochs)
    elif model_type == "rf":
        model = RandomForestClassifier(n_estimators=200, random_state=seed)
    elif model_type == "fmm":                            #  <<<<<< NEU
        model = FMNN_SK(
            gamma=0.8,
            theta_start=0.6,
            theta_min=0.3,
            theta_decay=0.95,
            bound_mode="max",
            epochs=num_epochs or 5,         # falls 0 übergeben
            onehot=False,                   # ggf. True für One-Hot
            device="cpu"
        )
    else:
        raise ValueError(f"Unknown model_type {model_type!r}")

    # ---------------------------------------------------
    # Cross-Validation
    # ---------------------------------------------------
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=skf, scoring="accuracy")
    model.fit(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    return {
        "model":    model_type,
        "acc_mean": cv_scores.mean(),
        "acc_std":  cv_scores.std(),
        "mcc_mean": 0,      # falls MCC intern schon berechnet → hier eintragen
        "mcc_std":  0,
        "test_acc": test_acc,
    }