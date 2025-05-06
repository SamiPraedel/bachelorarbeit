# run_one_dataset.py
import matplotlib
matplotlib.use("Agg")       
import matplotlib.pyplot as plt
import pandas as pd
from crossval_main import cross_val_experiment  


DATASET   = "ChessK"          
N_SPLITS  = 5
MFs       = 3
MAX_RULES = 729        
EPOCHS    = 300
LR        = 1e-3
SEED      = 42


results = []
for model_name in ["noHyb", "anfis", "rf"]:
    res = cross_val_experiment(
        dataset_name = DATASET,
        model_type   = model_name,
        num_mfs      = MFs,
        max_rules    = MAX_RULES,
        seed         = SEED,
        lr           = LR,
        num_epochs   = EPOCHS if model_name != "rf" else 0,
        n_splits     = N_SPLITS,
    )
    results.append(res)

df = pd.DataFrame(results)
print("\n", df, "\n")

# Matplotlib‑Tabelle
fig, ax = plt.subplots(figsize=(6, 1.2 + 0.4 * len(df)))
ax.axis("off")

tbl = ax.table(
    cellText=df[["model", "acc_mean", "acc_std", "mcc_mean", "mcc_std"]]
              .round(4).values,
    colLabels=["Model", "ACC µ", "ACC σ", "MCC µ", "MCC σ"],
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.3)

plt.tight_layout()
plt.savefig(f"{DATASET}_cv_table.png", dpi=300)
print(f"✔  Tabelle gespeichert:  {DATASET}_cv_table.png")


plt.figure(figsize=(5, 3))
plt.bar(df["model"], df["acc_mean"], yerr=df["acc_std"], capsize=4)
plt.ylabel("Accuracy (mean ± std)")
plt.title(f"{DATASET} – {k:=N_SPLITS}‑Fold CV")
plt.ylim(0, 1)
plt.xticks(rotation=0, ha="center")
plt.tight_layout()
plt.savefig(f"{DATASET}_cv_bar.png", dpi=300)
print(f"✔  Bar‑Plot gespeichert: {DATASET}_cv_bar.png")
