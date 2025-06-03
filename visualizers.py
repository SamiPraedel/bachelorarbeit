# visualizers.py
# ------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_firing_strengths(
    firing_strengths_np: np.ndarray,
    true_labels_np: np.ndarray,
    dataset_name: str,
    model_display_name: str,
    label_fraction_percentage: float,
    base_viz_path: str = "visualizations"
):

    os.makedirs(base_viz_path, exist_ok=True)
    plot_filename = os.path.join(base_viz_path, f"fs_{dataset_name}_{model_display_name}_{label_fraction_percentage:.0f}.png")

    plt.figure(figsize=(10, 7))
    if firing_strengths_np.shape[1] >= 2:
        scatter = plt.scatter(firing_strengths_np[:, 0], firing_strengths_np[:, 1],
                              c=true_labels_np, cmap='viridis', alpha=0.6, s=10)
        plt.colorbar(scatter, label='True Class')
        plt.xlabel("Firing Strength Dimension 1")
        plt.ylabel("Firing Strength Dimension 2")
        plt.title(f"Firing Strengths (First 2D)\n{model_display_name} on {dataset_name} ({label_fraction_percentage:.0f}% labeled)")
    elif firing_strengths_np.shape[1] == 1:
        for cls_label in np.unique(true_labels_np):
            plt.hist(firing_strengths_np[true_labels_np == cls_label, 0],
                     bins=30, alpha=0.6, label=f'Class {cls_label}')
        plt.xlabel("Firing Strength Value")
        plt.ylabel("Frequency")
        plt.title(f"Firing Strength Distribution\n{model_display_name} on {dataset_name} ({label_fraction_percentage:.0f}% labeled)")
        plt.legend()
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
    print(f"      Saved firing strength visualization to {plot_filename}")

# fitviz.py ---------------------------------------------------------------
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import learning_curve, validation_curve

sns.set(style="whitegrid")

def plot_epoch_curves(history, metric="loss", ax=None, smooth=0):
    """
    history : dict – keys 'train_loss','val_loss','train_acc','val_acc', ...
    metric  : 'loss' | 'acc' (oder anderer Schlüsselvorsatz)
    smooth  : int   – Glättungsfenster (moving average), 0 = off
    """
    tr_key = f"train_{metric}"
    va_key = f"val_{metric}"

    if tr_key not in history:
        print(f"Warning: Metric '{tr_key}' not found in history. Skipping plot.")
        return ax if ax is not None else plt.gca()

    tr_values = np.array(history[tr_key])
    epochs_tr = np.arange(1, len(tr_values) + 1)

    has_val = va_key in history
    if has_val:
        va_values = np.array(history[va_key])
        epochs_va = np.arange(1, len(va_values) + 1)

    if smooth > 0:
        if len(tr_values) >= smooth:
            k_smooth = np.ones(smooth) / smooth
            tr_values = np.convolve(tr_values, k_smooth, mode='valid')
            epochs_tr = np.arange(len(tr_values)) + smooth // 2 + 1
        if has_val and len(va_values) >= smooth:
            k_smooth_val = np.ones(smooth) / smooth
            va_values = np.convolve(va_values, k_smooth_val, mode='valid')
            epochs_va = np.arange(len(va_values)) + smooth // 2 + 1

    if ax is None: ax = plt.gca()
    ax.plot(epochs_tr, tr_values, label=f"train {metric}")
    if has_val:
        ax.plot(epochs_va, va_values, label=f"val {metric}")
    ax.set_xlabel("Epoch"); ax.set_ylabel(metric.capitalize())
    ax.set_title(f"{metric.capitalize()} vs Epochs")
    ax.legend(); return ax

def sklearn_learning_curve(estimator, X, y, metric, cv=5,
                           train_sizes=np.linspace(0.1,1.0,8), ax=None):
    tr, va = learning_curve(estimator, X, y,
                            train_sizes=train_sizes, cv=cv,
                            scoring=metric, n_jobs=-1, shuffle=True,
                            random_state=42)[:2]
    tr_mean, tr_std = tr.mean(1), tr.std(1)
    va_mean, va_std = va.mean(1), va.std(1)
    if ax is None: ax = plt.gca()
    ax.fill_between(train_sizes, tr_mean-tr_std, tr_mean+tr_std, alpha=.2)
    ax.fill_between(train_sizes, va_mean-va_std, va_mean+va_std, alpha=.2)
    ax.plot(train_sizes, tr_mean, "o-", label="train")
    ax.plot(train_sizes, va_mean, "o-", label="val")
    ax.set_xlabel("Train set fraction"); ax.set_ylabel(metric)
    ax.set_title("Learning curve"); ax.legend(); return ax


def sklearn_validation_curve(estimator, X, y, metric,
                             param_name, param_range, cv=5, ax=None):
    tr, va = validation_curve(estimator, X, y,
                              param_name, param_range,
                              scoring=metric, cv=cv, n_jobs=-1)
    tr_mean, va_mean = tr.mean(1), va.mean(1)
    if ax is None: ax = plt.gca()
    ax.plot(param_range, tr_mean, "o-", label="train")
    ax.plot(param_range, va_mean, "o-", label="val")
    ax.set_xlabel(param_name); ax.set_ylabel(metric)
    ax.set_title("Validation curve"); ax.legend(); return ax
