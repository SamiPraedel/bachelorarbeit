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