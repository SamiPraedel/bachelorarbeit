import umap

def plot_firing_strengths(model, X, cmap='viridis'):

    model.eval()
    with torch.no_grad():
        _, norm_fs, _ = model(X)
    
    # norm_fs hat Shape [N, num_rules]
    norm_fs_np = norm_fs.cpu().numpy()
    
    # Wähle für jeden Datenpunkt z. B. den maximalen Firing Strength-Wert als Farbe
    colors = norm_fs_np.max(axis=1)
    
    # UMAP-Anwendung: Reduziere die Dimension von norm_fs auf 2
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(norm_fs_np)
    
    # Plot erstellen
    plt.figure(figsize=(10,8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=cmap, s=50)
    plt.colorbar(scatter, label='Max Firing Strength')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title('UMAP-Visualisierung der Firing Strengths')
    plt.grid(True)
    plt.show()
    

def plot_firing_strength_heatmap(self, x_min, x_max, y_min, y_max, grid_size=100, rule_index=None):
        """
        Erzeugt eine Heatmap der Firing Strengths über einen 2D-Eingaberaum.
        Annahme: Das Modell hat 2 Eingangsdimensionen.
        
        :param x_min: Minimaler x-Wert.
        :param x_max: Maximaler x-Wert.
        :param y_min: Minimaler y-Wert.
        :param y_max: Maximaler y-Wert.
        :param grid_size: Auflösung des Gitters.
        :param rule_index: Index einer spezifischen Regel. Falls None, wird der maximale Firing Strength-Wert verwendet.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        # Erzeuge ein Gitter im Eingaberaum
        x_vals = np.linspace(x_min, x_max, grid_size)
        y_vals = np.linspace(y_min, y_max, grid_size)
        xx, yy = np.meshgrid(x_vals, y_vals)
        # Erstelle Eingabepunkte (hier gehen wir davon aus, dass es 2 Features gibt)
        grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
        
        # Führe einen Forward-Pass durch, um die Firing Strengths zu erhalten
        self.eval()
        with torch.no_grad():
            _, firing_strengths, _ = self(grid_tensor)
        
        # firing_strengths hat die Form [N, num_rules]
        if rule_index is not None:
            # Wähle den Firing Strength-Wert der angegebenen Regel
            selected_fs = firing_strengths[:, rule_index].cpu().numpy()
        else:
            # Wähle für jeden Punkt den maximalen Firing Strength-Wert
            selected_fs = firing_strengths.max(dim=1)[0].cpu().numpy()
        
        # Reshape in die Gitterform
        selected_fs = selected_fs.reshape(grid_size, grid_size)
        
        # Plotten der Heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(selected_fs, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis')
        plt.colorbar(label="Firing Strength")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Firing Strength Heatmap" + (f" (Regel {rule_index})" if rule_index is not None else " (maximal)"))
        #plt.show()

def plot_umap_fixed_rule(model, X, rule_index, cmap='viridis'):
    """
    Erzeugt einen UMAP-Plot der Firing Strengths, wobei die Punkte
    nach der Aktivierung einer festen Regel (rule_index) eingefärbt werden.
    
    Args:
        model: Das ANFIS-Modell.
        X: Eingabedaten (Tensor, z. B. [N, input_dim]).
        rule_index: Index der festen Regel, deren Aktivierung zur Farbgebung verwendet wird.
        cmap: Colormap für den Plot.
    """
    import umap.umap_ as umap
    import matplotlib.pyplot as plt
    
    model.eval()
    with torch.no_grad():
        # Berechne die normalized firing strengths: Shape [N, num_rules]
        _, firing_strengths, _ = model(X)
    
    # Konvertiere in ein NumPy-Array
    firing_strengths_np = firing_strengths.cpu().numpy()  # [N, num_rules]
    
    # Prüfe, ob der Regelindex gültig ist
    if rule_index < 0 or rule_index >= firing_strengths_np.shape[1]:
        raise ValueError(f"rule_index {rule_index} liegt außerhalb des gültigen Bereichs [0, {firing_strengths_np.shape[1]-1}]")
    
    # Verwende UMAP, um die gesamte Firing-Strength-Vektor (über alle Regeln) auf 2D zu reduzieren
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(firing_strengths_np)  # [N, 2]
    
    # Verwende den Firing-Strength-Wert der festen Regel als Farbe
    colors = firing_strengths_np[:, rule_index]
    
    plt.figure(figsize=(10,8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=cmap, s=50)
    plt.colorbar(scatter, label=f'Firing Strength von Regel {rule_index}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title(f'UMAP Plot der Firing Strengths (Feste Regel {rule_index})')
    plt.grid(True)
    plt.show()

import matplotlib.pyplot as plt
import torch

def plot_sample_firing_strengths(model, sample, rule_names=None, seed=42):
    """
    Plottet für ein einzelnes Sample die Firing Strengths aller Regeln als Balkendiagramm.
    
    :param model: Das ANFIS-Modell.
    :param sample: Ein einzelnes Eingabebeispiel als Torch-Tensor der Form [input_dim].
    :param rule_names: Optionale Liste von Namen für die Regeln (z.B. ['Rule 0', 'Rule 1', ...])
    """
    model.eval()
    with torch.no_grad():
        # Führe einen Forward-Pass durch. Beachte: sample.unsqueeze(0) macht aus [input_dim] einen Batch von 1
        _, firing_strengths, _ = model(sample.unsqueeze(0))  # Shape: [1, num_rules]
    
    # Entferne den Batch-Dimension und wandle in NumPy um
    fs = firing_strengths.squeeze(0).cpu().numpy()
    
    num_rules = fs.shape[0]
    x = list(range(num_rules))
    
    plt.figure(figsize=(10, 5))
    plt.bar(x, fs, color='skyblue')
    plt.xlabel("Regel-Index")
    plt.ylabel("Firing Strength")
    if rule_names is None:
        plt.title("Firing Strengths für das ausgewählte Sample")
    else:
        plt.title("Firing Strengths für das ausgewählte Sample\n" + ", ".join(rule_names))
    plt.xticks(x, rule_names if rule_names is not None else x, rotation=45)
    plt.tight_layout()
    plt.show()

def plot_TopK(topk_p, firing):
    plt.figure(figsize=(10,6))
    plt.imshow((firing>0).astype(float), cmap='Greys', aspect='auto')
    plt.title(f"Aktiv‑Maske (K={topk_p})"); plt.xlabel("Regel"); plt.ylabel("Sample")
    plt.show()

def plot_sorted_Fs(preds, firing_strengths):
    sortedPreds, preds_ind  = torch.sort(preds)
    
    #class_boundaries = np.nonzero(np.diff(sortedPreds))[0] + 1
    
    diffs = sortedPreds[1:] != sortedPreds[:-1]   # Boolean‑Mask, wo Klassen wechseln
    boundaries = torch.nonzero(diffs).squeeze() + 1 
    
    
    fig, ax = plt.subplots(figsize=(12, 6))
        
    sorted_Fs = firing_strengths[preds_ind.cpu()]

    # 1) Transpose if needed
    sorted_Fs = sorted_Fs.T  # shape [num_rules, N]

    # 2) Plot
    im = ax.imshow(sorted_Fs, aspect='auto', cmap='viridis', vmin=0, vmax=0.01)

    # 3) Draw vertical class boundaries in data coords
    for boundary in boundaries[1:-1]:
        ax.axvline(boundary - 0.5, color='white', linewidth=0.5)
    
    plot_TopK(0.1, sorted_Fs)

    ax.set_xlabel("Testbeispiel Index (sortiert nach Klasse)")
    ax.set_ylabel("Regel-Index")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Firing Strength")

    plt.title("Heatmap der Regelfiring Strengths mit Klassengrenzen")
    plt.show()
            
    return

def get_important_rule_per_class_torch(firing_strengths, labels):


    unique_classes = torch.unique(labels)
    important_rules = {}
    
    # Für jede Klasse:
    for cls in unique_classes:
      
        indices = (labels == cls).nonzero(as_tuple=True)[0]
        
        # Berechne den Mittelwert der Firing Strengths für jede Regel für diese Instanzen:
        avg_strengths = torch.mean(firing_strengths[indices, :], dim=0)
        
        # Finde den Regelindex mit dem höchsten Mittelwert:
        important_rule_idx = torch.argmax(avg_strengths).item()
        
        # Speichere das Ergebnis im Dictionary
        important_rules[int(cls.item())] = important_rule_idx
        
    return important_rules
