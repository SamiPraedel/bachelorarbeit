import numpy as np
import torch
import argparse

from torchmetrics.functional import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

from anfis_hybrid import HybridANFIS
from mlp_iris import FullyConnected, fit_mlp
from anfis_nonHyb import NoHybridANFIS
from PopFnn import POPFNN

from trainPF import train_popfnn, calculate_popfnn_silhouette
from trainAnfis import train_anfis_noHyb, train_anfis_hybrid
from data_utils import load_iris_data, load_heart_data, load_wine_data, load_abalon_data, load_Kp_chess_data, load_K_chess_data_splitted, load_K_chess_data_OneHot, load_Poker_data, load_Kp_chess_data_ord
from create_plots import plot_TopK, plot_sorted_Fs, plot_umap_fixed_rule, plot_sample_firing_strengths
from anfisHelper import weighted_sampler, rule_stats, set_rule_subset
# for training on cuda
import matplotlib.pyplot as plt
import os
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


def run_experiment(
    type,
    dataset,
    num_epochs,
    num_mfs,
    max_rules,
    seed,
    lr=1e-3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    DATASETS = {
        "iris":    (load_iris_data,3),
        "heart":   (load_heart_data,5),
        "wine":    (load_wine_data,11),
        "abalone": (load_abalon_data,3),
        "ChessK":  (load_K_chess_data_splitted,18),
        "ChessKp": (load_Kp_chess_data_ord,2),
        "Poker":   (load_Poker_data,10),
    }
    try:
        loader, num_classes = DATASETS[dataset]
    except KeyError:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    X_train, y_train, X_test, y_test, *rest = loader()

    if type == "anfis":
        X_train_np = X_train.numpy()
        X_train  = X_train.to(device)
        y_train  = y_train.to(device)
        X_test   = X_test.to(device)
        y_test   = y_test.to(device)
        
        model = HybridANFIS(
            input_dim=X_train.shape[1], 
            num_classes=num_classes,
            num_mfs=num_mfs,
            max_rules=max_rules,
            seed=seed
        )
        model.to(device)
        
        train_anfis_hybrid(model, X_train, y_train, num_epochs=num_epochs, lr=lr)
    elif type == "noHyb":

        X_train_t = X_train.clone().detach().float()
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t   = torch.tensor(X_test,   dtype=torch.float32)

        model = NoHybridANFIS(
            input_dim=X_train.shape[1], 
            num_classes=num_classes,
            num_mfs=num_mfs,
            max_rules=max_rules,
            seed=seed,
            zeroG=False
        ).to(device)

        train_loader = weighted_sampler(X_train_t, y_train_t, y_train)
        train_anfis_noHyb(model, X_train_t, y_train_t, num_epochs=num_epochs, lr=lr, dataloader=train_loader)
            
        #plot_umap_fixed_rule(model, X_val_t, rule_index=10, cmap='viridis')
        #plot_sample_firing_strengths(model, X_val_t[1])
    elif type == "Pop":
        
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_test  = X_test.to(device)
        y_test  = y_test.to(device)

        model = POPFNN(X_train.shape[1], num_classes, num_mfs).to(device)
        model.pop_init(X_train, y_train)
        model = train_popfnn(model, X_train, y_train, num_epochs, lr)

    elif type == "RandomF":
        model = RandomForestClassifier(n_estimators=25)
        model.fit(X_train,y_train)
    elif type == "neuralNet":
       model = FullyConnected(input_dim=X_train.shape[1], hidden1=30, hidden2=50, num_classes=num_classes)
       fit_mlp(model, X_train, y_train)

    else:
        raise ValueError("Unknown model type: " + type)

    # 3) Testphase
    
    with torch.no_grad():
        
        if type == "anfis" or type == "noHyb":
            # Ensure model and data are on the correct device for evaluation
            model.eval()
            X_test_eval = X_test.to(device)
            y_test_eval = y_test.to(device) # y_test needs to be on device for matthews_corrcoef

            logits, firing, _ = model(X_test_eval)          # firing: [B, R] (top-K schon gespart)

            preds = logits.argmax(1)                     # [B]

            acc = (preds.cpu() == y_test.cpu()).float().mean().item() * 100
            mcc = matthews_corrcoef(preds, y_test_eval, "multiclass",
                                    num_classes=num_classes).item()
            print(f"Accuracy: {acc:.2f}%   |   MCC: {mcc:.4f}")

            # Silhouette 
            y_np  = y_test.cpu().numpy()
            if 2 <= np.unique(y_np).size < len(y_np) - 1:
                sil = silhouette_score(firing.cpu().numpy(), y_np)
                print(f"Silhouette (true labels vs firing): {sil:.4f}")
                
            with torch.no_grad():
                μ = model._fuzzify(X_test.to(device))          # [B, d, M]
                mf_vec = μ.view(len(X_test), -1).cpu().numpy() # [B, d*M]

                sil_mf = silhouette_score(mf_vec, y_np)
                print(f"Silhouette (MF-Raum vs true labels): {sil_mf:.4f}")
            
            
            
            cov, H, mi = rule_stats(model, X_test_eval, y_test_eval)
            
            plt.hist(cov.cpu(), bins=50, log=True); plt.title("Coverage distribution")
            plt.scatter(cov.cpu(), H.cpu(), s=5); plt.xlabel("cov"); plt.ylabel("H") # This is line 145
            
            # Save the combined plot (histogram and scatter)
            save_dir = "visualizations"
            os.makedirs(save_dir, exist_ok=True) # Ensure the directory exists
            plot_filename = os.path.join(save_dir, "coverage_entropy_stats.png")
            plt.savefig(plot_filename)
            print(f"Saved rule statistics plot to {plot_filename}")
            plt.close()
            
            
            confident = (H < 0.603) | (cov/cov.sum() > 0.002)
            print(f"Verlässliche Regeln: {confident.sum()}/{len(cov)}")
            

            
            # 2D-T-SNE auf MF-Vektor
            
            Z = TSNE(n_components=2).fit_transform(mf_vec)      # mf_vec = μ Reshape
            plt.scatter(Z[:,0], Z[:,1], c=y_test, s=4)
            plot_filename = os.path.join(save_dir, "tsneplot.png")
            plt.savefig(plot_filename)
            plt.close()
            
            cov_thr = 0.001
            cov_rel = cov / cov.sum()              # cov aus rule_stats
            print(cov.sum())
            keep    = cov_rel > cov_thr
            num_kept_rules = keep.sum().item()
            print(f"{num_kept_rules} Regeln behalten")
            model.rules = model.rules.cpu() # Ensure rules are on CPU for further processing
            keep = keep.cpu()  # Convert to CPU tensor for indexing
            pruned_rules = model.rules[keep]  # pruned_rules is a CPU tensor
            
            Z_rules_tsne = TSNE(n_components=2, random_state=seed).fit_transform(model.rules.numpy()) # model.rules is already CPU
            # Use H (entropy of rules) for coloring. H has shape [num_rules]
            H_cpu = H.cpu() # Move H to CPU once for all plotting
            plt.scatter(Z_rules_tsne[:,0], Z_rules_tsne[:,1], c=H_cpu, s=4, cmap='viridis')
            plt.title(f"t-SNE of all {model.rules.shape[0]} rules (colored by entropy H)")
            plot_filename = os.path.join(save_dir, "tsne_rules.png")
            plt.savefig(plot_filename)
            plt.close()
            
            # t-SNE for pruned rules, colored by their entropy H (filtered by keep)
            plt.figure() # Ensure a new figure
            Z_pruned_rules_tsne = TSNE(n_components=2, random_state=seed).fit_transform(pruned_rules.numpy()) # pruned_rules is CPU tensor
            plt.scatter(Z_pruned_rules_tsne[:,0], Z_pruned_rules_tsne[:,1], c=H_cpu[keep], s=4, cmap='viridis') # H_cpu and keep are CPU tensors
            plt.title(f"t-SNE of {pruned_rules.shape[0]} pruned rules (colored by entropy H)")

            


            # Aus 'rule_stats' hast du schon entropie: Tensor der Länge R
            entropie = H

            # 1) In NumPy konvertieren
            ent_np = entropie.cpu().numpy()

            # 2) Einfaches Histogramm der Entropiewerte
            plt.hist(ent_np, bins=50, range=(0,1))
            plt.xlabel("Regel-Entropie H_r (normalisiert)")
            plt.ylabel("Anzahl Regeln")
            plt.title("Verteilung der H_r über alle R Regeln")
            plot_filename = os.path.join(save_dir, "Regelentropie.png")
            plt.savefig(plot_filename)
            plt.close()

            # 3) Vielleicht noch Quantile anschauen:
            # for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            #     print(f"{int(q*100)}%-Quantil von H: {np.quantile(ent_np, q):.3f}")

            # Use y_np (defined earlier as y_test.cpu().numpy())
            # Calculate silhouette score using firing strengths of PRUNED rules for X_test
            # 'firing' is from model(X_test_eval), shape [B, R_original], on device
            # 'keep' is a boolean tensor mask for rules, shape [R_original], on CPU
            
            keep = (cov/cov.sum() > 0.002)
            
            firing_np = firing.cpu().numpy()
            keep_np_bool = keep.cpu().numpy().astype(bool)

     
            firing_of_pruned_rules_np = firing_np[:, keep_np_bool]
            
            n_labels_for_pruned_firing = np.unique(y_np).size
            n_samples_for_pruned_firing = firing_of_pruned_rules_np.shape[0]
            n_features_for_pruned_firing = firing_of_pruned_rules_np.shape[1]

            if n_features_for_pruned_firing > 0 and \
                2 <= n_labels_for_pruned_firing < n_samples_for_pruned_firing:
                sil_score_pruned_firing = silhouette_score(firing_of_pruned_rules_np, y_np)
                print(f"Silhouette (firing of pruned rules vs true labels): {sil_score_pruned_firing:.4f}")
            else:
                print(f"Silhouette (firing of pruned rules vs true labels): N/A (conditions not met: "
                        f"features={n_features_for_pruned_firing}, labels={n_labels_for_pruned_firing}, samples={n_samples_for_pruned_firing})")



            # --- Visualisierung
            plot_sorted_Fs(preds.cpu(), firing.cpu().numpy())
        elif type == "Pop":
            model.eval()                             
            with torch.no_grad():
                preds = model(X_test).argmax(dim=1)

            acc = (preds == y_test).float().mean().item()
            print(f"Test-Accuracy: {acc*100:.2f}%")
            calculate_popfnn_silhouette()
            

            
        elif type == "neuralNet":
            outputs = model(X_test.to(device)) # Ensure X_test is on device
            preds = torch.argmax(outputs, dim=1)
            accuracy = metrics.accuracy_score(y_test, preds)

        else:
            y_pred = model.predict(X_test)
            print('Accuracy: ', metrics.accuracy_score(y_test,y_pred))
            # mcc = matthews_corrcoef(torch.tensor(y_pred), y_test,"multiclass",num_classes=5)
            # print(mcc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["anfis","noHyb","Pop", "RandomF","neuralNet"], default="noHyb",)
    parser.add_argument("--dataset", choices=["iris","heart","ChessK", "ChessKp","abalone", "Poker"], default="ChessK")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--num_mfs", type=int, default=4)
    parser.add_argument("--max_rules", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()

    run_experiment(
        type=args.type,
        dataset=args.dataset,
        num_epochs=args.epochs,
        lr=args.lr,
        num_mfs=args.num_mfs,
        max_rules=args.max_rules,
        seed=args.seed,
    )

    
