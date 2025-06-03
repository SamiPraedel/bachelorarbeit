import numpy as np
import torch
import argparse
 
from torchmetrics.functional import matthews_corrcoef as tm_matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import silhouette_score, balanced_accuracy_score, f1_score, matthews_corrcoef as sk_matthews_corrcoef
from sklearn.manifold import TSNE
import seaborn as sns

from anfis_hybrid import HybridANFIS
from mlp_iris import FullyConnected, fit_mlp
from anfis_nonHyb import NoHybridANFIS
from PopFnn import POPFNN

from trainPF import train_popfnn, calculate_popfnn_silhouette
from trainAnfis import train_anfis_noHyb, train_anfis_hybrid
from data_utils import load_iris_data, load_heart_data, load_wine_data, load_abalon_data, load_Kp_chess_data, load_K_chess_data_splitted, load_K_chess_data_OneHot, load_poker_data, load_Kp_chess_data_ord, load_shuttle_data, load_gamma_data
from create_plots import plot_TopK, plot_sorted_Fs, plot_umap_fixed_rule, plot_sample_firing_strengths
from anfisHelper import weighted_sampler, rule_stats, set_rule_subset, rule_stats_wta, per_class_rule_scores
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
        "Poker":   (load_poker_data,10),
        "shuttle": (load_shuttle_data, 7),
        "gamma":   (load_gamma_data, 2)
        
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
        
        train_anfis_hybrid(model, X_train, y_train, num_epochs=num_epochs, lr=lr, X_val=X_test, y_val=y_test)
    elif type == "noHyb":

        X_train_t = X_train.clone().detach().float()
        y_train_t = y_train.clone().detach().long()
        X_val_t   = X_test.clone().detach().float()

        model = NoHybridANFIS(
            input_dim=X_train.shape[1], 
            num_classes=num_classes,
            num_mfs=num_mfs,
            max_rules=max_rules,
            seed=seed,
            zeroG=False
        ).to(device)

        # Pass y_train as numpy array for weighted_sampler and specify batch_size
        #train_loader = weighted_sampler(X_train_t, y_train_t, y_train.cpu().numpy(), batch_size=128) # Assuming 128 is desired batch size
        train_anfis_noHyb(model, X_train_t, y_train_t, num_epochs=num_epochs, lr=lr, X_val=X_val_t, y_val=y_test) # Use the passed dataloader
            
        #plot_umap_fixed_rule(model, X_val_t, rule_index=10, cmap='viridis')
        #plot_sample_firing_strengths(model, X_val_t[1])
    elif type == "Pop":
        
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_test  = X_test.to(device)
        y_test  = y_test.to(device)

        model = POPFNN(X_train.shape[1], num_classes, num_mfs).to(device)
        model.pop_init(X_train, y_train)
        # Assuming train_popfnn is updated similarly to accept X_val, y_val
        model = train_popfnn(model, X_train, y_train, num_epochs, lr, X_val=X_test, y_val=y_test)

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
            model.eval()
            X_test_d = X_test.to(device)
            y_test_d = y_test.to(device)

            # ---------- Vorwärtslauf ------------------------------------
            logits, firing, _ = model(X_test_d)
            preds  = logits.argmax(1)

            acc = (preds.cpu() == y_test.cpu()).float().mean().item() * 100
            mcc = tm_matthews_corrcoef(preds, y_test_d, task="multiclass",
                                       num_classes=num_classes).item()
            print(f"Accuracy: {acc:.2f}%   |   MCC: {mcc:.4f}")

            # ---------- Silhouette in Regel- & MF-Raum ------------------
            y_np = y_test.cpu().numpy()
            if 2 <= np.unique(y_np).size < len(y_np)-1:
                sil_rule = silhouette_score(firing.cpu().numpy(), y_np)
                print(f"Silhouette (firing vs true):          {sil_rule:+.4f}")

            μ       = model._fuzzify(X_test_d)            # [B,d,M]
            mf_vec  = μ.view(len(X_test), -1).cpu().numpy()
            sil_mf  = silhouette_score(mf_vec, y_np)
            print(f"Silhouette (MF-Vector vs true):         {sil_mf:+.4f}")

            # ---------- Regel-Statistik & ODER-Pruning ------------------
            cov, H, _ = rule_stats(model, X_test_d, y_test_d)   # [R]
            cov_rel   = cov / cov.sum()

            tau_cov, tau_H = 0.01, 0.5
            keep = (cov_rel > tau_cov) | (H < tau_H)
            print(f"Verlässliche Regeln: {keep.sum()}/{len(cov)}")

            # ---------- Cover/H-Scatter & Histogram speichern -----------
            save_dir = "visualizations"; os.makedirs(save_dir, exist_ok=True)

            fig, ax = plt.subplots(1,2,figsize=(10,4))
            ax[0].hist(cov.cpu(), bins=50, log=True)
            ax[0].set_title("Coverage distribution"); ax[0].set_xlabel("cov")

            sc = ax[1].scatter(cov.cpu(), H.cpu(), c=keep.cpu(), cmap="coolwarm", s=4)
            ax[1].set_xlabel("cov"); ax[1].set_ylabel("H"); ax[1].set_title("cov vs H")
            plt.tight_layout(); plt.savefig(f"{save_dir}/cov_H_stats.png"); plt.close()

            # ---------- t-SNE: MF-Vector & Rules ------------------------
            Z = TSNE(2).fit_transform(mf_vec)
            plt.scatter(Z[:,0], Z[:,1], c=y_np, s=5, cmap='tab10')
            plt.title("t-SNE of MF-Vectors (true labels)")
            plt.savefig(f"{save_dir}/tsne_mf_vec.png"); plt.close()

            # t-SNE aller Regeln (farbig nach keep)
            Zr = TSNE(2, random_state=0).fit_transform(model.rules.cpu().numpy())
            plt.scatter(Zr[:,0], Zr[:,1], c=keep.cpu(), s=5, cmap='coolwarm')
            plt.title("t-SNE of rules (blue=pruned, red=kept)")
            plt.savefig(f"{save_dir}/tsne_rules_keep.png"); plt.close()

            # ---------- Silhouette NUR noch kept-Rules ------------------
            firing_kept = firing[:, keep].cpu().numpy()
            if firing_kept.shape[1] > 0:
                sil_keep = silhouette_score(firing_kept, y_np)
                print(f"Silhouette (kept rules):               {sil_keep:+.4f}")
            else:
                print("Silhouette (kept rules):               N/A – no rules kept")
                
                
            cov, pur, main, score, p_rc = per_class_rule_scores(model, X_test_d, y_test_d)

            k = 10                     # z.B. 10 beste Regeln je Klasse
            top_rules = {c: [] for c in range(model.num_classes)}

            for c in range(model.num_classes):
                idx_c = torch.where(main == c)[0]             # Regeln, deren Hauptklasse c ist
                if len(idx_c) > 0: 
                    sel    = score[idx_c].topk(min(k, len(idx_c))).indices
                    top_rules[c] = idx_c[sel]
                    # print(f"Klasse {c}: Top-{len(sel)} Regeln -> Score {score[idx_c[sel]]}")
                else:
                    top_rules[c] = torch.tensor([], dtype=torch.long, device=main.device) # Store empty tensor
            
            

            list_of_rule_tensors = [rules_for_class for rules_for_class in top_rules.values() if isinstance(rules_for_class, torch.Tensor) and rules_for_class.numel() > 0]
            
            if list_of_rule_tensors:
                sel_idx = torch.cat(list_of_rule_tensors)
            else:
                sel_idx = torch.tensor([], dtype=torch.long, device=main.device) # Ensure sel_idx is an empty tensor if no rules

            for c, idx in top_rules.items():
                if idx.numel() > 0: # Only plot if there are rules for this class
                    plt.scatter(cov[idx].cpu(), pur[idx].cpu(), label=f"class {c}")
            plt.xlabel("cov"); plt.ylabel("purity"); plt.legend(); 
            plt.title("purity vs coverage der Top-Regeln")
            plt.savefig(f"{save_dir}/purity vs coverage der Top-Regeln.png"); plt.close()

            if sel_idx.numel() > 0:
                heat = p_rc[sel_idx].cpu()            # Reinheitsmatrix [R_sel, C]
                sns.heatmap(heat, annot=True, cmap="rocket_r")
                plt.title("Klassenanteile der Top-Regeln"); plt.ylabel("Regeln"); plt.xlabel("Klassen")
                plt.savefig(f"{save_dir}/Klassenanteile der Top-Regeln.png"); plt.close()
                            
                # Ensure perplexity is less than n_samples for TSNE
                n_samples_for_tsne = model.rules[sel_idx.cpu()].shape[0]
                perplexity_val = min(30, n_samples_for_tsne - 1) if n_samples_for_tsne > 1 else 5
                
                if n_samples_for_tsne > 1 : # TSNE requires n_samples > n_components
                    X_tsne = model.rules[sel_idx.cpu()].numpy()
                    labels_tsne = main[sel_idx.cpu()].cpu().numpy()

                    # Calculate Silhouette Score for the selected rules in their original space
                    if 2 <= np.unique(labels_tsne).size < n_samples_for_tsne -1:
                        try:
                            sil_top_rules = silhouette_score(X_tsne, labels_tsne)
                            print(f"Silhouette (Top Rules vs Main Class):   {sil_top_rules:+.4f}")
                        except ValueError as e:
                            print(f"Silhouette (Top Rules vs Main Class):   N/A (Error: {e})")
                    else:
                        print(f"Silhouette (Top Rules vs Main Class):   N/A (Unique labels: {np.unique(labels_tsne).size}, Samples: {n_samples_for_tsne})")

                    Z = TSNE(n_components=2, perplexity=perplexity_val, random_state=0).fit_transform(X_tsne)
                    plt.scatter(Z[:,0], Z[:,1], c=labels_tsne, cmap='tab10', s=20)
                    plt.title("t-SNE der Top-Regeln")
                    plt.savefig(f"{save_dir}/t-SNE der Top-Regeln.png"); plt.close()
                else:
                    print("Skipping t-SNE of Top-Regeln: Not enough samples after selection.")
            else:
                print("Skipping Heatmap and t-SNE of Top-Regeln: No rules selected.")
            
            # Corrected metric calculations using defined variables
            y_test_cpu_np = y_test.cpu().numpy()
            preds_cpu_np = preds.cpu().numpy()
            print("Balanced Acc :", balanced_accuracy_score(y_test_cpu_np, preds_cpu_np) * 100)
            print("Macro-F1     :", f1_score(y_test_cpu_np, preds_cpu_np, average='macro') * 100)
            print("MCC (sklearn):", sk_matthews_corrcoef(y_test_cpu_np, preds_cpu_np))



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
    parser.add_argument("--dataset", choices=["iris","heart","ChessK", "ChessKp","abalone", "Poker", "shuttle", "gamma"], default="ChessK")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--num_mfs", type=int, default=3)
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

    
