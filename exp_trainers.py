# Dispatcher import for experiment runners
from lb_scratch import ExperimentRunner
# trainers.py
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score

# Import model-specific classes and trainers for specialized handling
from anfis_hybrid import HybridANFIS
from PopFnn import POPFNN
from kmFmmc import FMNC
from trainAnfis import train_anfis_hybrid, train_anfis_noHyb
from trainPF import train_popfnn
from anfis_nonHyb import NoHybridANFIS
# You will need to make your GraphSSL class available for import
# For example, by placing it in a file like `graph_ssl.py`
from lb_scratch import GraphSSL 
from anfisHelper import initialize_mfs_with_kmeans

def _train_supervised_step(model, X_l, y_l, X_tr,device, lr=0.01, epochs=100, **kwargs):
    """
    A helper function to perform a standard supervised training run.
    This is a building block for more complex SSL trainers.
    It dispatches to model-specific trainers where necessary.
    """
    print(f"      Running initial supervised training for {epochs} epochs...")

    if isinstance(model, HybridANFIS):
        print("      Using specialized HybridANFIS trainer...")
        model.train_anfis_hybrid(model, X_l, y_l, X_tr, epochs=epochs) 
       
    elif isinstance(model, POPFNN):
        print("      Using specialized POPFNN trainer...")
        model.train_popfnn(X_l,y_l, X_tr, epochs=100)
        
        
    elif isinstance(model, NoHybridANFIS):
        print("      Using specialized NoHybridANFIS trainer...")
        # Assuming NoHybridANFIS has a specific training method
        #train_anfis_noHyb(model, X_l, y_l, X_tr, num_epochs=epochs, lr=lr)
        initialize_mfs_with_kmeans(model, X_tr)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits, fs, mask = model(X_l.to(device))
            loss = F.cross_entropy(logits, y_l.to(device))
            #lb_loss  = model.load_balance_loss(fs, mask)
            #loss     = loss + lb_loss
            loss.backward()
            optimizer.step()
    elif isinstance(model, FMNC):
        model.seed_boxes_kmeans(X_l, y_l, k=3)
        model.fit(X_l, y_l, epochs=1, shuffle=True)
        
     

    else:
        # Generic trainer for end-to-end differentiable models like NoHybridANFIS.
        print("      Using generic supervised trainer...")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X_l.to(device))
            logits = output[0] if isinstance(output, tuple) else output
            loss = F.cross_entropy(logits, y_l.to(device))
            loss.backward()
            optimizer.step()

    print("      Supervised training complete.")
    return model

def train_rulespace_ssl(
    model_instance,
    X_l, y_l, X_tr, y_tr, y_semi_sup, X_te,
    device,
    lr=0.01, epochs=500,
    ssl_method='grf', k=15, sigma=0.2, **kwargs
):
    # 1. Initial supervised training on labeled data
    model_instance = _train_supervised_step(model_instance, X_l, y_l, X_tr, device, lr=lr, epochs=epochs, **kwargs)

    # 2. Feature Extraction from the Rule Space
    print("      Extracting rule space features for GraphSSL...")
    model_instance.eval()
    with torch.no_grad():
        # Assumes model's forward pass returns: (logits, firing_strengths, other_params)
        _, rule_activations_train, _ = model_instance(X_tr.to(device))
        _, rule_activations_test, _ = model_instance(X_te.to(device))
    print(f"      Shape of rule space features: {rule_activations_train.shape}")

    # 3. Graph-based SSL fitting and prediction
    graph_ssl_model = GraphSSL(method=ssl_method, device=device, k=k, sigma=sigma)
    
    print(f"      Fitting GraphSSL model ({ssl_method.upper()})...")
    graph_ssl_model.fit(rule_activations_train, y_semi_sup)
    
    # 4. Evaluate pseudo-label accuracy on the training set (for diagnostics)
    unlabeled_mask = (y_semi_sup == -1)
    pseudo_label_acc = accuracy_score(y_tr.numpy()[unlabeled_mask], graph_ssl_model.transduction_[unlabeled_mask])
    print(f"      Diagnostic Pseudo-Label Accuracy: {pseudo_label_acc * 100:.2f}%")
    
    # 5. Predict on the test set's rule space and return the predictions
    y_pred_np = graph_ssl_model.predict(rule_activations_test)
    return y_pred_np



# --- Dispatcher for ExperimentRunner static methods ---
def run_experiment(model_instance, method, *args, **kwargs):
    """
    Dispatch to ExperimentRunner based on `method`.
    method should be one of:
      - 'supervised'
      - 'rule_space'
      - 'mf_space'
      - 'raw_space'
      - 'fmv_clp'
      - 'mv_grf'
    Additional positional and keyword args are passed to the runner.
    """
    runner = ExperimentRunner()
    method_map = {
        'supervised': 'run_supervised_baseline',
        'rule_space': 'run_rule_space_ssl',
        'mf_space':    'run_mf_space_ssl',
        'raw_space':   'run_raw_space_ssl',
        'fmv_clp':     'run_fmv_clp',
        'mv_grf':      'run_mv_grf',
    }
    if method not in method_map:
        raise ValueError(f"Unknown method: {method}. Valid methods: {list(method_map.keys())}")
    func_name = method_map[method]
    func = getattr(runner, func_name)
    # Call the runner method: first argument is model_instance when applicable
    return func(model_instance, *args, **kwargs) if method == 'supervised' or method.endswith('space_ssl') else func(*args, **kwargs)