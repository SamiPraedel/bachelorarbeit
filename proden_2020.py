import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#-------------------
# Modell
#-------------------


class HybridANFIS(nn.Module):
    def __init__(self, input_dim, num_classes, num_mfs, max_rules):
        super(HybridANFIS, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_mfs = num_mfs
        self.num_rules = num_mfs ** input_dim  # Total number of rules

        # Membership function parameters (Gaussian)
        self.centers = nn.Parameter(torch.rand(input_dim, num_mfs))  # Centers
        #print(self.centers[0])
        self.widths = nn.Parameter(torch.rand(input_dim, num_mfs))  # Widths

        # Im __init__ deines Modells
        full_rules = torch.cartesian_prod(*[torch.arange(self.num_mfs) for _ in range(self.input_dim)])
        # full_rules.shape = [8192, 13]

        # max_rules
        idx = torch.randperm(full_rules.size(0))[:max_rules]
        self.rules = full_rules[idx]  # => shape [max_rules, input_dim]
        self.num_rules = self.rules.size(0)  # =max_rules

        # Consequent parameters (initialized randomly)
        self.consequents = nn.Parameter(torch.rand(self.num_rules, input_dim + 1, num_classes))
        self.consequents.requires_grad = False


    def gaussian_mf(self, x, center, width):
        gaus = torch.exp(-((x - center) ** 2) / (2 * width ** 2))
        if torch.isnan(center).any():
            print("center in gaus")
        
        if torch.isnan(width).any():
            print("width in gauss")
        
        if torch.isinf(gaus).any():
            print("x in gauss  ", torch.exp(-((x - center) ** 2) / (2 * width ** 2)))

        return gaus

    def forward(self, x):
        batch_size = x.size(0)

        # Step 1: Compute membership values
        mfs = []
        for i in range(self.input_dim):
            if torch.isnan(x).any():
                print("vor unsqueeze")
            x_i = x[:, i].unsqueeze(1)  # Shape: [batch_size, 1]
            if torch.isnan(x_i).any():
                print("nach unsqueeze")
            #print(self.centers)
            center_i = self.centers[i]  # Shape: [num_mfs]


            width_i = self.widths[i]    # Shape: [num_mfs]

            if torch.isnan(center_i).any():
                print("center in gaus")
        
            if torch.isnan(width_i).any():
                print("width in gauss")

  

            mf_i = self.gaussian_mf(x_i, center_i, width_i)  # Shape: [batch_size, num_mfs]
            if torch.isnan(self.gaussian_mf(x_i, center_i, width_i)).any():
                print("mfi")
                if torch.isnan(x_i).any():
                    print(" hihi ")



            mfs.append(mf_i)

        mfs = torch.stack(mfs, dim=1)  # Shape: [batch_size, input_dim, num_mfs]
        if torch.isnan(mfs).any():
            print("width in gauss")
        



 
        # Step 2: Compute rule activations
        #full_rules = torch.cartesian_prod(*[torch.arange(self.num_mfs) for _ in range(self.input_dim)])



        # rules.shape => [num_rules, input_dim]

        rules_idx = self.rules.unsqueeze(0).expand(batch_size, -1, -1).permute(0, 2, 1)
        

        # rules_idx.shape => [batch_size, input_dim, num_rules]

        # Now gather along dim=2 in 'mfs'
        # mfs.shape => [batch_size, input_dim, num_mfs]
        rule_mfs = torch.gather(mfs, dim=2, index=rules_idx)
        
        # rule_mfs.shape => [batch_size, input_dim, num_rules]

        # Multiply membership values across input_dim
        fiering_strengths = torch.prod(rule_mfs, dim=1)
 

        # shape => [batch_size, num_rules]


        # Step 3: Normalize rule activations
        eps = 1e-9
        normalized_firing_strengths = fiering_strengths / (fiering_strengths.sum(dim=1, keepdim=True) + eps)
   

        # Step 4: Compute rule outputs
        x_ext = torch.cat([x, torch.ones(batch_size, 1)], dim=1)  # Add bias term, shape: [batch_size, input_dim + 1]
        rule_outputs = torch.einsum('br,brc->bc', normalized_firing_strengths, 
                                    torch.einsum('bi,rjc->brc', x_ext, self.consequents))
        
        
        
        # Shape: [batch_size, num_classes]
        return rule_outputs, normalized_firing_strengths, x_ext

    def update_consequents(self, normalized_firing_strengths, x_ext, Y):
        """
        Update consequent parameters using Least Squares Estimation (LSE).
        :param normalized_firing_strengths: Normalized rule activations, shape: [batch_size, num_rules]
        :param x_ext: Extended inputs (with bias), shape: [batch_size, input_dim + 1]
        :param y: Target outputs (one-hot encoded), shape: [batch_size, num_classes]
        """
        batch_size = normalized_firing_strengths.size(0)

        # Prepare the design matrix (Phi)
        Phi = normalized_firing_strengths.unsqueeze(2) * x_ext.unsqueeze(1)  # Shape: [batch_size, num_rules, input_dim + 1]
        Phi = Phi.view(batch_size, self.num_rules * (self.input_dim + 1))  # Flattened design matrix

        # Solve the least-squares problem: Phi.T @ Phi @ consequents = Phi.T @ y
        #Phi_T_Phi = torch.matmul(Phi.T, Phi)
  
 

        B = torch.linalg.lstsq(Phi, Y).solution

        #PhiT = Phi.transpose(0,1)
        #lambda_ = 1e-3
        #A = PhiT @ Phi
        #A += lambda_ * torch.eye(A.shape[0], device=A.device)
        #B = torch.linalg.solve(A, PhiT @ Y)
        

        if torch.isnan(B).any():
            print("NaN in LSE-solution B!")

        #B = torch.linalg.solve(Phi_T_Phi, Phi_T_Y)  # => [P, C]

        # Anschließende Reshape
        self.consequents.data = B.view(self.num_rules, self.input_dim + 1, self.num_classes)








""" Module for PRODEN. """

import random
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
#from tqdm import tqdm
import torch.nn.functional as F
from pll_classifier_base import PllBaseClassifier
from result import SplitResult
#from anfis_NN import HybridANFIS


class Proden(PllBaseClassifier):
    """
    PRODEN by Lv et al.,
    "Progressive Identification of True Labels for Partial-Label Learning"
    """

    def __init__(
        self, rng: np.random.Generator, debug: bool = False, **kwargs,
    ) -> None:
        self.rng = rng
        self.device = torch.device("cpu")
        self.debug = debug
        #self.loop_wrapper = tqdm if debug else (lambda x: x)
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(int(self.rng.integers(1000)))
        torch.manual_seed(int(self.rng.integers(1000)))
        self.model: Optional[nn.Module] = None

    def fit(
        self, inputs: np.ndarray, partial_targets: np.ndarray,
    ) -> SplitResult:
        """ Fits the model to the given inputs.

        Args:
            inputs (np.ndarray): The inputs.
            partial_targets (np.ndarray): The partial targets.

        Returns:
            SplitResult: The disambiguated targets.
        """

        num_epoch = 200
        # if len(inputs.shape) == 2:
        #     # Tabular data (Batch, Features)
        #     self.model = HybridANFIS(inputs.shape[1], partial_targets.shape[1])
        #     self.device = torch.device("cpu")  # Always fastest on CPU
        #     batch_size = 1024
        # elif len(inputs.shape) == 4:
        #     # Image data (Batch, Channels, Height, Width)
        #     self.model = LeNet(partial_targets.shape[1])
        #     batch_size = 64
        #     if torch.cuda.is_available():
        #         cuda_idx = random.randrange(torch.cuda.device_count())
        #         self.device = torch.device(f"cuda:{cuda_idx}")
        #     elif torch.backends.mps.is_available():
        #         self.device = torch.device("mps")
        #     else:
        #         self.device = torch.device("cpu")
        # else:
        #     raise ValueError(f"Malformed data of shape {inputs.shape}.")
        input_dim = inputs.shape[1]
        num_classes = partial_targets.shape[1]
        num_mfs = 2
        max_rules = 40

        self.model = HybridANFIS(input_dim, num_classes, num_mfs, max_rules)

        self.model.to(self.device)

        # Data preparation
        x_train = torch.tensor(inputs, dtype=torch.float32)
        y_train = torch.tensor(partial_targets, dtype=torch.float32)
        train_indices = torch.arange(x_train.shape[0], dtype=torch.int32)
        loss_weights = torch.tensor(partial_targets, dtype=torch.float32)
        loss_weights /= loss_weights.sum(dim=1, keepdim=True)
        data_loader = DataLoader(
            TensorDataset(train_indices, x_train, y_train, loss_weights),
            batch_size=batch_size, shuffle=True, generator=self.torch_rng,
        )

        # Optimizer
        self.model.train()
        optimizer = torch.optim.Adam([self.model.centers, self.model.widths])

        # Training loop
        for _ in self.loop_wrapper(range(num_epoch)):
            for idx, inputs_i, partial_targets_i, w_ij in data_loader:
                # Move to device
                inputs_i = inputs_i.to(self.device)
                partial_targets_i = partial_targets_i.to(self.device)
                w_ij = w_ij.to(self.device)

                # Forward-backward pass
                probs = self.model(inputs_i)
                loss = torch.mean(torch.sum(
                    w_ij * -torch.log(probs + 1e-10), dim=1,
                ))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 3) LSE-Update der Konklusionsgewichte
                with torch.no_grad():
                # Y als One-Hot
                    Y_onehot = F.one_hot(Y, num_classes=self.model.num_classes).float()
                    self.model.update_consequents(
                    self.model.firing_strengths.detach(), 
                    self.model.x_ext.detach(), 
                    Y_onehot
                )

                # Update weights
                with torch.no_grad():
                    updated_w = partial_targets_i * probs
                    updated_w /= torch.sum(updated_w, dim=1, keepdim=True)
                    loss_weights[idx] = updated_w.to("cpu")

        # Return results
        return SplitResult.from_scores(self.rng, loss_weights.numpy())

    def predict(self, inputs: np.ndarray) -> SplitResult:
        """ Predict the labels.

        Args:
            inputs (np.ndarray): The inputs.

        Returns:
            SplitResult: The predictions.
        """

        if self.model is None:
            raise ValueError()

        if len(inputs.shape) == 2:
            batch_size = 1024
        elif len(inputs.shape) == 4:
            batch_size = 64
        else:
            raise ValueError(f"Malformed data of shape {inputs.shape}.")
        inference_loader = DataLoader(
            TensorDataset(torch.tensor(
                inputs, dtype=torch.float32)),
            batch_size=batch_size, shuffle=False,
        )

        # Switch to eval mode
        self.model.eval()
        all_results = []
        with torch.no_grad():
            for x_batch in inference_loader:
                x_batch = x_batch[0].to(self.device)
                all_results.append(
                    self.model(x_batch).to("cpu").numpy())
            train_probs = np.vstack(all_results)
        return SplitResult.from_scores(self.rng, train_probs)


