########################################################################################################################
# Apache License 2.0
########################################################################################################################
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2025 Nina de Lacy

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import torch
from .binary_metrics import binary_metrics, flagged_at_top_k_ppv, threshold_at_specificity_k, nb_weight_from_pt
from copy import deepcopy
from time import time
from torch import nn
from typing import Callable, Optional, Union

########################################################################################################################
# Define a callable class for loss functions in PyTorch
########################################################################################################################
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

########################################################################################################################
# Define a class for early stopping (used in the validation process)
########################################################################################################################
class EarlyStopping:
    """
    Early stopping technique used during model training to speed up runtime and prevent over-fitting.

    A. Runtime parameters
    ---------------------
    A1. patience: A non-negative integer.
        Number of times that a worse result (depending on A2) can be tolerated.
        Default setting: patience=10
    A2. min_delta: A non-negative float.
        The threshold where loss_old + min_delta < loss_new is considered as worse.
        Default setting: min_delta=1e-8

    B. Attributes
    -------------
    B1. min_loss: A float, recording the minimum loss encountered in the training process.
    B2. counter: A non-negative integer, recording the number of times a worse result is observed. The counter restarts
                 when a better result (i.e., smaller loss) is observed.
    (A1-A2 are initialized as instance attributes.)

    C. Methods
    ---------
    C1. refresh()
        Reset the attributes B1 and B2 to their original values.
    C2. early_stop(loss)
    :param loss: An integer or float. The loss obtained in a given training epoch.
    :return: A boolean indicating whether the training process should be stopped.
    """
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 1e-8):

        # Type and value check
        assert isinstance(patience, int), \
            f"patience must be an integer. Now its type is {type(patience)}."
        assert patience >= 0, \
            f"patience must be a non-negative integer. Now it is {patience}."
        self.patience = patience
        try:
            min_delta = float(min_delta)
        except TypeError:
            raise TypeError(f"min_delta must be a float. Now its type is {type(min_delta)}.")
        assert min_delta >= 0, \
            f"min_delta must be a non-negative float. Now it is {min_delta}."
        self.min_delta = min_delta
        self.min_loss = float('inf')
        self.counter = 0
        self.best_state_dict = None

    def reset(self):
        self.min_loss = float('inf')
        self.counter = 0
        self.best_state_dict = None

    def early_stop(self, loss: float, model=None):
        try:
            loss = float(loss)
        except TypeError:
            raise TypeError(f"loss must be a float. Now its type is {type(loss)}.")
        if loss < self.min_loss:
            self.min_loss, self.counter = loss, 0
            if model is not None:
                self.best_state_dict = deepcopy(model.state_dict())
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

########################################################################################################################
# Define a superclass (inherited from torch.nn.Module) for various model classes (e.g., ANN, LSTM, Transformer, TCN)
########################################################################################################################


class DL_Class(nn.Module):
    """
    A superclass for various model classes that will be used in this module.

    A. Runtime parameters
    ---------------------
    (None)

    B. Attributes
    -------------
    B1. dummy_param: A torch.nn.Parameter object.
        An identifier of the physical location of the model.

    C. Methods
    ----------
    C1. set_device(device_str)
        :param device_str: A string or torch.device object. The physical location of the model to be set.
    C2. get_device()
        :return: A string. The current physical location of the model.
    C3. get_n_params()
        :return: An integer. The number of parameters of the model.
    """
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

    def set_device(self, device_str: Union[str, torch.device]):
        assert type(device_str) in [str, torch.device], \
            f'device_str must be a torch.device object or a string. Now its type is {type(device_str)}.'
        self.to(device_str)

    def get_device(self):
        return self.dummy_param.device

    def get_n_params(self):
        return sum(p.numel() for p in self.parameters())        # For AIC & BIC calculation

########################################################################################################################
# Define a function to create a feedforward (aka ANN) model
########################################################################################################################
def create_ann(n_feat: int,
               n_units: int,
               n_layers: int = 3):
    modules = [nn.Linear(n_feat, n_units), nn.ReLU()]
    for _ in range(n_layers-1):
        modules += [nn.Linear(n_units, n_units), nn.ReLU()]
    modules += [nn.Linear(n_units, 1)]
    M = nn.Sequential(*modules)
    for name, param in M.named_parameters():
        if 'weight' in name:
            torch.nn.init.xavier_uniform_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)
    return M

########################################################################################################################
# Define a function to train a transformer model (with internal validation)
########################################################################################################################
def make_weight_tensor(y: torch.Tensor, cost_mul: float):
    return torch.where(y==0, torch.as_tensor(cost_mul, dtype=y.dtype, device=y.device),
                             torch.as_tensor(1, dtype=y.dtype, device=y.device))


def train_ann(M, X_train, y_train, X_val, y_val, cost_mul=None):
    
    # Prepare model
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    M = M.to(device)

    # Prepare data
    X_train = torch.tensor(np.array(X_train, dtype=np.float32)).to(device)
    X_val = torch.tensor(np.array(X_val, dtype=np.float32)).to(device)
    y_train = torch.Tensor(y_train).to(device)
    y_val = torch.Tensor(y_val).to(device)

    # Data batching
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    val_data = torch.utils.data.TensorDataset(X_val, y_val)
    batch_size = 512
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=False)

    # Set up the loss function
    if cost_mul is None:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')

    # Set up the optimizer and the early stopper
    opt = torch.optim.AdamW(M.parameters(), lr=1e-4, weight_decay=1e-2)
    early = EarlyStopping(patience=10, min_delta=1e-4)

    # Training loop
    max_epochs = 100
    for epoch in range(max_epochs):
        M.train()
        train_running_loss = 0
        train_running_weight = 0
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = M(xb).squeeze(-1)
            loss_raw = criterion(logits, yb)
            if cost_mul is None:
                loss = loss_raw
                train_running_loss += loss.item() * xb.size(0)
            else:
                w = make_weight_tensor(yb, cost_mul)
                loss = (loss_raw * w).sum() / (w.sum() + 1e-12)
                train_running_loss += (loss_raw * w).sum().item()
                train_running_weight += w.sum().item()
            loss.backward()
            opt.step()

        if cost_mul is None:
            train_loss = train_running_loss / len(train_loader.dataset)
        else:
            train_loss = train_running_loss / (train_running_weight + 1e-12)

        # Internal validation
        M.eval()
        val_running_loss = 0
        val_running_weight = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = M(xb).squeeze(-1)
                val_loss_raw = criterion(logits, yb)
                if cost_mul is None:
                    val_loss = val_loss_raw
                    val_running_loss += val_loss.item() * xb.size(0)
                else:
                    w = make_weight_tensor(yb, cost_mul)
                    val_loss = (val_loss_raw * w).sum() / (w.sum() + 1e-12)
                    val_running_loss += (val_loss_raw * w).sum().item()
                    val_running_weight += w.sum().item()

        if cost_mul is None:
            val_loss = val_running_loss / len(val_loader.dataset)
        else:
            val_loss = val_running_loss / (val_running_weight + 1e-12)

        if early.early_stop(val_loss, model=M):
            if early.best_state_dict is not None:
                M.load_state_dict(early.best_state_dict)
            print(f"Early stopped at epoch {epoch+1} with best val. loss = {early.min_loss:.4f}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{max_epochs} completed with val. loss = {val_loss:.4f} (best = {early.min_loss:.4f})")

    if early.best_state_dict is not None:
        M.load_state_dict(early.best_state_dict)

    return M

########################################################################################################################
# Define a function to evaluate a transformer model
########################################################################################################################
def eval_ann(M, X_test, y_test, prefix=''):
    t0 = time()
    M.eval()
    with torch.no_grad():
        device = next(M.parameters()).device
        X_test = torch.tensor(np.array(X_test, dtype=np.float32)).to(device)
        y_test = torch.Tensor(y_test).to(device)
        y_prob = torch.sigmoid(M(X_test).view(-1))
        t1 = time()        
    threshold_tuple_list: list[tuple[float, str]] = [(0.5, ''),
                                                    (flagged_at_top_k_ppv(y_prob, k=1), '@Precision1%'),
                                                    (flagged_at_top_k_ppv(y_prob, k=2), '@Precision2%'),
                                                    (flagged_at_top_k_ppv(y_prob, k=5), '@Precision5%'),
                                                    (threshold_at_specificity_k(y_test, y_prob, 99), '@99Spec'),
                                                    (threshold_at_specificity_k(y_test, y_prob, 95), '@95Spec'),
                                                    (threshold_at_specificity_k(y_test, y_prob, 90), '@90Spec')]
    nbw = nb_weight_from_pt(1/11)
    output_dict: dict[str, float] = {}
    for threshold_tuple in threshold_tuple_list:
        suffix: str = threshold_tuple[1]
        n_params: int = sum(p.numel() for p in M.parameters() if p.requires_grad)

        cur_result: dict[str, float] = binary_metrics(y_true=y_test,
                                                      y_prob=y_prob,
                                                      y_pred_override=None if (suffix == '' or 'Spec' in suffix) else threshold_tuple[0],
                                                      threshold=0.5 if (suffix == '' or 'Precision' in suffix) else threshold_tuple[0],
                                                      nb_weight=nbw,
                                                      n_params=n_params,
                                                      decimals=5,
                                                      verbose=False,
                                                      prefix=prefix)
        output_dict |= {f'{k}{suffix}': v for k, v in cur_result.items()}
    return output_dict, round(t1-t0, 3)
