__all__ = ["built_in_losses", "try_emma"]

import logging
import time
from typing import *

import numpy as np
import torch
from numpy import ndarray

from .models import KernelType, ArtificialDataset, Dataset, EMMAResult
from .kernels import loss_torch, ProportionModule, get_kernel

built_in_losses = (
    "1-norm", "2-norm", "3-norm", "4-norm",
    "mae", "mse", "rmse", "rmlse", "lmse", "angular", "cosine")

torch.set_default_dtype(torch.float32)


class EMMAModule(torch.nn.Module):
    def __init__(self, n_samples: int, n_members: int, classes_phi: np.ndarray,
                 kernel_type: KernelType, x0: np.ndarray = None):
        super().__init__()
        self.n_samples = n_samples
        self.n_members = n_members
        self.n_classes = len(classes_phi)
        self._interval_phi = np.abs((classes_phi[0] - classes_phi[-1]) / (classes_phi.shape[0] - 1))
        self._classes_phi = torch.nn.Parameter(
            torch.from_numpy(classes_phi).repeat(1, n_members, 1), requires_grad=False)
        self.kernel_type = kernel_type
        self.proportions = ProportionModule(n_samples, n_members)
        self.end_members = get_kernel(kernel_type, 1, self.n_members, self.n_classes, x0)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # n_samples x n_members
        proportions = self.proportions().squeeze(1)
        # n_members x n_classes
        end_members = self.end_members(self._classes_phi, self._interval_phi).squeeze(0)
        # n_samples x n_classes
        # distributions = proportions @ end_members
        return proportions, end_members


def try_emma(dataset: Union[ArtificialDataset, Dataset], kernel_type: KernelType, n_members: int, x0: ndarray = None,
             device="cpu", loss="lmse", pretrain_epochs=0, min_epochs=100, max_epochs=10000,
             precision: Union[int, float] = 6, learning_rate=5e-3, betas=(0.8, 0.5), update_end_members=True,
             need_history=True, logger: logging.Logger = None, progress_callback: Callable[[float], None] = None):
    assert isinstance(dataset, (ArtificialDataset, Dataset))
    assert isinstance(kernel_type, KernelType)
    assert isinstance(n_members, int)
    if x0 is not None:
        assert isinstance(x0, ndarray)
        assert x0.ndim == 2
        assert x0.shape[1] == n_members
        x0 = x0.astype(np.float32)
    available_devices = ["cpu"]
    if torch.cuda.is_available():
        available_devices.append("cuda")
    available_devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
    if "cuda:0" in available_devices:
        available_devices.remove("cuda:0")
    assert device in available_devices
    assert loss in built_in_losses
    assert isinstance(pretrain_epochs, int)
    assert isinstance(min_epochs, int)
    assert isinstance(max_epochs, int)
    assert isinstance(precision, (int, float))
    assert isinstance(learning_rate, float)
    assert isinstance(betas, tuple)
    assert len(betas) == 2
    beta1, beta2 = betas
    assert isinstance(beta1, float)
    assert isinstance(beta2, float)
    assert pretrain_epochs >= 0
    assert min_epochs > 0
    assert max_epochs > 0
    assert 1.0 < precision < 100.0
    assert learning_rate > 0.0
    assert 0.0 < beta1 < 1.0
    assert 0.0 < beta2 < 1.0
    if logger is None:
        logger = logging.getLogger("QGrain")
    else:
        assert isinstance(logger, logging.Logger)

    start_text = f"""Performing the EMMA algorithm on the dataset ({dataset.name}).
    Kernel type: {kernel_type.name}
    Number of end members: {n_members}
    x0: {x0 if x0 is None else x0.tolist()}
    Device: {device}
    Loss: {loss}
    Pretrain epochs: {pretrain_epochs}
    Minimum epochs: {min_epochs}
    Maximum epochs: {max_epochs}
    Precision decimals (10^-x): {precision}
    Learning rate: {learning_rate}
    Betas of weight decay: {betas}
    Update end members: {update_end_members}
    Need history: {need_history}"""
    logger.debug(start_text)

    observation = torch.from_numpy(dataset.distributions.astype(np.float32)).to(device)
    emma = EMMAModule(len(dataset), n_members, dataset.classes_phi.astype(np.float32), kernel_type, x0).to(device)
    loss_func = loss_torch(loss)
    optimizer = torch.optim.Adam(emma.parameters(), lr=learning_rate, betas=betas)
    total_loss_series = []
    history: List[Tuple[ndarray, ndarray]] = []
    emma.end_members.requires_grad_(False)
    max_total_epochs = pretrain_epochs + max_epochs
    start = time.time()
    for pretrain_epoch in range(pretrain_epochs):
        proportions, end_members = emma()
        prediction = proportions @ end_members
        loss_i = loss_func(prediction, observation)
        total_loss_series.append(loss_i.item())
        optimizer.zero_grad()
        loss_i.backward()
        torch.nn.utils.clip_grad_norm_(emma.parameters(), 1e-1)
        optimizer.step()
        if need_history:
            history.append((proportions.detach().cpu().numpy(), end_members.detach().cpu().numpy()))
        if progress_callback is not None:
            progress_callback(pretrain_epoch / max_total_epochs)

    emma.end_members.requires_grad_(update_end_members)
    for epoch in range(max_epochs):
        # train
        proportions, end_members = emma()
        prediction = proportions @ end_members
        loss_i = loss_func(prediction, observation)
        if np.isnan(loss_i.item()):
            logger.warning("Loss is NaN, training has beem terminated.")
            break
        total_loss_series.append(loss_i.item())
        optimizer.zero_grad()
        loss_i.backward()
        torch.nn.utils.clip_grad_norm_(emma.parameters(), 1e-1)
        optimizer.step()
        if need_history:
            history.append((proportions.detach().cpu().numpy(), end_members.detach().cpu().numpy()))
        if progress_callback is not None:
            progress_callback((pretrain_epochs+epoch) / max_total_epochs)
        if epoch > min_epochs:
            delta_loss = np.mean(total_loss_series[-100:-80])-np.mean(total_loss_series[-20:])
            if delta_loss < 10**(-precision):
                break

    # algorithm finished, preparing the result
    if device[:4] == "cuda":
        torch.cuda.synchronize()
    time_spent = time.time() - start
    settings = dict(device=device, loss=loss, pretrain_epochs=pretrain_epochs, min_epochs=min_epochs,
                    max_epochs=max_epochs, precision=precision, learning_rate=learning_rate,
                    betas=betas, update_end_members=update_end_members, need_history=need_history)
    total_loss_series = np.array(total_loss_series)
    loss_series = {loss: total_loss_series}
    if need_history:
        proportions = np.concatenate([np.expand_dims(proportions, axis=0) for proportions, _ in history], axis=0)
        end_members = np.concatenate([np.expand_dims(end_members, axis=0) for _, end_members in history], axis=0)
    else:
        with torch.no_grad():
            proportions, end_members = emma()
            proportions = np.expand_dims(proportions.cpu().numpy(), axis=0)
            end_members = np.expand_dims(end_members.cpu().numpy(), axis=0)
    result = EMMAResult(dataset, kernel_type, n_members, proportions, end_members, time_spent, x0, settings, loss_series)
    if progress_callback is not None:
        progress_callback(1.0)
    return result
