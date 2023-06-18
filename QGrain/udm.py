__all__ = ["try_udm"]

import logging
import time
from typing import *

import numpy as np
import torch
from scipy.special import softmax
from scipy.spatial.distance import pdist

from .models import KernelType, Dataset, UDMResult, ArtificialDataset
from .kernels import ProportionModule, get_kernel

torch.set_default_dtype(torch.float32)


class UDMModule(torch.nn.Module):
    def __init__(self, n_samples: int, n_components: int, classes_phi: np.ndarray,
                 kernel_type: KernelType, x0: np.ndarray = None):
        super().__init__()
        self.n_samples = n_samples
        self.n_components = n_components
        self.n_classes = len(classes_phi)
        self._interval_phi = np.abs((classes_phi[0] - classes_phi[-1]) / (classes_phi.shape[0] - 1))
        self._classes_phi = torch.nn.Parameter(torch.from_numpy(classes_phi).repeat(n_samples, n_components, 1),
                                               requires_grad=False)
        self.kernel_type = kernel_type
        self.proportions = ProportionModule(n_samples, n_components)
        self.components = get_kernel(kernel_type, n_samples, self.n_components, self.n_classes, x0)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # n_samples x 1 x n_members
        proportions = self.proportions()
        # n_samples x n_members x n_classes
        components = self.components(self._classes_phi, self._interval_phi)
        return proportions, components

    @property
    def all_parameters(self) -> np.ndarray:
        with torch.no_grad():
            all_parameters = torch.cat([self.components._params, self.proportions._params], dim=1)
            all_parameters = all_parameters.detach().cpu().numpy()
        return all_parameters


def try_udm(dataset: Union[ArtificialDataset, Dataset], kernel_type: KernelType, n_components: int,
            x0: np.ndarray = None, device="cpu", pretrain_epochs=200, min_epochs=200, max_epochs=2000,
            precision: Union[int, float] = 6, learning_rate=5e-3, betas=(0.8, 0.5),
            consider_distance=False, constraint_level: Union[int, float] = 2.0, need_history=True,
            logger: logging.Logger = None, progress_callback: Callable[[float], None] = None) -> UDMResult:
    assert isinstance(dataset, (ArtificialDataset, Dataset))
    assert isinstance(kernel_type, KernelType)
    assert isinstance(n_components, int)
    if x0 is not None:
        assert isinstance(x0, np.ndarray)
        assert x0.ndim == 2
        assert x0.shape[1] == n_components
        x0 = x0.astype(np.float32)
    available_devices = ["cpu"]
    if torch.cuda.is_available():
        available_devices.append("cuda")
    available_devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
    if "cuda:0" in available_devices:
        available_devices.remove("cuda:0")
    assert device in available_devices
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
    assert isinstance(constraint_level, (int, float))
    assert pretrain_epochs >= 0
    assert min_epochs > 0
    assert max_epochs > 0
    assert 1.0 < precision < 100.0
    assert learning_rate > 0.0
    assert 0.0 < beta1 < 1.0
    assert 0.0 < beta2 < 1.0
    assert -10.0 < constraint_level < 10.0
    if logger is None:
        logger = logging.getLogger("QGrain")
    else:
        assert isinstance(logger, logging.Logger)

    start_text = f"""Performing the UDM algorithm on the dataset ({dataset.name}).
    Kernel type: {kernel_type.name}
    Number of end members: {n_components}
    x0: {x0 if x0 is None else x0.tolist()}
    Device: {device}
    Pretrain epochs: {pretrain_epochs}
    Minimum epochs: {min_epochs}
    Maximum epochs: {max_epochs}
    Precision decimals (10^-x): {precision}
    Learning rate: {learning_rate}
    Betas of weight decay: {betas}
    Consider distance: {consider_distance}
    Constraint level: {constraint_level}
    Need history: {need_history}"""
    logger.debug(start_text)

    observation = torch.from_numpy(dataset.distributions.astype(np.float32)).to(device)
    udm = UDMModule(len(dataset), n_components, dataset.classes_phi.astype(np.float32), kernel_type, x0).to(device)
    optimizer = torch.optim.Adam(udm.parameters(), lr=learning_rate, betas=betas)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    distribution_loss_series = []
    component_loss_series = []
    history: List[np.ndarray] = [udm.all_parameters]
    udm.components.requires_grad_(False)
    max_total_epochs = pretrain_epochs + max_epochs
    start_time = time.time()
    for pretrain_epoch in range(pretrain_epochs):
        proportions, components = udm()
        prediction = (proportions @ components).squeeze(1)
        distribution_loss = torch.log10(torch.mean(torch.square(observation - prediction)))
        distribution_loss_series.append(distribution_loss.item())
        component_loss_series.append(0.0)
        optimizer.zero_grad()
        distribution_loss.backward()
        torch.nn.utils.clip_grad_norm_(udm.parameters(), 1e-1)
        optimizer.step()
        if need_history:
            history.append(udm.all_parameters)
        if progress_callback is not None:
            progress_callback(pretrain_epoch / max_total_epochs)

    udm.components.requires_grad_(True)
    n_batches = 0
    start = 0
    while start < len(dataset) - 2:
        start += 150
        n_batches += 1
    space_locations = np.zeros((len(dataset), 3), dtype=np.float32)
    space_locations[:, -1] = np.linspace(1, len(dataset) / 5, len(dataset), dtype=np.float32)
    space_locations = torch.from_numpy(space_locations).to(device)

    for epoch in range(max_epochs):
        # train
        proportions, components = udm()
        prediction = (proportions @ components).squeeze(1)
        distribution_loss = torch.log10(torch.mean(torch.square(prediction - observation)))

        if consider_distance and len(dataset) <= 400:
            component_loss = 0
            component_ratios = torch.softmax(torch.sum(torch.std(components, dim=0), dim=1), dim=0)
            # component_ratios = torch.mean(proportions[:, 0], dim=0)
            for i in range(n_components):
                key = proportions[:, 0, i] > 1e-3
                space_distances = torch.nn.functional.pdist(space_locations[key])
                space_weights = torch.softmax(-space_distances, dim=0)
                component_distances = torch.nn.functional.pdist(components[:, i, :][key])
                component_loss += torch.sum(component_distances * space_weights) * component_ratios[i]
        elif consider_distance and len(dataset) > 400:
            component_loss = 0
            start = 0
            while start < len(dataset) - 2:
                component_ratios = torch.softmax(torch.sum(torch.std(components[start: start+200], dim=0), dim=1), dim=0)
                # component_ratios = torch.mean(proportions[start: start + 200, 0], dim=0)
                for i in range(n_components):
                    key = proportions[start: start+200, 0, i] > 1e-3
                    space_distances = torch.nn.functional.pdist(space_locations[start: start+200][key])
                    space_weights = torch.softmax(-space_distances, dim=0)
                    component_distances = torch.nn.functional.pdist(components[start: start+200, i, :][key])
                    component_loss += torch.sum(component_distances * space_weights) * component_ratios[i]
                start += 150
            component_loss /= n_batches
        else:
            # component_loss = torch.log10(torch.mean(torch.std(components, dim=0)))
            component_loss = torch.mean(torch.std(components, dim=0))

        loss = distribution_loss + (10 ** constraint_level) * component_loss
        if np.isnan(loss.item()):
            logger.warning("Loss is NaN, training has beem terminated.")
            break
        distribution_loss_series.append(distribution_loss.item())
        component_loss_series.append((10 ** constraint_level) * component_loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(udm.parameters(), 1e-1)
        optimizer.step()
        scheduler.step()
        if need_history:
            history.append(udm.all_parameters)
        if progress_callback is not None:
            progress_callback((pretrain_epochs + epoch) / max_total_epochs)
        if epoch > min_epochs:
            delta_loss = np.mean(distribution_loss_series[-100:-80]) - np.mean(distribution_loss_series[-20:])
            if delta_loss < 10 ** (-precision):
                break

    # algorithm finished, preparing the result
    if device[:4] == "cuda":
        torch.cuda.synchronize()
    time_spent = time.time() - start_time
    settings = dict(device=device, pretrain_epochs=pretrain_epochs, min_epochs=min_epochs,
                    max_epochs=max_epochs, precision=precision, learning_rate=learning_rate, betas=betas,
                    consider_distance=consider_distance, constraint_level=constraint_level, need_history=need_history)
    distribution_loss_series = np.array(distribution_loss_series)
    component_loss_series = np.array(component_loss_series)
    loss_series = {"total": distribution_loss_series + component_loss_series,
                   "distribution": distribution_loss_series,
                   "component": component_loss_series}
    if need_history:
        parameters = np.concatenate([np.expand_dims(p, axis=0) for p in history], axis=0)
    else:
        parameters = np.expand_dims(udm.all_parameters, axis=0)
    result = UDMResult(dataset, kernel_type, n_components, parameters, time_spent, x0, settings, loss_series)
    if progress_callback is not None:
        progress_callback(1.0)
    return result
