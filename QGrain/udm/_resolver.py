import time
import typing

import numpy as np
import torch
from scipy.interpolate import interp1d

from ..emma import KERNEL_CLASS_MAP, KernelType, Proportion
from ..model import GrainSizeDataset, GrainSizeSample
from ..ssu import (DISTRIBUTION_CLASS_MAP, DistributionType,
                   SSUAlgorithmSetting, try_sample)
from ..statistic import convert_φ_to_μm, logarithmic
from ._result import UDMResult
from ._setting import UDMAlgorithmSetting


class UDMModule(torch.nn.Module):
    def __init__(self,
                 n_samples: int,
                 n_components: int,
                 classes_φ: np.ndarray,
                 kernel_type: KernelType,
                 params: np.ndarray = None):
        super().__init__()
        self.n_samples = n_samples
        self.n_components = n_components
        self.n_classes = len(classes_φ)
        self.interval = np.abs((classes_φ[0]-classes_φ[-1]) / (classes_φ.shape[0]-1))
        self.classes = torch.nn.Parameter(torch.from_numpy(classes_φ).repeat(n_samples, n_components, 1), requires_grad=False)
        self.kernel_type = kernel_type
        kernel_class = KERNEL_CLASS_MAP[kernel_type]
        self.proportions = Proportion(n_samples, n_components)
        self.components = kernel_class(n_samples, self.n_components, self.n_classes, params)

    def forward(self):
        # n_samples x 1 x n_members
        proportions = self.proportions()
        # n_samples x n_members x n_classes
        components = self.components(self.classes, self.interval)
        return proportions, components


class UDMResolver:
    def __init__(self):
        pass

    def try_fit(self, dataset: GrainSizeDataset,
                kernel_type: KernelType,
                n_components: int,
                initial_params: np.ndarray = None,
                resolver_setting: UDMAlgorithmSetting = None,
                save_history = False):
        if resolver_setting is None:
            s = UDMAlgorithmSetting()
        else:
            assert isinstance(resolver_setting, UDMAlgorithmSetting)
            s = resolver_setting

        X = torch.from_numpy(dataset.distribution_matrix.astype(np.float32)).to(s.device)
        classes_φ = dataset.classes_φ.astype(np.float32)
        udm = UDMModule(dataset.n_samples, n_components, classes_φ, kernel_type, initial_params).to(s.device)
        optimizer = torch.optim.Adam(udm.parameters(), lr=s.learning_rate, betas=s.betas)

        start = time.time()
        distribution_loss_series = []
        component_loss_series = []
        history_params = []
        udm.components.requires_grad = False
        udm.components.params.requires_grad = False
        for pretrain_epoch in range(s.pretrain_epochs):
            proportions, components = udm()
            X_hat = (proportions @ components).squeeze(1)
            distribution_loss = torch.log10(torch.mean(torch.square(X - X_hat)))
            distribution_loss_series.append(distribution_loss.item())
            component_loss_series.append(0.0)
            print(f"Pretrain Stage -- Epoch {pretrain_epoch}, GSD Loss {distribution_loss.item():0.4f}")
            optimizer.zero_grad()
            distribution_loss.backward()
            optimizer.step()

            if save_history:
                params = torch.cat([udm.components.params, udm.proportions.params], dim=1).detach().cpu().numpy()
                history_params.append(params)

        udm.components.requires_grad = True
        udm.components.params.requires_grad = True
        for epoch in range(s.max_epochs):
            # train
            proportions, components = udm()
            X_hat = (proportions @ components).squeeze(1)
            distribution_loss = torch.log10(torch.mean(torch.square(X_hat - X)))
            component_loss = torch.mean(torch.std(components, dim=0))
            loss = distribution_loss + (10**s.constraint_level) * component_loss
            distribution_loss_series.append(distribution_loss.item())
            component_loss_series.append(component_loss.item())
            print(f"Training -- Epoch {epoch}, GSD Loss {distribution_loss.item():0.4f}, Component Loss {component_loss.item(): 0.4f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if save_history:
                params = torch.cat([udm.components.params, udm.proportions.params], dim=1).detach().cpu().numpy()
                history_params.append(params)

            if np.isnan(loss.item()):
                break

            if epoch > s.min_epochs:
                delta_loss = np.mean(distribution_loss_series[-100:-80])-np.mean(distribution_loss_series[-20:])
                if delta_loss < 10**(-s.precision):
                    break

        time_spent = time.time() - start
        params = torch.cat([udm.components.params, udm.proportions.params], dim=1).detach().cpu().numpy()
        result = UDMResult(
            dataset, kernel_type, n_components,
            params,
            distribution_loss_series,
            component_loss_series,
            initial_params=initial_params,
            resolver_setting=resolver_setting,
            time_spent=time_spent)
        return result
