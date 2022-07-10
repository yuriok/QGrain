import logging
import time
import typing

import numpy as np
import torch

from ..emma import KernelType, ProportionModule, get_kernel
from ..models import Dataset
from ._result import UDMResult
from ._setting import UDMAlgorithmSetting


class UDMModule(torch.nn.Module):
    def __init__(self,
                 n_samples: int,
                 n_components: int,
                 classes_φ: np.ndarray,
                 kernel_type: KernelType,
                 parameters: np.ndarray = None):
        super().__init__()
        self.n_samples = n_samples
        self.n_components = n_components
        self.n_classes = len(classes_φ)
        self.__interval = np.abs((classes_φ[0]-classes_φ[-1]) / (classes_φ.shape[0]-1))
        self.__classes = torch.nn.Parameter(torch.from_numpy(classes_φ).repeat(n_samples, n_components, 1), requires_grad=False)
        self.kernel_type = kernel_type
        self.proportions = ProportionModule(n_samples, n_components)
        self.components = get_kernel(kernel_type, n_samples, self.n_components, self.n_classes, parameters)

    def forward(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # n_samples x 1 x n_members
        proportions = self.proportions()
        # n_samples x n_members x n_classes
        components = self.components(self.__classes, self.__interval)
        return proportions, components

    @property
    def all_parameters(self) -> np.ndarray:
        return torch.cat([self.components._params, self.proportions._params], dim=1).detach().cpu().numpy()


class UDMResolver:
    logger = logging.getLogger("QGrain.UDMResolver")
    def __init__(self):
        pass

    def try_fit(
            self, dataset: Dataset,
            kernel_type: KernelType,
            n_components: int,
            resolver_setting: UDMAlgorithmSetting = None,
            parameters: np.ndarray = None,
            callback: typing.Callable = None) -> UDMResult:
        if resolver_setting is None:
            s = UDMAlgorithmSetting()
        else:
            assert isinstance(resolver_setting, UDMAlgorithmSetting)
            s = resolver_setting

        X = torch.from_numpy(dataset.distributions.astype(np.float32)).to(s.device)
        classes_φ = dataset.classes_phi.astype(np.float32)
        udm = UDMModule(len(dataset), n_components, classes_φ, kernel_type, parameters).to(s.device)
        optimizer = torch.optim.Adam(udm.parameters(), lr=s.learning_rate, betas=s.betas)

        start = time.time()
        distribution_loss_series = []
        component_loss_series = []
        history = []
        udm.components.requires_grad_(False)
        max_total_epochs = s.pretrain_epochs + s.max_epochs
        for pretrain_epoch in range(s.pretrain_epochs):
            proportions, components = udm()
            X_hat = (proportions @ components).squeeze(1)
            distribution_loss = torch.log10(torch.mean(torch.square(X - X_hat)))
            distribution_loss_series.append(distribution_loss.item())
            component_loss_series.append(0.0)
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(udm.parameters(), 1e-1)
            distribution_loss.backward()
            optimizer.step()
            history.append(udm.all_parameters)
            if callback is not None:
                callback(pretrain_epoch / max_total_epochs)

        udm.components.requires_grad_(True)
        for epoch in range(s.max_epochs):
            # train
            proportions, components = udm()
            X_hat = (proportions @ components).squeeze(1)
            distribution_loss = torch.log10(torch.mean(torch.square(X_hat - X)))
            component_loss = torch.mean(torch.std(components, dim=0))
            loss = distribution_loss + (10**s.constraint_level) * component_loss
            if np.isnan(loss.item()):
                self.logger.warning("Loss is NaN, training terminated.")
                break
            distribution_loss_series.append(distribution_loss.item())
            component_loss_series.append((10**s.constraint_level) * component_loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(udm.parameters(), 1e-1)
            optimizer.step()
            history.append(udm.all_parameters)
            if callback is not None:
                callback((s.pretrain_epochs+epoch) / max_total_epochs)
            if epoch > s.min_epochs:
                delta_loss = np.mean(distribution_loss_series[-100:-80])-np.mean(distribution_loss_series[-20:])
                if delta_loss < 10**(-s.precision):
                    break

        if s.device == "cuda":
            torch.cuda.synchronize()
        time_spent = time.time() - start
        result = UDMResult(
            dataset, kernel_type, n_components,
            parameters,
            s,
            np.array(distribution_loss_series),
            np.array(component_loss_series),
            time_spent,
            udm.all_parameters,
            history)
        if callback is not None:
            callback(1.0)
        return result
