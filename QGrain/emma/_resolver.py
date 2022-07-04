import time
import typing

import numpy as np
import torch

from ..model import GrainSizeDataset
from ._distance import get_distance_func_by_name
from ._kernel import KernelType, ProportionModule, get_kernel
from ._result import EMMAResult
from ._setting import EMMAAlgorithmSetting


class EMMAModule(torch.nn.Module):
    def __init__(self,
                 n_samples: int,
                 n_members: int,
                 classes_φ: np.ndarray,
                 kernel_type: KernelType,
                 parameters: np.ndarray = None):
        super().__init__()
        self.n_samples = n_samples
        self.n_members = n_members
        self.n_classes = len(classes_φ)
        self.__interval = np.abs((classes_φ[0]-classes_φ[-1]) / (classes_φ.shape[0]-1))
        self.__classes = torch.nn.Parameter(torch.from_numpy(classes_φ).repeat(1, n_members, 1), requires_grad=False)
        self.kernel_type = kernel_type
        self.proportions = ProportionModule(n_samples, n_members)
        self.end_members = get_kernel(kernel_type, 1, self.n_members, self.n_classes, parameters)

    def forward(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # n_samples x n_members
        proportions = self.proportions().squeeze(1)
        # n_members x n_classes
        end_members = self.end_members(self.__classes, self.__interval).squeeze(0)
        # n_samples x n_classes
        # distributions = proportions @ end_members
        return proportions, end_members


class EMMAResolver:
    def __init__(self):
        pass

    def try_fit(self, dataset: GrainSizeDataset,
                kernel_type: KernelType,
                n_members: int,
                resolver_setting: EMMAAlgorithmSetting=None,
                parameters=None,
                update_end_members = True):
        if resolver_setting is None:
            s = EMMAAlgorithmSetting(max_epochs=2000, precision=6, learning_rate=5e-3)
        else:
            assert isinstance(resolver_setting, EMMAAlgorithmSetting)
            s = resolver_setting

        X = torch.from_numpy(dataset.distribution_matrix.astype(np.float32)).to(s.device)
        classes_φ = dataset.classes_φ.astype(np.float32)
        emma = EMMAModule(dataset.n_samples, n_members, classes_φ, kernel_type, parameters).to(s.device)

        distance_func = get_distance_func_by_name(s.distance)
        optimizer = torch.optim.Adam(emma.parameters(), lr=s.learning_rate, betas=s.betas)
        loss_series = []
        history = []
        start = time.time()
        emma.end_members.requires_grad_(False)
        for pretrain_epoch in range(s.pretrain_epochs):
            proportions, end_members = emma()
            X_hat = proportions @ end_members
            loss = distance_func(X_hat, X)
            loss_series.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append((proportions.detach().cpu().numpy(), end_members.detach().cpu().numpy()))
        emma.end_members.requires_grad_(update_end_members)
        epochs = 0
        while True:
            # train
            proportions, end_members = emma()
            X_hat = proportions @ end_members
            loss = distance_func(X_hat, X)
            loss_series.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append((proportions.detach().cpu().numpy(), end_members.detach().cpu().numpy()))
            epochs += 1
            if epochs < s.min_epochs:
                continue
            delta_loss = np.mean(loss_series[-100:-80])-np.mean(loss_series[-20:])
            if delta_loss < 10**(-s.precision):
                break
            if epochs > s.max_epochs:
                break
        if resolver_setting.device == "cuda":
            torch.cuda.synchronize()
        time_spent = time.time() - start
        result = EMMAResult(
            dataset,
            kernel_type,
            n_members,
            parameters,
            s,
            proportions.detach().cpu().numpy(),
            end_members.detach().cpu().numpy(),
            time_spent,
            history)
        return result
