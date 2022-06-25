import time
import typing

import numpy as np
import torch
from scipy.interpolate import interp1d

from ..model import GrainSizeDataset, GrainSizeSample
from ..ssu import (DISTRIBUTION_CLASS_MAP, DistributionType,
                   SSUAlgorithmSetting, try_sample)
from ..statistic import convert_φ_to_μm, logarithmic
from ._distance import get_distance_func_by_name
from ._kernel import KERNEL_CLASS_MAP, KernelType, Proportion
from ._result import EMMAResult
from ._setting import EMMAAlgorithmSetting


class EMMAModule(torch.nn.Module):
    def __init__(self,
                 n_samples: int,
                 n_members: int,
                 classes_φ: np.ndarray,
                 kernel_type: KernelType,
                 params: np.ndarray = None):
        super().__init__()
        self.n_samples = n_samples
        self.n_members = n_members
        self.n_classes = len(classes_φ)
        self.interval = np.abs((classes_φ[0]-classes_φ[-1]) / (classes_φ.shape[0]-1))
        self.classes = torch.nn.Parameter(torch.from_numpy(classes_φ).repeat(1, n_members, 1), requires_grad=False)
        self.kernel_type = kernel_type
        kernel_class = KERNEL_CLASS_MAP[kernel_type]
        self.proportions = Proportion(n_samples, n_members)
        self.end_members = kernel_class(1, self.n_members, self.n_classes, params)

    def forward(self):
        # n_samples x n_members
        proportions = self.proportions().squeeze(1)
        # n_members x n_classes
        end_members = self.end_members(self.classes, self.interval).squeeze(0)
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
            setting = EMMAAlgorithmSetting(max_epochs=2000, precision=6, learning_rate=5e-3)
        else:
            assert isinstance(resolver_setting, EMMAAlgorithmSetting)
            setting = resolver_setting

        X = torch.from_numpy(dataset.distribution_matrix.astype(np.float32)).to(setting.device)
        classes_φ = dataset.classes_φ.astype(np.float32)
        emma = EMMAModule(dataset.n_samples, n_members, classes_φ, kernel_type, parameters).to(setting.device)

        distance_func = get_distance_func_by_name(setting.distance)
        optimizer = torch.optim.Adam(emma.parameters(), lr=setting.learning_rate, betas=setting.betas)
        loss_series = []
        history = []
        start = time.time()

        emma.end_members.requires_grad = False
        emma.end_members.params.requires_grad = False
        for pretrain_epoch in range(setting.pretrain_epochs):
            proportions, end_members = emma()
            X_hat = proportions @ end_members
            loss = distance_func(X_hat, X)
            loss_series.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append((proportions.detach().cpu().numpy(), end_members.detach().cpu().numpy()))

        emma.end_members.requires_grad = update_end_members
        emma.end_members.params.requires_grad = update_end_members
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
            if epochs < setting.min_epochs:
                continue
            delta_loss = np.mean(loss_series[-100:-80])-np.mean(loss_series[-20:])
            if delta_loss < 10**(-setting.precision):
                break
            if epochs > setting.max_epochs:
                break
        torch.cuda.synchronize()
        time_spent = time.time() - start
        result = EMMAResult(
            dataset,
            kernel_type,
            n_members,
            parameters,
            setting,
            proportions.detach().cpu().numpy(),
            end_members.detach().cpu().numpy(),
            time_spent,
            history)
        return result
