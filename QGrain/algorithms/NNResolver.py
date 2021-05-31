import time

import numpy as np
import torch
from QGrain.algorithms import FittingState
from QGrain.algorithms import DistributionType
from QGrain.algorithms.kernels import get_distance_func_by_name, get_initial_guess, KERNEL_MAP, N_PARAMS_MAP
from QGrain.ssu import SSUResult, SSUTask
from QGrain.models.NNResolverSetting import NNResolverSetting
from torch.nn import Module, Parameter, ReLU, Softmax


class NNMixedDistribution(Module):
    def __init__(self, distribution_type: DistributionType, n_components: int, initial_guess=None):
        super().__init__()
        self.distribution_type = distribution_type
        self.n_components = n_components
        self.softmax = Softmax(dim=0)
        self.relu = ReLU()
        self.kernels = []
        n_params = N_PARAMS_MAP[distribution_type]
        kernel_class = KERNEL_MAP[distribution_type]
        if initial_guess is None:
            self.fractions = Parameter(torch.rand(n_components), requires_grad=True)
            for i in range(n_components):
                kernel = kernel_class()
                self.__setattr__(f"kernel_{i+1}", kernel)
                self.kernels.append(kernel)
        else:
            assert len(initial_guess) == (n_params+1) * n_components - 1
            expanded = list(initial_guess)
            expanded.append(1-sum(expanded[-n_components+1:]))
            self.fractions = Parameter(torch.Tensor(expanded[-n_components:]), requires_grad=True)
            for i in range(n_components):
                kernel = kernel_class(*expanded[i*n_params:(i+1)*n_params])
                self.__setattr__(f"kernel_{i+1}", kernel)
                self.kernels.append(kernel)

    @property
    def params(self):
        params = []
        for kernel in self.kernels:
            params.extend(kernel.params)
        with torch.no_grad():
            params.extend([value.item() for value in self.softmax(self.fractions)[:-1]])
            return params

    def forward(self, classes_φ):
        fractions = self.softmax(self.fractions).reshape(1, -1)
        distributions = []
        for i in range(self.n_components):
            distribution = self.kernels[i](classes_φ)
            distributions.append(distribution.reshape(1, -1))
        distributions = torch.cat(distributions, dim=0)
        mixed = fractions @ distributions
        return mixed, fractions, distributions

class NNResolver:
    def __init__(self):
        pass

    def try_fit(self, task: SSUTask):
        assert task.resolver == "neural"
        if task.resolver_setting is None:
            setting = NNResolverSetting()
        else:
            assert isinstance(task.resolver_setting, NNResolverSetting)
            setting = task.resolver_setting

        initial_guess = task.initial_guess
        if task.reference is not None:
            initial_guess = get_initial_guess(task.distribution_type, task.reference)

        start = time.time()
        X = torch.from_numpy(np.array(task.sample.classes_φ, dtype=np.float32))
        y = torch.from_numpy(np.array(task.sample.distribution, dtype=np.float32))
        X, y = X.to(setting.device), y.to(setting.device)
        model = NNMixedDistribution(task.distribution_type, task.n_components, initial_guess=initial_guess).to(setting.device)
        distance = get_distance_func_by_name(setting.distance)
        optimizer = torch.optim.Adam(model.parameters(), lr=setting.lr, eps=setting.eps)

        def train():
            # Compute prediction error
            y_hat, _, _ = model(X)
            loss = distance(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss.item()
        epoch = 0
        loss_series = []
        history = []
        while True:
            loss_value = train()
            loss_series.append(loss_value)
            history.append(model.params)
            epoch += 1
            if epoch < setting.min_niter:
                continue
            if loss_value < setting.tol:
                break
            delta_loss = np.std(loss_series[-50:])
            if delta_loss < setting.ftol:
                break
            if epoch > setting.max_niter:
                break
        time_spent = time.time() - start
        finished_params = model.params
        history.append(finished_params)
        result = SSUResult(task, finished_params, history=history, time_spent=time_spent)
        return FittingState.Succeeded, result
