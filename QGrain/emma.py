

import time
import typing

import numpy as np
import torch
from QGrain import DistributionType
from QGrain.kernels import KERNEL_MAP, NonparametricKernel, get_distance_func_by_name
from QGrain.models.EMMAResult import EMMAResult
from QGrain.models.GrainSizeDataset import GrainSizeDataset
from QGrain.models.NNResolverSetting import NNResolverSetting
from scipy.interpolate import interp1d
from torch.nn import Module, Parameter, Softmax
from torch.optim import Adam


class CustomizedEndMemberModule(Module):
    def __init__(self, end_members: typing.List[np.ndarray]):
        super().__init__()
        end_member_data = []
        for i, end_member in enumerate(end_members):
            end_member_data.append(torch.from_numpy(end_member.astype(np.float32)).reshape(1, -1))
        end_member_data = torch.cat(end_member_data, dim=0)
        self.__end_members = Parameter(end_member_data, requires_grad=False)

    def forward(self):
        return self.__end_members

    @property
    def end_members(self):
        with torch.no_grad():
            return self.__end_members

class EndMemberModule(Module):
    def __init__(self, distribution_type: DistributionType, n_members: int, classes_φ: torch.Tensor):
        super().__init__()
        n_classes = classes_φ.shape[0]
        if distribution_type == DistributionType.Nonparametric:
            kernel_class = lambda: NonparametricKernel(n_classes)
        else:
            kernel_class = KERNEL_MAP[distribution_type]
        self.kernels = []
        for i in range(n_members):
            kernel_i = kernel_class()
            self.__setattr__(f"kernel_{i+1}", kernel_i)
            self.kernels.append(kernel_i)
        self.__classes_φ = classes_φ

    def forward(self):
        # n_members x n_classes
        end_members = torch.cat([kernel(self.__classes_φ).reshape(1, -1) for kernel in self.kernels], dim=0)
        return end_members

    @property
    def end_members(self):
        with torch.no_grad():
            return self.forward()

class ProportionModule(Module):
    def __init__(self, n_samples: int, n_members: int):
        super().__init__()
        self.samples = []
        self.softmax = Softmax(dim=0)
        for i in range(n_samples):
            sample_i = Parameter(torch.rand(n_members), requires_grad=True)
            self.__setattr__(f"sample_{i+1}", sample_i)
            self.samples.append(sample_i)

    def forward(self):
        # n_samples x n_members
        proportions = torch.cat([self.softmax(sample).view(1, -1) for sample in self.samples], dim=0)
        return proportions

    @property
    def proportions(self):
        with torch.no_grad():
            return self.forward()

class EMMAModule(Module):
    def __init__(self, n_samples, classes_φ, n_members, distribution_type=DistributionType.Weibull):
        super().__init__()
        self.proportion_module = ProportionModule(n_samples, n_members)
        self.end_member_module = EndMemberModule(distribution_type, n_members, classes_φ)

    def forward(self):
        # n_samples x n_members
        proportions = self.proportion_module()
        # n_members x n_classes
        end_members = self.end_member_module()
        # n_samples x n_classes
        X_hat = proportions @ end_members
        return X_hat

    @property
    def proportions(self):
        return self.proportion_module.proportions

    @property
    def end_members(self):
        return self.end_member_module.end_members

    @property
    def X_hat(self):
        with torch.no_grad():
            return self.forward()

class CustomizedEMMAModule(Module):
    def __init__(self, n_samples: int, end_members: typing.List[np.ndarray]):
        super().__init__()
        n_members = len(end_members)
        self.proportion_module = ProportionModule(n_samples, n_members)
        self.end_member_module = CustomizedEndMemberModule(end_members)

    def forward(self):
        # n_samples x n_members
        proportions = self.proportion_module()
        # n_members x n_classes
        end_members = self.end_member_module()
        # n_samples x n_classes
        X_hat = proportions @ end_members
        return X_hat

    @property
    def proportions(self):
        return self.proportion_module.proportions

    @property
    def end_members(self):
        return self.end_member_module.end_members

    @property
    def X_hat(self):
        with torch.no_grad():
            return self.forward()

class EMMAResolver:
    def __init__(self):
        pass

    def try_fit(self, dataset: GrainSizeDataset,
                distribution_type: DistributionType,
                n_members: int,
                resolver_setting: NNResolverSetting):
        if resolver_setting is None:
            setting = NNResolverSetting(max_niter=1000, tol=1e-4, ftol=1e-6, lr=1e-2)
        else:
            assert isinstance(resolver_setting, NNResolverSetting)
            setting = resolver_setting
        start = time.time()
        X = torch.from_numpy(dataset.X.astype(np.float32)).to(setting.device)
        classes_φ = torch.from_numpy(dataset.classes_φ.astype(np.float32)).to(setting.device)
        emma = EMMAModule(dataset.n_samples, classes_φ, n_members, distribution_type=distribution_type).to(setting.device)
        distance_func = get_distance_func_by_name(setting.distance)
        optimizer = Adam(emma.parameters(), lr=setting.lr, eps=setting.eps)
        loss_series = []
        history = []
        iteration = 0
        while True:
            # train
            X_hat = emma()
            loss = distance_func(X_hat, X)
            loss_series.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append((emma.proportions.cpu().numpy(), emma.end_members.cpu().numpy()))
            iteration += 1
            if iteration < setting.min_niter:
                continue
            if loss_series[-1] < setting.tol:
                break
            std = np.std(loss_series[-50:])
            # print(std)
            if std < setting.ftol:
                break
            if iteration > setting.max_niter:
                break
        time_spent = time.time() - start
        result = EMMAResult(dataset,
                           distribution_type,
                           n_members,
                           setting,
                           emma.proportions.cpu().numpy(),
                           emma.end_members.cpu().numpy(),
                           time_spent,
                           history)
        return result

    def try_fit_with_fixed_ems(self, dataset: GrainSizeDataset,
                               em_classes_φ: np.ndarray,
                               em_distributions: typing.List[np.ndarray],
                               resolver_setting: NNResolverSetting):
        if resolver_setting is None:
            setting = NNResolverSetting(max_niter=1000, tol=1e-4, ftol=1e-6, lr=1e-2)
        else:
            assert isinstance(resolver_setting, NNResolverSetting)
            setting = resolver_setting
        start = time.time()
        X = torch.from_numpy(dataset.X.astype(np.float32)).to(setting.device)
        classes_φ = torch.from_numpy(dataset.classes_φ.astype(np.float32)).to(setting.device)
        n_members = len(em_distributions)
        fixed_end_members = []
        for em_distribution in em_distributions:
            trans = interp1d(em_classes_φ, em_distribution)
            fixed = trans(dataset.classes_φ)
            fixed_end_members.append(fixed)

        emma = CustomizedEMMAModule(dataset.n_samples, fixed_end_members).to(setting.device)
        distance_func = get_distance_func_by_name(setting.distance)
        optimizer = Adam(emma.parameters(), lr=setting.lr, eps=setting.eps)
        loss_series = []
        history = []
        iteration = 0
        while True:
            # train
            X_hat = emma()
            loss = distance_func(X_hat, X)
            loss_series.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append((emma.proportions.cpu().numpy(), emma.end_members.cpu().numpy()))
            iteration += 1
            if iteration < setting.min_niter:
                continue
            if loss_series[-1] < setting.tol:
                break
            std = np.std(loss_series[-50:])
            # print(std)
            if std < setting.ftol:
                break
            if iteration > setting.max_niter:
                break
        time_spent = time.time() - start
        result = EMMAResult(dataset,
                           DistributionType.Customized,
                           n_members,
                           setting,
                           emma.proportions.cpu().numpy(),
                           emma.end_members.cpu().numpy(),
                           time_spent,
                           history)
        return result

if __name__ == "__main__":
    import numpy as np
    from QGrain.algorithms import DistributionType
    from QGrain.charts.EMMAResultChart import EMMAResultChart
    from QGrain.entry import setup_app
    from QGrain.artificial import get_random_dataset, get_random_sample
    from QGrain.models.GrainSizeDataset import GrainSizeDataset
    from QGrain.models.NNResolverSetting import NNResolverSetting
    sample = get_random_sample(n_classes=51)
    dataset = get_random_dataset()
    samples = [dataset.get_sample(i) for i in range(dataset.n_samples)]
    sample_names = [sample.name for sample in samples]
    sample_distributions = [sample.distribution for sample in samples]
    test_dataset = GrainSizeDataset()
    test_dataset.add_batch(dataset.classes_μm, sample_names, sample_distributions)

    app, splash = setup_app()
    main = EMMAResultChart(toolbar=True)
    main.show()
    splash.finish(main)

    emma = EMMAResolver()
    resolver_setting = NNResolverSetting(device="cuda", max_niter=500, lr=1e-3)
    result = emma.try_fit_with_fixed_ems(test_dataset, sample.classes_φ, [comp.distribution for comp in sample.components], resolver_setting)

    main.show_result(result)
    app.exec_()
