import numpy as np
import pytest

from QGrain.distributions import DistributionType
from QGrain.kernels import KernelType
from QGrain.generate import random_dataset, SIMPLE_PRESET
from QGrain.models import EMMAResult
from QGrain.emma import try_emma, built_in_losses


class TestTryEMMA:
    dataset = random_dataset(**SIMPLE_PRESET, n_samples=200)
    x0 = np.array([[mean for (mean, std) in component] for component in SIMPLE_PRESET["target"]]).T
    x0 = x0[1:-1]

    @classmethod
    def log_message(cls, result: EMMAResult):
        print("\n", f"The fitting task [{result.n_samples}, {result.n_members}, {result.kernel_type.name}] of "
                    f"dataset [{result.dataset.name}] was finished using {result.n_iterations} iterations and "
                    f"{result.time_spent:.2f} s.\nFitting settings: {result.settings}.\n"
                    f"MSE: {result.loss('mse')}, LMSE: {result.loss('lmse')}, angular: {result.loss('angular')}.\n",
              sep="", end="\n")

    def test_one(self):
        result = try_emma(self.dataset, KernelType.Normal, self.dataset.n_components)
        self.log_message(result)
        assert isinstance(result, EMMAResult)

    def test_has_x0(self):
        result = try_emma(self.dataset, KernelType.Normal, self.dataset.n_components, x0=self.x0, pretrain_epochs=100)
        self.log_message(result)
        assert isinstance(result, EMMAResult)

    def test_cuda(self):
        result = try_emma(self.dataset, KernelType.Normal, self.dataset.n_components, x0=self.x0,
                          device="cuda", pretrain_epochs=100)
        self.log_message(result)
        assert isinstance(result, EMMAResult)

    def test_cuda0(self):
        result = try_emma(self.dataset, KernelType.Normal, self.dataset.n_components, x0=self.x0,
                          device="cuda:0", pretrain_epochs=100)
        self.log_message(result)
        assert isinstance(result, EMMAResult)

    def test_no_device(self):
        with pytest.raises(AssertionError):
            result = try_emma(self.dataset, KernelType.Normal, self.dataset.n_components, x0=self.x0,
                              device="cuda:1", pretrain_epochs=100)

    def test_progress_callback(self):
        def callback(p: float):
            assert 0.0 <= p <= 1.0
        result = try_emma(self.dataset, KernelType.Normal, self.dataset.n_components,
                          pretrain_epochs=100, progress_callback=callback)

    def test_result_properties(self):
        result = try_emma(self.dataset, KernelType.Normal, self.dataset.n_components)
        properties = ["dataset", "n_samples", "n_members", "n_classes", "n_iterations", "kernel_type",
                      "proportions", "end_members", "time_spent", "x0", "history", "settings"]
        for prop in properties:
            assert hasattr(result, prop)

    def test_history(self):
        result = try_emma(self.dataset, KernelType.Normal, self.dataset.n_components)
        proportions_bytes = result.proportions.tobytes()
        end_members_bytes = result.end_members.tobytes()
        for h in result.history:
            assert isinstance(h, EMMAResult)
            assert h is not result
            # will not modify the data of original object
            assert result.proportions.tobytes() == proportions_bytes
            assert result.end_members.tobytes() == end_members_bytes

    def test_loss(self):
        result = try_emma(self.dataset, KernelType.Normal, self.dataset.n_components)
        for loss_name in built_in_losses:
            loss_series = result.loss_series(loss_name)
            class_wise_losses = result.class_wise_losses(loss_name)
            sample_wise_losses = result.sample_wise_losses(loss_name)
