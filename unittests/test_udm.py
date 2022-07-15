import numpy as np
import pytest

from QGrain.emma import built_in_losses
from QGrain.generate import random_dataset, SIMPLE_PRESET
from QGrain.kernels import KernelType
from QGrain.models import UDMResult
from QGrain.udm import try_udm


class TestTryUDM:
    dataset = random_dataset(**SIMPLE_PRESET, n_samples=200)
    x0 = np.array([[mean for (mean, std) in component] for component in SIMPLE_PRESET["target"]]).T
    x0 = x0[1:-1]

    @classmethod
    def log_message(cls, result: UDMResult):
        print("\n", f"The fitting task [{result.n_samples}, {result.n_components}, {result.kernel_type.name}] of "
                    f"dataset [{result.dataset.name}] was finished using {result.n_iterations} iterations and "
                    f"{result.time_spent:.2f} s.\nFitting settings: {result.settings}.\n"
                    f"MSE: {result.loss('mse')}, LMSE: {result.loss('lmse')}, angular: {result.loss('angular')}.\n",
              sep="", end="\n")

    def test_one(self):
        result = try_udm(self.dataset, KernelType.Normal, self.dataset.n_components)
        self.log_message(result)
        assert isinstance(result, UDMResult)

    def test_has_x0(self):
        result = try_udm(self.dataset, KernelType.Normal, self.dataset.n_components, x0=self.x0, pretrain_epochs=100)
        self.log_message(result)
        assert isinstance(result, UDMResult)

    def test_cuda(self):
        result = try_udm(self.dataset, KernelType.Normal, self.dataset.n_components, x0=self.x0,
                         device="cuda", pretrain_epochs=100)
        self.log_message(result)
        assert isinstance(result, UDMResult)

    def test_cuda0(self):
        result = try_udm(self.dataset, KernelType.Normal, self.dataset.n_components, x0=self.x0,
                         device="cuda:0", pretrain_epochs=100)
        self.log_message(result)
        assert isinstance(result, UDMResult)

    def test_no_device(self):
        with pytest.raises(AssertionError):
            result = try_udm(self.dataset, KernelType.Normal, self.dataset.n_components, x0=self.x0,
                             device="cuda:1", pretrain_epochs=100)

    def test_progress_callback(self):
        def callback(p: float):
            assert 0.0 <= p <= 1.0

        result = try_udm(self.dataset, KernelType.Normal, self.dataset.n_components,
                         pretrain_epochs=100, progress_callback=callback)

    def test_result_properties(self):
        result = try_udm(self.dataset, KernelType.Normal, self.dataset.n_components)
        properties = ["dataset", "n_samples", "n_components", "n_classes", "n_iterations", "kernel_type",
                      "proportions", "components", "time_spent", "x0", "history", "settings"]
        for prop in properties:
            assert hasattr(result, prop)

    def test_history(self):
        result = try_udm(self.dataset, KernelType.Normal, self.dataset.n_components)
        proportions_bytes = result.proportions.tobytes()
        components_bytes = result.components.tobytes()
        for h in result.history:
            assert isinstance(h, UDMResult)
            assert h is not result
            # will not modify the data of original object
            assert result.proportions.tobytes() == proportions_bytes
            assert result.components.tobytes() == components_bytes

    def test_loss(self):
        result = try_udm(self.dataset, KernelType.Normal, self.dataset.n_components)
        for loss_name in built_in_losses:
            loss_series = result.loss_series(loss_name)
            assert isinstance(loss_series, np.ndarray)
            assert len(loss_series) == result.n_iterations
            class_wise_losses = result.class_wise_losses(loss_name)
            assert isinstance(class_wise_losses, np.ndarray)
            assert len(class_wise_losses) == result.n_classes
            sample_wise_losses = result.sample_wise_losses(loss_name)
            assert isinstance(sample_wise_losses, np.ndarray)
            assert len(sample_wise_losses) == result.n_samples
