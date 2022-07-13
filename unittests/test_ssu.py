import numpy as np
import pytest

from QGrain.distributions import DistributionType
from QGrain.generate import random_dataset, SIMPLE_PRESET
from QGrain.models import SSUResult
from QGrain.ssu import *


class TestTrySSU:
    dataset = random_dataset(**SIMPLE_PRESET, n_samples=100)
    x0 = np.array([[mean for (mean, std) in component] for component in SIMPLE_PRESET["target"]]).T
    x0 = x0[1:]

    @classmethod
    def log_message(cls, result: SSUResult, message: str):
        print("\n", f"The fitting task [{len(result)}, {result.distribution_type.name}] of sample [{result.name}] "
                    f"was finished using {result.n_iterations} iterations, message: {message}.\n"
                    f"MSE: {result.loss('mse')}, LMSE: {result.loss('lmse')}, angular: {result.loss('angular')}.\n"
                    f"Target Mz: ({', '.join([f'{c.mean:.2f}' for c in result.sample])}), "
                    f"Estimated Mz: ({', '.join([f'{c.mean:.2f}' for c in result])}).\n"
                    f"Target So: ({', '.join([f'{c.sorting_coefficient:.2f}' for c in result.sample])}), "
                    f"Estimated So: ({', '.join([f'{c.sorting_coefficient:.2f}' for c in result])}).\n"
                    f"Target p: ({', '.join([f'{c.proportion:.2f}' for c in result.sample])}), "
                    f"Estimated p: ({', '.join([f'{c.proportion:.2f}' for c in result])}).", sep="", end="\n")

    def test_one(self):
        result, message = try_ssu(self.dataset[0], DistributionType.Normal, self.dataset.n_components,
                                  loss="lmse")
        self.log_message(result, message)
        assert isinstance(result, SSUResult)

    def test_has_x0(self):
        result, message = try_ssu(self.dataset[0], DistributionType.Normal, self.dataset.n_components,
                                  x0=self.dataset.parameters[0, 1:, :], loss="lmse")
        self.log_message(result, message)
        assert isinstance(result, SSUResult)

    def test_try_global(self):
        result, message = try_ssu(self.dataset[0], DistributionType.Normal, self.dataset.n_components, try_global=True)
        self.log_message(result, message)
        assert isinstance(result, SSUResult)

    def test_all_samples(self):
        for i, sample in enumerate(self.dataset):
            result, message = try_ssu(sample, DistributionType.Normal, self.dataset.n_components, x0=self.x0)
            assert isinstance(result, SSUResult)

    def test_try_dataset(self):
        options = dict(x0=self.x0)
        results, failed_indexes = try_dataset(self.dataset, DistributionType.Normal,
                                              self.dataset.n_components, n_processes=4,
                                              options=options)
        print("\n", "Using try_dataset to fit all samples", len(results), len(failed_indexes))

    def test_all_losses(self):
        for loss in built_in_losses:
            result, message = try_ssu(self.dataset[0], DistributionType.Normal, self.dataset.n_components,
                                      x0=self.x0, loss=loss)
            if isinstance(result, SSUResult):
                print(loss)
                self.log_message(result, message)
            else:
                print("\n", loss, message, end="\n")

    def test_all_optimizers(self):
        for optimizer in built_in_optimizers:
            result, message = try_ssu(self.dataset[0], DistributionType.Normal, self.dataset.n_components,
                                      x0=self.x0, loss="rmse", optimizer=optimizer)
            if isinstance(result, SSUResult):
                print(optimizer)
                self.log_message(result, message)
            else:
                print("\n", optimizer, message, end="\n")


if __name__ == "__main__":
    pytest.main(["-s"])
