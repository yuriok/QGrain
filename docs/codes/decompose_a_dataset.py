import pickle
import numpy as np

from QGrain.models import KernelType
from QGrain.generate import random_dataset, SIMPLE_PRESET
from QGrain.ssu import try_ssu
from QGrain.emma import try_emma
from QGrain.udm import try_udm

dataset = random_dataset(**SIMPLE_PRESET, n_samples=200)
x0 = np.mean(dataset.parameters, axis=0)
ssu_results = []
for sample in dataset:
    result, msg = try_ssu(sample, dataset.distribution_type, dataset.n_components, x0=x0)
    assert result is not None
    ssu_results.append(result)
kernel_type = KernelType.__members__[dataset.distribution_type.name]
emma_result = try_emma(dataset, kernel_type, dataset.n_components, x0=x0[:-1])
udm_result = try_udm(dataset, kernel_type, dataset.n_components, x0=x0[:-1])

with open("./results.ssu", "wb") as f:
    pickle.dump(ssu_results, f)
with open("./result.emma", "wb") as f:
    pickle.dump(emma_result, f)
with open("./result.udm", "wb") as f:
    pickle.dump(udm_result, f)
