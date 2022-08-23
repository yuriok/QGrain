import numpy as np
from QGrain.models import ArtificialDataset, DistributionType
from QGrain.distributions import Normal
from QGrain.io import save_artificial_dataset

distribution_type = DistributionType.Normal
n_components = 3
n_samples = 200

# generate signal series
x = np.linspace(-10, 10, n_samples)
series_1 = np.sin(x)
series_2 = np.cos(x)

# use series to generate parameters
# mean, std are the function parameters of normal distribution
# weight is used to calculate the proportion
C1_mean = np.random.random(n_samples) * 0.05 + 10.2
C1_std = np.random.random(n_samples) * 0.05 + 0.55
C1_weight = np.random.random(n_samples) * 0.01
C2_mean = series_1 * 0.1 + 7.8 + np.random.random(n_samples) * 0.01
C2_std = np.random.random(n_samples) * 0.04 + 0.8
C2_weight = series_1 * 0.2 + 1.0 + np.random.random(n_samples) * 0.01
C3_mean = series_2 * 0.2 + 5.5 + np.random.random(n_samples) * 0.01
C3_std = np.random.random(n_samples) * 0.04 + 0.7
C3_weight = series_2 * 0.4 + 1.0 + np.random.random(n_samples) * 0.01

# pack the parameters
parameters = np.ones((n_samples, Normal.N_PARAMETERS+1, n_components))
parameters[:, 0, 0] = C1_mean
parameters[:, 1, 0] = C1_std
parameters[:, 2, 0] = C1_weight
parameters[:, 0, 1] = C2_mean
parameters[:, 1, 1] = C2_std
parameters[:, 2, 1] = C2_weight
parameters[:, 0, 2] = C3_mean
parameters[:, 1, 2] = C3_std
parameters[:, 2, 2] = C3_weight

# construct the dataset
dataset = ArtificialDataset(parameters, distribution_type,
                            min_size=0.02, max_size=2000, n_classes=101,
                            precision=4, noise=5)
save_artificial_dataset(dataset, "./Artificial Dataset.xlsx")
