from QGrain.artificial._param import *
from QGrain.artificial._sample import *
from QGrain.artificial._setting import *

LOESS = [dict(shape=(0.0, 0.10), loc=(10.2, 0.1), scale=(1.1, 0.1), weight=(1.0, 0.1)),
         dict(shape=(0.0, 0.10), loc=(7.5, 0.1), scale=(1.2, 0.1), weight=(2.0, 0.1)),
         dict(shape=(0.0, 0.10), loc=(5.0, 0.2), scale=(1.0, 0.1), weight=(4.0, 0.2))]

LACUSTRINE = [dict(shape=(0.0, 0.10), loc=(10.2, 0.1), scale=(1.1, 0.1), weight=(1.0, 0.1)),
              dict(shape=(0.0, 0.10), loc=(7.5, 0.1), scale=(1.2, 0.1), weight=(2.0, 0.1)),
              dict(shape=(0.0, 0.10), loc=(5.0, 0.2), scale=(1.0, 0.1), weight=(4.0, 0.2)),
              dict(shape=(0.0, 0.10), loc=(2.2, 0.4), scale=(1.0, 0.2), weight=(3.0, 1.0))]

def get_random_dataset(target=LOESS, n_samples=100,
                       min_μm=0.02, max_μm=2000.0, n_classes=101,
                       precision=4, noise=5):
    random_setting = RandomSetting(target)
    params_array = random_setting.get_random_params(n_samples=n_samples)
    dataset = ArtificialDataset(params_array,
                                min_μm=min_μm, max_μm=max_μm, n_classes=n_classes,
                                precision=precision, noise=noise)
    return dataset


def get_random_sample(target=LOESS,
                      min_μm=0.02, max_μm=2000.0, n_classes=101,
                      precision=4, noise=5):
    random_setting = RandomSetting(target)
    params_array = random_setting.get_random_params(n_samples=1)
    dataset = ArtificialDataset(params_array,
                                min_μm=min_μm, max_μm=max_μm, n_classes=n_classes,
                                precision=precision, noise=noise)
    sample = dataset.get_sample(0)
    sample.name = "Artificial Sample"
    return sample
