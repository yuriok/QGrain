import pickle

import matplotlib.pyplot as plt

from QGrain.algorithms.ClassicResolver import ClassicResolver
from QGrain.algorithms.distributions import BaseDistribution
from QGrain.algorithms.moments import logarithmic
from QGrain.models.ClassicResolverSetting import ClassicResolverSetting
from QGrain.models.EMAResult import EMAResult
from QGrain.models.FittingResult import FittingResult
from QGrain.models.FittingTask import FittingTask
from QGrain.algorithms.ema import EMAResolver
from QGrain.models.GrainSizeDataset import GrainSizeDataset

plt.style.use(["science", "no-latex"])
import os

from QGrain.algorithms import FittingState

dump_filename = r"C:\Users\yurio\Desktop\WB1_Normal_EMA.dump"
with open(dump_filename, "rb") as f:
    results = pickle.load(f) # type: list[EMAResult]
    print(f"There are {len(results)} EMA results in this dump file.")
    for i, result in enumerate(results):
        print(f"Result {i+1}: N_samples={result.n_samples}, "+\
            f"N_classes={result.n_classes}, "+\
            f"Distribution type={result.distribution_type.name}, "+\
            f"N_members={result.n_members}, "+\
            f"N_iterations={result.n_iterations}")

plt.figure(figsize=(4, 3))
n_members_series = [result.n_members for result in results]
diatances_series = [result.get_distance("angular") for result in results]
plt.plot(n_members_series, diatances_series, marker=".", ms=8, mec="None")
plt.xlabel("$N_{members}$")
plt.ylabel("Angular distance (°)")
plt.tight_layout()
plt.show()

print("Using EMA result (N_members=4) for SSU reference.")
result = results[4]


dataset = result.dataset

from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
import os
cluser = AgglomerativeClustering(n_clusters=5)
tags = cluser.fit_predict(dataset.X)
tag_set = set(tags)
for tag in tag_set:
    plt.figure(figsize=(4, 3))
    key = np.equal(tags, tag)
    for d in dataset.X[key]:
        plt.plot(dataset.classes_μm, d, c="gray")
    plt.xscale("log")
    plt.xlabel("$Grain-size [μm]$")
    plt.ylabel("Frequency")
    plt.tight_layout()
    # filename = os.path.join(r"C:\Users\yurio\Desktop\figures", f"tag={tag}.png")
    # plt.savefig(filename)
plt.show()


new_dataset = GrainSizeDataset()
names = []
distributions = []
selected_tag = 0
for i, tag in enumerate(tags):
    if tag == selected_tag:
        names.append(dataset.samples[i].name)
        distributions.append(dataset.samples[i].distribution)
new_dataset.add_batch(dataset.classes_μm, names, distributions)

ema_resolver = EMAResolver()
ema_results = []
from QGrain.models.NNResolverSetting import NNResolverSetting
ema_setting = NNResolverSetting(max_niter=400, lr=0.1)
for n_members in range(1, 11):
    ema_result = ema_resolver.try_fit(new_dataset,
                                      result.distribution_type,
                                      n_members=n_members,
                                      resolver_setting=ema_setting)
    ema_results.append(ema_result)

with open("./test_ssu.dump", "wb") as f:
    pickle.dump(ema_results, f)
