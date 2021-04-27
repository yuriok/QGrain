import pickle

import matplotlib.pyplot as plt

from QGrain.algorithms.ClassicResolver import ClassicResolver
from QGrain.algorithms.distributions import BaseDistribution
from QGrain.algorithms.moments import logarithmic
from QGrain.models.ClassicResolverSetting import ClassicResolverSetting
from QGrain.models.EMAResult import EMAResult
from QGrain.models.FittingResult import FittingResult
from QGrain.models.FittingTask import FittingTask

plt.style.use(["science", "no-latex"])
import os

from QGrain.algorithms import FittingState

dump_filename = r"test_ssu.dump"
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
result = results[2]

resolver = ClassicResolver()
resolver_setting = ClassicResolverSetting()


em_moments = [logarithmic(result.dataset.classes_φ, result.end_members[i]) for i in range(result.n_members)]
plt.figure(figsize=(4, 3))
for i, (distribution, moments) in enumerate(zip(result.end_members, em_moments)):
    plt.plot(result.dataset.classes_μm, distribution, label=f"EM{i+1} (m={moments['mean']:0.2f}, σ={moments['std']:0.2f})")
plt.xscale("log")
plt.xlabel("$Grain-size [μm]$")
plt.ylabel("Frequency")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

reference = [dict(mean=moments["mean"], std=moments["std"], skewness=moments["skewness"]) for moments in em_moments]
reference.sort(key=lambda x: x["mean"])
print(f"EM references: {reference}.")


plt.figure(figsize=(8, 4))
plt.ion()
plt.show()
def plot_result(result: FittingResult):
    plt.scatter(result.classes_μm, result.sample.distribution, s=2, c="black", label="Target")
    plt.plot(result.classes_μm, result.distribution, c="black", label="Mixed")
    for i, comp in enumerate(result.components):
        plt.plot(result.classes_μm, comp.distribution*comp.fraction, label=f"C{i+1}")
    plt.xscale("log")
    plt.xlabel("$Grain-size [μm]$")
    plt.ylabel("Frequency")
    plt.legend(loc="upper left")

for i, (sample, fractions) in enumerate(zip(result.dataset.samples, result.fractions)):
    if i < 1000:
        continue
    print(f"Trying the sample {sample.name}.")
    valid_reference = []
    valid_fractions = []
    for f, ref in zip(fractions, reference):
        if f > 0.01:
            valid_reference.append(ref)
            valid_fractions.append(f)
    print(f"Valid N_members for this sample is {len(valid_reference)}.")
    initial_guess = BaseDistribution.get_initial_guess(result.distribution_type, valid_reference, fractions=valid_fractions)
    # print(f"Initial guesss: {initial_guess}.")
    task_has_with = FittingTask(
        sample,
        result.distribution_type,
        len(valid_reference),
        resolver_setting=resolver_setting,
        initial_guess=initial_guess)
    task_has_without = FittingTask(
        sample,
        result.distribution_type,
        len(valid_reference),
        resolver_setting=resolver_setting)
    state_without, ssu_result_without = resolver.try_fit(task_has_without)
    state_with, ssu_result_with = resolver.try_fit(task_has_with)
    if state_without == FittingState.Succeeded and state_with == FittingState.Succeeded:
        plt.clf()
        plt.subplot(1, 2, 1)
        plot_result(ssu_result_without)
        plt.title("Without EMA reference")

        plt.subplot(1, 2, 2)
        plot_result(ssu_result_with)
        plt.title("With EMA reference")
        plt.pause(0.1)
        if i == 0:
            plt.tight_layout()
    else:
        print(f"Fitting failed: {state_without.name}, {state_with.name}.")


# EMA can not find the correct EMs for some real-world samples, if the varation of sedimentary facies is very frequent.
