import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from QGrain.io import DataLayoutSetting, GrainSizeDataset, load_dataset

# layout = DataLayoutSetting(distribution_start_column=4)
# dataset = load_dataset(r"C:\Users\yurio\Desktop\Huxian\Huxian Grain Size Analysis.xlsx",
#                        sheet_index=0, layout=layout)
# with open("./test.dump", "wb") as f:
#     pickle.dump(dataset, f)
with open("./test.dump", "rb") as f:
    dataset = pickle.load(f) # type: GrainSizeDataset

X = dataset.distribution_matrix
pca = PCA()
transformed = pca.fit_transform(X)
labels = [f"{v:0.4f}" for v in dataset.classes_μm]

n_samples, n_features = X.shape
plt.style.use(["science", "no-latex"])
plt.figure(figsize=(6, 5))
cmap = plt.get_cmap("tab10")

plt.subplot(2, 2, 1)
plt.scatter(transformed[:, 0], transformed[:, 1], c="black", s=2.0, alpha=0.05)
plt.plot(pca.components_[0, :], pca.components_[1, :], color="black", lw=1.0, alpha=1.0)
xi = np.argmax(pca.components_[0, :])
yi = np.argmax(pca.components_[1, :])
plt.arrow(0, 0, pca.components_[0, xi], pca.components_[1, xi], color=cmap(0), width=0.001, alpha=1.0)
plt.text(pca.components_[0, xi], pca.components_[1, xi], f"{dataset.classes_μm[xi]: 0.4f}", color=cmap(0), ha='center', va='center')
plt.arrow(0, 0, pca.components_[0, yi], pca.components_[1, yi], color=cmap(1), width=0.001, alpha=1.0)
plt.text(pca.components_[0, yi], pca.components_[1, yi], f"{dataset.classes_μm[yi]: 0.4f}", color=cmap(1), ha='center', va='center')
plt.xlabel("PC1")
plt.ylabel("PC2")
# plt.gca().set_aspect(1.0)

plt.subplot(2, 2, 2)
plt.plot(dataset.classes_μm, pca.components_[0], color=cmap(0), label="PC1")
plt.plot(dataset.classes_μm, pca.components_[1], color=cmap(1), label="PC2")
plt.xscale("log")
# plt.xticks([1e-1, 1e0, 1e1, 1e2, 1e3], ["0.1", "1", "10", "100", "1000"])
plt.xlabel("Grain size [μm]")
plt.ylabel("Transformed value")
plt.legend(loc="upper left")

plt.subplot(2, 1, 2)
for i in range(2):
    plt.plot(transformed[:, i], color=cmap(i), label=f"PC{i+1}", lw=1.0, alpha=0.8)
plt.xlabel("Sample index")
plt.ylabel("Transformed value")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
