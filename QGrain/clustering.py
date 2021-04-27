import os
import pickle
from time import time

import matplotlib.pyplot as plt
import numpy as np
from QGrain.models.DataLoader import (DataLayoutSettings, DataLoader,
                                      ReadFileType)
from QGrain.models.GrainSizeDataset import GrainSizeDataset
from scipy import ndimage
from sklearn import datasets, manifold
from sklearn.cluster import OPTICS, AgglomerativeClustering, KMeans
from sklearn.decomposition import NMF, PCA, KernelPCA
from sklearn.preprocessing import StandardScaler

plt.style.use(["science", "no-latex"])

cmap = plt.get_cmap("tab10")


binary_data_path = "./WB1_GrainSizeData.dump"
if os.path.exists(binary_data_path):
    dataset = GrainSizeDataset.load(binary_data_path)
else:
    loader = DataLoader()
    data_layout = DataLayoutSettings()
    dataset = loader.try_load_data(r"C:\Users\yurio\Desktop\WB1粒度数据.xlsx", ReadFileType.XLSX, data_layout)
    GrainSizeDataset.dump(dataset, binary_data_path)

result_path = "D:\\Clusering Results"
classes_μm = dataset.classes_μm
classes_φ = dataset.classes_φ
X = dataset.X

X_to_clusering = X
scalar = StandardScaler()


pca = PCA(n_components=0.99)
pca_result = pca.fit_transform(X)
plt.figure(figsize=(6, 4))
for i in range(pca_result.shape[1]):
    plt.plot(pca_result[:, i], label=f"PC_{i+1}", c=cmap(i))
plt.legend()
plt.show()
plt.figure(figsize=(6, 4))
for comp in pca.components_:
    plt.plot(classes_μm, comp, label=f"PC_{i+1}")
plt.xlabel("Grain-size (μm)")
plt.ylabel("Frequency")
plt.xscale("log")
plt.title("PC")
plt.tight_layout()
plt.show()

plt.figure(figsize=(4, 3))
for x in X:
    plt.plot(classes_μm, x, c="grey", linewidth=0.1)
plt.xlabel("Grain-size (μm)")
plt.ylabel("Frequency")
plt.xscale("log")
plt.title("Raw Sample")
plt.tight_layout()
plt.savefig(os.path.join(result_path, "Raw Sample.png"), dpi=300)
plt.savefig(os.path.join(result_path, "Raw Sample.svg"))


X_to_clusering = pca_result
# setting distance_threshold=0 ensures we compute the full tree.
# clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
# clustering.fit_predict(X_to_clusering)
from scipy.cluster.hierarchy import dendrogram, fcluster, fclusterdata, linkage

# def plot_dendrogram(model, **kwargs):
#     # Create linkage matrix and then plot the dendrogram

#     # create the counts of samples under each node
#     counts = np.zeros(model.children_.shape[0])
#     n_samples = len(model.labels_)
#     for i, merge in enumerate(model.children_):
#         current_count = 0
#         for child_idx in merge:
#             if child_idx < n_samples:
#                 current_count += 1  # leaf node
#             else:
#                 current_count += counts[child_idx - n_samples]
#         counts[i] = current_count

#     linkage_matrix = np.column_stack([model.children_, model.distances_,
#                                       counts]).astype(float)
#     dendrogram(linkage_matrix, **kwargs)
#     return linkage_matrix


linkage_matrix = linkage(X_to_clusering, method="ward")

plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, no_labels=False, p=100, truncate_mode='lastp', show_contracted=True)
plt.title("Hierarchy Clustering (p=100)")
plt.xlabel("Sample Count")
plt.ylabel("Distance")
# plt.grid()
# plt.yscale("log")
plt.tight_layout()
plt.savefig(os.path.join(result_path, "Truncate_p100.png"), dpi=300)
plt.savefig(os.path.join(result_path, "Truncate_p100.svg"))
plt.show()

t_distance = 1.4
cluster_labels = fcluster(linkage_matrix, t=t_distance, criterion="distance")

flags = set(cluster_labels)
n_clusters = len(flags)

plt.figure(figsize=(3, 2))
for x, y in zip(X, cluster_labels):
    plt.plot(classes_μm, x, c=cmap(y-1), linewidth=0.1)
plt.xlabel("Grain-size (μm)")
plt.ylabel("Frequency")
plt.xscale("log")
plt.title("Clustering Result")
plt.tight_layout()
plt.savefig(os.path.join(result_path, "Clustering Result.png"), dpi=300)
plt.savefig(os.path.join(result_path, "Clustering Result.svg"))
plt.show()

for f in flags:
    plt.figure(figsize=(3, 2))
    for x, y in zip(X, cluster_labels):
        if y == f:
            plt.plot(classes_μm, x, c="gray", linewidth=0.1)
    plt.xlabel("Grain-size (μm)")
    plt.ylabel("Frequency")
    plt.xscale("log")
    # plt.title(f"Cluster_{f}")
    # plt.xticks([])
    # plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, f"Cluster_{f}.png"), dpi=300)
    plt.savefig(os.path.join(result_path, f"Cluster_{f}.svg"))
    plt.show()
