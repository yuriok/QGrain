import sys

import numpy as np
import pytest
from qt_material import apply_stylesheet

from QGrain.charts import *
from QGrain.distributions import DistributionType
from QGrain.emma import *
from QGrain.generate import *
from QGrain.kernels import KernelType
from QGrain.models import SSUResult
from QGrain.ssu import *
from QGrain.udm import *


class TestBoxplotChart:
    datasets = [np.random.rand(100)*0.3+1.0, np.random.rand(100)*0.1+2.0, np.random.rand(100)*0.6+0.5]
    labels = ["Series 1", "Series 2", "Series 3"]
    app = QtWidgets.QApplication(sys.argv)
    apply_stylesheet(app, theme="light_cyan.xml", invert_secondary=True)
    setup_matplotlib()
    os.makedirs("./.temp", exist_ok=True)

    def test_plot(self):
        chart = BoxplotChart()
        chart.show_dataset(self.datasets, self.labels, "Values", "This is a title")
        chart.save_chart("./.temp/boxplot.png", dpi=300.0, transparent=False)


class TestFrequencyChart:
    dataset = random_dataset(**SIMPLE_PRESET, n_samples=100)

    def test_plot(self):
        chart = FrequencyChart()
        chart.show_samples(self.dataset[:10], append=True)
        chart._axes.set_xticks([1e-1, 1e0, 1e1, 1e2, 1e3], ["0.1", "1", "10", "100", "100"])
        chart.save_chart("./.temp/frequency.png", dpi=300.0, transparent=False)

class TestFrequency3DChart:
    dataset = random_dataset(**SIMPLE_PRESET, n_samples=100)

    def test_plot(self):
        chart = Frequency3DChart()
        chart.show_samples(self.dataset[:], append=True)
        chart.save_chart("./.temp/frequency3d.png", dpi=300.0, transparent=False)


class TestCumulativeChart:
    dataset = random_dataset(**SIMPLE_PRESET, n_samples=100)

    def test_plot(self):
        chart = CumulativeChart()
        chart.show_samples(self.dataset[:10], append=True)
        chart._axes.set_xticks([1e-1, 1e0, 1e1, 1e2, 1e3], ["0.1", "1", "10", "100", "100"])
        chart.save_chart("./.temp/cumulative.png", dpi=300.0, transparent=False)


class TestDiagrams:
    dataset_1 = random_dataset(**LOESS_PRESET, n_samples=100)
    dataset_2 = random_dataset(**LACUSTRINE_PRESET, n_samples=100)

    def test_gsm_folk54(self):
        chart = Folk54GSMDiagramChart()
        cmap = plt.get_cmap("tab10")
        chart.show_samples(self.dataset_1[:50], append=True, marker=".", mfc=cmap(0))
        chart.show_samples(self.dataset_2[:50], append=True, marker=".", mfc=cmap(1))
        chart.save_chart("./.temp/folk54_gsm.png", dpi=300.0, transparent=False)

    def test_ssc_folk54(self):
        chart = Folk54SSCDiagramChart()
        cmap = plt.get_cmap("tab10")
        chart.show_samples(self.dataset_1[:50], append=True, marker=".", mfc=cmap(0))
        chart.show_samples(self.dataset_2[:50], append=True, marker=".", mfc=cmap(1))
        chart.save_chart("./.temp/folk54_ssc.png", dpi=300.0, transparent=False)

    def test_gsm_bp12(self):
        chart = BP12GSMDiagramChart()
        cmap = plt.get_cmap("tab10")
        chart.show_samples(self.dataset_1[:50], append=True, marker=".", mfc=cmap(0))
        chart.show_samples(self.dataset_2[:50], append=True, marker=".", mfc=cmap(1))
        chart.save_chart("./.temp/bp12_gsm.png", dpi=300.0, transparent=False)

    def test_ssc_bp12(self):
        chart = BP12SSCDiagramChart()
        cmap = plt.get_cmap("tab10")
        chart.show_samples(self.dataset_1[:50], append=True, marker=".", mfc=cmap(0))
        chart.show_samples(self.dataset_2[:50], append=True, marker=".", mfc=cmap(1))
        chart.save_chart("./.temp/bp12_ssc.png", dpi=300.0, transparent=False)


class TestHierarchicalChart:
    dataset = random_dataset(**SIMPLE_PRESET, n_samples=100)

    def test_plot(self):
        chart = HierarchicalChart()
        chart.show_result(self.dataset)
        chart.save_chart("./.temp/hierarchical.png", dpi=300.0, transparent=False)


class TestPCAResultChart:
    dataset = random_dataset(**SIMPLE_PRESET, n_samples=500)

    def test_plot(self):
        chart = PCAResultChart()
        chart.show_dataset(self.dataset)
        chart.save_chart("./.temp/pca.png", dpi=300.0, transparent=False)


class TestDistributionChart:
    dataset = random_dataset(**SIMPLE_PRESET, n_samples=100)

    def test_plot(self):
        chart = DistributionChart()
        chart.show_chart(self.dataset[0])
        chart.save_chart("./.temp/distribution.png", dpi=300.0, transparent=False)

    def test_animation(self):
        ssu_result, message = try_ssu(self.dataset[0], DistributionType.Normal, self.dataset.n_components)
        assert isinstance(ssu_result, SSUResult)
        chart = DistributionChart()
        chart.show_chart(ssu_result)
        chart.show_animation(ssu_result)
        chart.save_animation("./.temp/ssu_animation.html")


class TestEMMAResultChart:
    dataset = random_dataset(**SIMPLE_PRESET, n_samples=100)
    emma_result = try_emma(dataset, KernelType.Normal, 3, learning_rate=1e-2, max_epochs=500)

    def test_plot(self):
        chart = EMMAResultChart()
        chart.show_chart(self.emma_result)
        chart.save_chart("./.temp/emma.png", dpi=300.0, transparent=False)

    def test_animation(self):
        chart = EMMAResultChart()
        chart.show_chart(self.emma_result)
        chart.show_animation(self.emma_result)
        chart.save_animation("./.temp/emma_animation.html")


class TestUDMResultChart:
    dataset = random_dataset(**SIMPLE_PRESET, n_samples=100)
    x0 = np.array([[mean for (mean, std) in component] for component in SIMPLE_PRESET["target"]]).T
    x0 = x0[1:-1]
    udm_result = try_udm(dataset, KernelType.Normal, 3, x0=x0, learning_rate=1e-2, max_epochs=500)

    def test_plot(self):
        chart = UDMResultChart()
        chart.show_chart(self.udm_result)
        chart.save_chart("./.temp/udm.png", dpi=300.0, transparent=False)

    def test_animation(self):
        chart = UDMResultChart()
        chart.show_chart(self.udm_result)
        chart.show_animation(self.udm_result)
        chart.save_animation("./.temp/udm_animation.html")


if __name__ == "__main__":
    pytest.main(["-s"])
