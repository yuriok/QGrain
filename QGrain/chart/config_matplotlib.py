__all__ = [
    "normal_color",
    "highlight_color",
    "synchronize_theme",
    "setup_matplotlib"]

import os

import matplotlib.pyplot as plt


def normal_color():
    return os.environ["QTMATERIAL_SECONDARYTEXTCOLOR"]

def highlight_color():
    return os.environ["QTMATERIAL_PRIMARYCOLOR"]

def synchronize_theme():
    plt.rcParams["axes.facecolor"] = os.environ["QTMATERIAL_SECONDARYCOLOR"]
    plt.rcParams["figure.facecolor"] = os.environ["QTMATERIAL_SECONDARYCOLOR"]
    plt.rcParams["axes.edgecolor"] = os.environ["QTMATERIAL_SECONDARYTEXTCOLOR"]
    plt.rcParams["axes.titlecolor"] = os.environ["QTMATERIAL_SECONDARYTEXTCOLOR"]
    plt.rcParams["axes.labelcolor"] = os.environ["QTMATERIAL_SECONDARYTEXTCOLOR"]
    plt.rcParams["grid.color"] = os.environ["QTMATERIAL_SECONDARYTEXTCOLOR"]
    plt.rcParams["legend.labelcolor"] = os.environ["QTMATERIAL_SECONDARYTEXTCOLOR"]
    plt.rcParams["xtick.color"] = os.environ["QTMATERIAL_SECONDARYTEXTCOLOR"]
    plt.rcParams["xtick.labelcolor"] = os.environ["QTMATERIAL_SECONDARYTEXTCOLOR"]
    plt.rcParams["ytick.color"] = os.environ["QTMATERIAL_SECONDARYTEXTCOLOR"]
    plt.rcParams["ytick.labelcolor"] = os.environ["QTMATERIAL_SECONDARYTEXTCOLOR"]

    plt.rcParams["savefig.dpi"] = 300.0
    plt.rcParams["savefig.transparent"] = True
    plt.rcParams["figure.max_open_warning"] = False

def setup_matplotlib():
    plt.style.use(["science", "no-latex"])
    plt.set_cmap("tab10")
    plt.rcParams['axes.unicode_minus'] = False
    synchronize_theme()
