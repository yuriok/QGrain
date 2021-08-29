import os

from setuptools import find_packages, setup

setup(
    name="QGrain",
    version="0.3.4.2",
    description="An easy-to-use software for the comprehensive analysis of grain-size distributions",
    platforms="all",
    author="Yuming Liu",
    author_email="liuyuming@ieecas.cn",
    url="https://github.com/yuriok/QGrain",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "PySide2",
        "qtawesome",
        "scikit-learn",
        "torch",
        "xlrd",
        "openpyxl",
        "SciencePlots"],
    entry_points = {
        'console_scripts': [
            'QGrain=QGrain.entry:qgrain_console'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
