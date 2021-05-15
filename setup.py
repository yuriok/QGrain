import os

from setuptools import find_packages, setup

setup(
    name="QGrain",
    version="0.3.2",
    description="QGrain is an esay to use tool that can analyse the grain-size distributions of sediments.",
    platforms="all",
    author="Yuming Liu",
    author_email="liuyuming@ieecas.cn",
    url="https://github.com/QGrain-Organization/QGrain",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "torch",
        "xlrd",
        "openpyxl",
        "PySide2",
        "qtawesome",
        "matplotlib",
        "SciencePlots"],
    entry_points = {
        'console_scripts': [
            'qgrain=QGrain.entry:qgrain_console'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
