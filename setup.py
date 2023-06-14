from setuptools import find_packages, setup

setup(
    name="QGrain",
    version="0.5.4.2",
    description="An easy-to-use software for the comprehensive analysis of grain size distributions",
    platforms="all",
    author="Yuming Liu",
    author_email="liuyuming@ieecas.cn",
    url="https://github.com/yuriok/QGrain",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "psutil",
        "numpy",
        "scipy",
        "scikit-learn",
        "grpcio",
        "protobuf",
        "xlrd",
        "openpyxl",
        "PySide6>=6.2.0",
        "matplotlib>=3.5.0",
        "SciencePlots",
        "qt-material"],
    extras_require={"server": ["torch"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Natural Language :: Chinese (Simplified)",
        "Natural Language :: English"

    ],
    entry_points={
        "console_scripts": ["qgrain=QGrain:main"]
    }
)
