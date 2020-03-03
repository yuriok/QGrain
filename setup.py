from setuptools import setup, find_packages
import os

setup(
    name="QGrain",
    version="0.2.7.2",
    description="QGrain is an esay to use tool that can unmix and analyse the multi-modal grain size distribution.",
    platforms="all",
    author="Yuming Liu",
    author_email="liuyuming@ieecas.cn",
    url="https://github.com/QGrain-Organization/QGrain",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.18.1',
        'PySide2>=5.14.1',
        "scikit-learn>=0.22.1",
        "scipy>=1.4.1",
        "shiboken2>=5.14.1",
        "xlrd>=1.2.0",
        "XlsxWriter>=1.2.7",
        "xlwt>=1.3.0",
        "Pillow>=7.0.0",
        "opencv-python>=4.2.0.32"
    ],
    entry_points = {
        'console_scripts': ['qgrain=QGrain.main:main'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
