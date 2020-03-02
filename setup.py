from setuptools import setup, find_packages
import os

setup(
    name="QGrain",
    version="0.2.7",
    description="QGrain is an esay to use tool that can unmix and analyse the multi-modal grain size distribution.",
    platforms="all",
    author="Yuming Liu",
    author_email="liuyuming@ieecas.cn",
    url="https://github.com/QGrain-Organization/QGrain",
    license="MIT",
    packages=find_packages(),
    include_package_data=True)
