#!/usr/bin/env python
######################################################################
# \file setup.py
#######################################################################
from setuptools import setup, find_packages

__author__ = "Giovanni Sutanto, Isabel Rayas, Peter Englert, Ragesh Ramachandran, and Gaurav Sukhatme"
__copyright__ = "Copyright 2020, Robotic Embedded Systems Laboratory (RESL), USC"

install_requires = [
    "dill", "matplotlib", "numpy", "pandas", "plotly", "scipy==1.4.1", 
    "smallab==1.3.3", "tensorflow==2.2.0", "torch", "torchvision", "tqdm"
]
dependency_links = [
    'https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.2.0-cp37-cp37m-manylinux2010_x86_64.whl'
]

setup(
    name="smp_manifold_learning",
    version="1.0",
    author="Giovanni Sutanto",
    author_email="gsutanto@alumni.usc.edu",
    description="Sequential Manifold Planning - Manifold Learning",
    packages=find_packages(),
    install_requires=install_requires,
    dependency_links=dependency_links,
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.7',
)
