#!/usr/bin/python

from setuptools import setup

setup(
    name = "pyQAOA",
    version = "0.0.0",
    author = "Rigetti Computing",
    description = "A Quantum Approximate Optimization Algorithm",
    packages = ["pyqaoa"],
    install_requires = [
        'numpy',
        'scipy',
        'pyquil',
        'mock',
        'networkx',
        'matplotlib'
    ]
    #license = "",
    # long_description=read('README'),
)
