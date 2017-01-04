#!/usr/bin/python
from setuptools import setup

setup(
    name="Grove",
    version="0.0.0",
    author="Rigetti Computing",
    author_email="softapps@rigetti.com",
    description="A collection of quantum algorithms built using pyQuil and Forest",
    url="https://github.com/rigetticomputing/grove.git",
    packages=["grove", "grove.pyqaoa", "grove.VQE", "grove.qft",
              "grove.phaseestimation", "grove.teleportation"],
    install_requires=[
        'numpy',
        'scipy',
        'pyquil',
        'mock',
        'networkx',
        'matplotlib'
    ],
    license='Apache2',
    keywords='quantum quil programming hybrid'
)
