#!/usr/bin/python
##############################################################################
# Copyright 2016-2017 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################

from setuptools import setup, find_packages
from grove import __version__

setup(
    name="quantum-grove",
    version=__version__,
    author="Rigetti Computing",
    author_email="softapps@rigetti.com",
    description="A collection of quantum algorithms built using pyQuil and Forest",
    url="https://github.com/rigetticomputing/grove.git",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
        'numpy',
        'scipy',
        'pyquil >= 1.6.1',
        'funcsigs',
        'mock',
        'networkx',
        'matplotlib'
    ],
    extras_require={
        'tomography': [
            'cython',
            'cvxpy',
            'scs',
            'qutip'
        ]
    },
    setup_requires=[
        'pytest-runner',
        'numpy'
    ],
    tests_require=[
        'tox',
        'pytest >= 3.0.0',
        'mock'
    ],
    license='LICENSE',
    keywords='quantum quil programming hybrid'
)
