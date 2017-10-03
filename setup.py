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

from setuptools import setup

setup(
    name="quantum-grove",
    version="1.1.0",
    author="Rigetti Computing",
    author_email="softapps@rigetti.com",
    description="A collection of quantum algorithms built using pyQuil and Forest",
    url="https://github.com/rigetticomputing/grove.git",
    packages=[
        "grove",
        "grove.pyqaoa",
        "grove.pyvqe",
        "grove.qft",
        "grove.alpha",
        "grove.alpha.phaseestimation",
        "grove.alpha.teleport",
        "grove.alpha.deutsch_jozsa",
        "grove.alpha.arbitrary_state",
        "grove.alpha.bernstein_vazirani",
        "grove.alpha.simon",
        "grove.alpha.amplification",
        "grove.alpha.fermion_transforms"
    ],
    install_requires=[
        'numpy',
        'scipy',
        'pyquil >= 1.0.0',
        'mock',
        'networkx',
        'matplotlib'
    ],
    setup_requires=['pytest-runner'],
    tests_require=[
        'pytest >= 3.0.0',
        'mock'
    ],
    license='LICENSE',
    keywords='quantum quil programming hybrid'
)
