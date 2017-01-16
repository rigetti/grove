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
    version="0.0.2",
    author="Rigetti Computing",
    author_email="softapps@rigetti.com",
    description="A collection of quantum algorithms built using pyQuil and Forest",
    url="https://github.com/rigetticomputing/grove.git",
    download_url="https://github.com/rigetticomputing/grove/tarball/0.0.2",
    packages=["grove", "grove.pyqaoa", "grove.pyvqe", "grove.qft",
              "grove.phaseestimation", "grove.teleport"],
    install_requires=[
        'numpy',
        'scipy',
        'pyquil',
        'mock',
        'networkx',
        'matplotlib'
    ],
    license='LICENSE',
    keywords='quantum quil programming hybrid'
)
