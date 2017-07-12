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

import sys, os
#Add the analytical gradient directory
dirname = os.path.dirname(os.path.abspath(__file__))
analytical_gradient_dir = os.path.join(dirname, '../analytical_gradient')
sys.path.append(analytical_gradient_dir)

import pytest

import maxcut_qaoa_core
import analytical_gradient

def test_edges_to_graph():
    n_2_path_graph_edges = [(0,1)]
    n_2_path_graph = maxcut_qaoa_core.edges_to_graph(n_2_path_graph_edges)
    assert n_2_path_graph.nodes() == [0,1]
