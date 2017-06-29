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

"""
This module performs a comparative analysis of the repetition costs associated
with different optimization methods
Todo:
    -Generate random 3-regular graphs of sufficiently large size
    -Provide evidence that the new algorithm offers a general speed-up
    -Reproduce the given stopping criteria
    -Record wall-clock time
"""

import numpy as np
import networkx as nx
from grove.pyqaoa.maxcut_qaoa import maxcut_qaoa
from minimizer_kwargs_options import minimizer_kwargs_dict

num_edges_for_each_node = 3 #Analysis on 3-regular graphs
num_nodes = 5
num_random_graphs = 5
a_random_graph = nx.random_regular_graph(num_edges_for_each_node, num_nodes)
random_graphs = [
square_ring = [(0,1), (1,2), (2,3), (3,0)]
trotterization_steps = 2 #Referred to as "p" in the paper

inst = maxcut_qaoa(square_ring, steps=trotterization_steps,
        minimizer_kwargs=minimizer_kwargs_dict['L-BFGS-B'])
betas, gammas = inst.get_angles()

probs = inst.probabilities(np.hstack((betas, gammas)))
for state, prob in zip(inst.states, probs):
    print state, prob
print "Repetition Cost: "
print inst.repetition_cost

print "Most frequent bitstring from sampling:"
most_freq_string, sampling_results = inst.get_string(
	betas, gammas)
print most_freq_string
