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
from demo_options import minimizer_kwargs_dict

test_graph = [(0,1)]
trotterization_steps = 1 #Referred to as "p" in the paper

inst = maxcut_qaoa(test_graph, steps=trotterization_steps,
        minimizer_kwargs=minimizer_kwargs_dict['L-BFGS-B'])
betas, gammas = inst.get_angles()
print("betas: " + str(betas))
print("gammas: " + str(gammas))

probs = inst.probabilities(np.hstack((betas, gammas)))
for state, prob in zip(inst.states, probs):
    print state, prob
print "Repetition Cost: "
print inst.repetition_cost

print "Most frequent bitstring from sampling:"
most_freq_string, sampling_results = inst.get_string(
	betas, gammas)
print most_freq_string
