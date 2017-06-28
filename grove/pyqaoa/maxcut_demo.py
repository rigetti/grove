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
This module demonstrates the code presented in the README
"""

import numpy as np
from grove.pyqaoa.maxcut_qaoa import maxcut_qaoa


square_ring = [(0,1), (1,2), (2,3), (3,0)]
trotterization_steps = 2
inst = maxcut_qaoa(square_ring, steps=trotterization_steps)
betas, gammas = inst.get_angles()

probs = inst.probabilities(np.hstack((betas, gammas)))
for state, prob in zip(inst.states, probs):
    print state, prob
print "Num Steps: "
print len(inst.states)

print "Most frequent bitstring from sampling:"
most_freq_string, sampling_results = inst.get_string(
	betas, gammas)
print most_freq_string
