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

from pyquil.gates import *
from grove.qft.fourier import inverse_qft
import pyquil.quil as pq
from grove.pyqaoa.utils import compare_progs # This would be nice to have in
                                             # general purpose Util module


def test_simple_inverse_qft():
    
    trial_prog = pq.Program()
    trial_prog.inst(X(0))
    trial_prog = trial_prog + inverse_qft([0])
    
    result_prog = pq.Program().inst([X(0), H(0)])
    
    compare_progs(trial_prog, result_prog)


def test_multi_qubit_qft():
    
    trial_prog = pq.Program()
    trial_prog.inst(X(0), X(1), X(2))
    trial_prog = trial_prog + inverse_qft([0, 1, 2])
    
    result_prog = pq.Program().inst([X(0), X(1), X(2),
                                     SWAP(0, 2), H(0),
                                     CPHASE(-1.5707963267948966, 0, 1),
                                     CPHASE(-0.7853981633974483, 0, 2),
                                     H(1), CPHASE(-1.5707963267948966, 1, 2),
                                     H(2)])
    
    compare_progs(trial_prog, result_prog)
