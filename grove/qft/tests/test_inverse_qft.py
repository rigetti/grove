##############################################################################
#
#    A collection of tests for the inverse quantum Fourier transform.
#
#    Written by Aaron Vontell on January 28th, 2017
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

import pyquil.forest as Forest
from pyquil.gates import X
from grove.qft.fourier import *

def test_inverse_qft():
    
    qvm = Forest.Connection()
    
    p = pq.Program()
    
    # Apply the QFT to 0, 1, 1, 0, 1, 0 (from n - 1 to 0), apply the QFT^-1,
    # and verify that the original result is obtained
    p.inst(X(1), X(2), X(4))
    p = p + qft([0, 1, 2, 3, 4, 5])
    p = p + inverse_qft([0, 1, 2, 3, 4, 5])
    p.measure(0, 0).measure(1, 1).measure(2, 2).measure(3, 3).measure(4, 4).measure(5, 5)
    wave_fn = qvm.wavefunction(p)
    result = qvm.run(p, [0, 1, 2, 3, 4, 5])
    
    assert result == [[0, 1, 1, 0, 1, 0]]
    # 010110 = 22, accounting for float precision errors
    assert abs(wave_fn[0][22] - 1) < 10 ** -9