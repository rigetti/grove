##############################################################################
# Copyright 2017-2018 Rigetti Computing
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

import numpy as np
import pytest
from mock import Mock, patch
from pyquil.api import QPUConnection
from pyquil.api.errors import DeviceRetuningError
from pyquil.device import Qubit, Edge
from pyquil.gates import X

import grove.tomography.qpu_characterization as qc
import grove.tomography.utils as ut
from grove.tomography.process_tomography import ProcessTomography

qt = ut.import_qutip()
cvxpy = ut.import_cvxpy()

if not qt:
    pytest.skip("Qutip not installed, skipping tests", allow_module_level=True)

if not cvxpy:
    pytest.skip("CVXPY not installed, skipping tests", allow_module_level=True)


def test_estimate():

    dummy_isa = qc.ISA(
        name='dummy',
        version='0.0',
        timestamp=0,
        qubits=[
           Qubit(0, "Xhalves", False),
           Qubit(1, "Xhalves", False)
        ],
        edges=[
           Edge((0, 1), "CZ", False)
        ]
    )
    dummy_cliques = [dummy_isa.edges]

    with patch("grove.tomography.qpu_characterization.parallel_process_tomographies") as ppt:

        tomo = Mock(spec=ProcessTomography)
        tomo.to_kraus.return_value = [
            np.array([[1+0j]])
        ]
        tomo.avg_gate_fidelity.return_value = 0.99
        has_failed = [False]

        def mock_ppt(gate, cliques, nsamples, cxn, **kwargs):
            if not has_failed[0]:
                has_failed[0] = True
                raise DeviceRetuningError("Failure!")
            return [tomo for _ in cliques], None, [np.array([[.8]])] * len(cliques)

        ppt.side_effect = mock_ppt
        res = qc.estimate(dummy_isa, 100, Mock(spec=QPUConnection), dummy_cliques, retune_sleep=1.)

        assert res.gates[0].kraus_ops[0].tolist() == [[1+0j]]
        assert res.gates[0].fidelity == 0.99

        assert qc.NoiseModel.from_dict(res.to_dict()) == res

    with pytest.raises(ValueError):
        qc.estimate(dummy_isa, 100, Mock(spec=QPUConnection), [], retune_sleep=1.)


def test_parallel_sample_assignment_probs():
    cxn = Mock(spec=QPUConnection)
    nsamples = 100

    res00 = [[0, 0]]*nsamples
    res11 = [[1, 1]]*nsamples

    res01 = [[0, 1]]*nsamples
    res10 = [[1, 0]]*nsamples

    cxn.run_and_measure.side_effect = [
        res00,
        res11,
    ]

    res = qc.parallel_sample_assignment_probs([[0], [1]], nsamples, cxn, shuffle=False)
    assert [r.tolist() for r in res] == [[[1., 0],
                                          [0., 1.]],
                                         [[1., 0],
                                          [0., 1.]]]

    with pytest.raises(ValueError):
        qc.parallel_sample_assignment_probs([[0], [1, 2]], nsamples, cxn, shuffle=False)


def test_parallel_process_tomographies():
    with patch("grove.tomography.qpu_characterization.parallel_sample_assignment_probs") as pap, \
         patch("grove.tomography.qpu_characterization.run_in_parallel") as rip:

        pap.return_value = [np.eye(2)]
        rip.return_value = [np.zeros(4*4*2)]

        qc.parallel_process_tomographies(X, [[0]], 100, Mock(spec=QPUConnection))

    with pytest.raises(ValueError):
        qc.parallel_process_tomographies(X, [[0], [1, 2]], 100, Mock(spec=QPUConnection))
