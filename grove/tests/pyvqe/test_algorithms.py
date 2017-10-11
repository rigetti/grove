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

from grove.pyvqe.vqe import VQE, parity_even_p
from pyquil.quil import Program
from pyquil.wavefunction import Wavefunction
from pyquil.gates import RX, H, RZ
from pyquil.paulis import PauliSum, PauliTerm
from mock import Mock, MagicMock, patch
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import funcsigs
from collections import OrderedDict
import pytest


def test_vqe_run():
    """VQE initialized and then minimizer is called to return result. Checks
    correct sequence of execution"""
    def param_prog(alpha):
        return Program([H(0), RZ(alpha)(0)])

    hamiltonian = np.array([[1, 0], [0, -1]])
    initial_param = 0.0

    minimizer = MagicMock(spec=minimize, func_code=minimize.__code__)
    fake_result = Mock()
    fake_result.fun = 1.0
    fake_result.x = [0.0]
    fake_result.status = 0  # adding so we avoid writing to logger
    minimizer.return_value = fake_result

    # not actually called in VQE run since we are overriding minmizer to just
    # return a value. Still need this so I don't try to call the QVM server.
    fake_qvm = Mock(spec=['wavefunction'])

    with patch("funcsigs.signature") as patch_signature:
        func_sigs_fake = MagicMock(spec=funcsigs.Signature)
        func_sigs_fake.parameters.return_value = \
            OrderedDict({
                'fun': funcsigs.Parameter.POSITIONAL_OR_KEYWORD,
                'x0': funcsigs.Parameter.POSITIONAL_OR_KEYWORD,
                'args': funcsigs.Parameter.POSITIONAL_OR_KEYWORD,
                'method': funcsigs.Parameter.POSITIONAL_OR_KEYWORD,
                'jac': funcsigs.Parameter.POSITIONAL_OR_KEYWORD,
                'hess': funcsigs.Parameter.POSITIONAL_OR_KEYWORD,
                'hessp': funcsigs.Parameter.POSITIONAL_OR_KEYWORD,
                'bounds': funcsigs.Parameter.POSITIONAL_OR_KEYWORD,
                'constraints': funcsigs.Parameter.POSITIONAL_OR_KEYWORD,
                'tol': funcsigs.Parameter.POSITIONAL_OR_KEYWORD,
                'callback': funcsigs.Parameter.POSITIONAL_OR_KEYWORD,
                'options': funcsigs.Parameter.POSITIONAL_OR_KEYWORD
            })

        patch_signature.return_value = func_sigs_fake
        inst = VQE(minimizer)

        t_result = inst.vqe_run(param_prog, hamiltonian, initial_param, qvm=fake_qvm)
        assert np.isclose(t_result.fun, 1.0)


def test_expectation():
    """expectation() routine can take a PauliSum operator on a matrix.  Check
    this functionality and the return of the correct scalar value"""
    X = np.array([[0, 1], [1, 0]])

    def RX_gate(phi):
        return expm(-1j*phi*X)

    def rotation_wavefunction(phi):
        state = np.array([[1], [0]])
        return RX_gate(phi).dot(state)

    prog = Program([RX(-2.5)(0)])
    hamiltonian = PauliTerm("Z", 0, 1.0)

    minimizer = MagicMock()
    fake_result = Mock()
    fake_result.fun = 1.0
    minimizer.return_value = fake_result

    fake_qvm = Mock(spec=['wavefunction', 'expectation', 'run'])
    fake_qvm.wavefunction.return_value = (Wavefunction(rotation_wavefunction(-2.5)), [0])
    fake_qvm.expectation.return_value = [0.28366219]
    # for testing expectation
    fake_qvm.run.return_value = [[0], [0]]

    inst = VQE(minimizer)
    energy = inst.expectation(prog, PauliSum([hamiltonian]), None, fake_qvm)
    assert np.isclose(energy, 0.28366219)

    hamiltonian = np.array([[1, 0], [0, -1]])
    energy = inst.expectation(prog, hamiltonian, None, fake_qvm)
    assert np.isclose(energy, 0.28366219)

    prog = Program(H(0))
    hamiltonian = PauliSum([PauliTerm('X', 0)])
    energy = inst.expectation(prog, hamiltonian, 2, fake_qvm)
    assert np.isclose(energy, 1.0)


def test_parity_even_p():
    state_index = 11
    marked_qubits = [0, 1]
    with pytest.raises(AssertionError):
        parity_even_p("", marked_qubits)
    assert parity_even_p(state_index, marked_qubits)


if __name__ == "__main__":
    test_expectation()
