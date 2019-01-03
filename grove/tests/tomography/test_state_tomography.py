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

import json
import os

import numpy as np
import pytest

from unittest.mock import MagicMock, patch
from pyquil.api import Job, QVMConnection

from grove.tomography.operator_utils import make_diagonal_povm
from grove.tomography.process_tomography import (TRACE_PRESERVING)
from grove.tomography.state_tomography import (DEFAULT_STATE_TOMO_SETTINGS,
                                               state_tomography_programs,
                                               do_state_tomography, StateTomography,
                                               UNIT_TRACE,
                                               POSITIVE)
from grove.tomography.tomography import (MAX_QUBITS_STATE_TOMO,
                                         default_channel_ops)
from grove.tomography.utils import (make_histogram,
                                    sample_bad_readout, basis_state_preps,
                                    estimate_assignment_probs, BELL_STATE_PROGRAM,
                                    BAD_2Q_READOUT, EPS, SEED, import_qutip, import_cvxpy)
from grove.tomography.operator_utils import POVM_PI_BASIS

qt = import_qutip()
cvxpy = import_cvxpy()

if not qt:
    pytest.skip("Qutip not installed, skipping tests", allow_module_level=True)

if not cvxpy:
    pytest.skip("CVXPY not installed, skipping tests", allow_module_level=True)


SHOTS_PATH = os.path.join(os.path.dirname(__file__), 'state_shots.json')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'state_results.json')
sample_bad_readout = MagicMock(sample_bad_readout)
sample_bad_readout.side_effect = [np.array(shots) for shots in json.load(open(SHOTS_PATH, 'r'))]

# these mocks are set up such that a single mock Job is returned by the QVMConnection's wait_for_job
# but calling job.result() returns a different value every time via the side_effect defined below
cxn = MagicMock(QVMConnection)
job = MagicMock(Job)
# repeat twice for run_and_measure and run
job.result.side_effect = json.load(open(RESULTS_PATH, 'r')) * 2
cxn.wait_for_job.return_value = job


def test_state_tomography():
    num_qubits = len(BELL_STATE_PROGRAM.get_qubits())
    dimension = 2 ** num_qubits
    tomo_seq = list(state_tomography_programs(BELL_STATE_PROGRAM))
    num_samples = 3000
    np.random.seed(SEED)
    qubits = [qubit for qubit in range(num_qubits)]
    tomo_preps = basis_state_preps(*qubits)
    state_prep_hists = []
    for i, tomo_prep in enumerate(tomo_preps):
        readout_result = sample_bad_readout(tomo_prep, num_samples, BAD_2Q_READOUT, cxn)
        state_prep_hists.append(make_histogram(readout_result, dimension))
    assignment_probs = estimate_assignment_probs(state_prep_hists)
    histograms = np.zeros((len(tomo_seq), dimension))
    for i, tomo_prog in enumerate(tomo_seq):
        readout_result = sample_bad_readout(tomo_prog, num_samples, BAD_2Q_READOUT, cxn)
        histograms[i] = make_histogram(readout_result, dimension)
    povm = make_diagonal_povm(POVM_PI_BASIS ** num_qubits, assignment_probs)
    channel_ops = list(default_channel_ops(num_qubits))
    for settings in [DEFAULT_STATE_TOMO_SETTINGS,
                     DEFAULT_STATE_TOMO_SETTINGS._replace(constraints={UNIT_TRACE, POSITIVE,
                                                                       TRACE_PRESERVING})]:
        state_tomo = StateTomography.estimate_from_ssr(histograms, povm, channel_ops, settings)
        amplitudes = np.array([1, 0, 0, 1]) / np.sqrt(2.)
        state = qt.Qobj(amplitudes, dims=[[2, 2], [1, 1]])
        rho_ideal = state * state.dag()
        assert abs(1 - state_tomo.fidelity(rho_ideal)) < EPS
    with patch("grove.tomography.utils.state_histogram"), patch(
            "grove.tomography.state_tomography.plt"):
        state_tomo.plot()


def test_do_state_tomography():
    nsamples = 3000
    qubits = list(range(MAX_QUBITS_STATE_TOMO + 1))
    # Test with too many qubits.
    with pytest.raises(ValueError):
        _ = do_state_tomography(BELL_STATE_PROGRAM, nsamples, cxn, qubits)
    state_tomo, assignment_probs, histograms = do_state_tomography(BELL_STATE_PROGRAM, nsamples,
                                                                   cxn)
    amplitudes = np.array([1, 0, 0, 1]) / np.sqrt(2.)
    state = qt.Qobj(amplitudes, dims=[[2, 2], [1, 1]])
    rho_ideal = state * state.dag()
    assert abs(1 - state_tomo.fidelity(rho_ideal)) < EPS
    for histogram in histograms:
        assert np.sum(histogram) == nsamples
    num_qubits = len(BELL_STATE_PROGRAM.get_qubits())
    assert np.isclose(assignment_probs, np.eye(2 ** num_qubits), atol=EPS).all()

    # ensure that use_run works.
    state_tomo2, _, _ = do_state_tomography(BELL_STATE_PROGRAM, nsamples, cxn, use_run=True)
    assert np.allclose(state_tomo2.rho_est.data.toarray(), state_tomo.rho_est.data.toarray(),
                       atol=1e-3)

