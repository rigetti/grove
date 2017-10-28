import itertools

import numpy as np
import pytest
import qutip as qt
from mock import patch, Mock


from grove.benchmarking.utils import (
    POVM_PI_BASIS, make_diagonal_povm, quil_to_operator,
    n_qubit_ground_state, make_histogram, sample_bad_readout, TOMOGRAPHY_GATES, basis_state_preps,
    estimate_assignment_probs)
from grove.benchmarking.tomography import (DEFAULT_STATE_TOMO_SETTINGS,
                                           DEFAULT_PROCESS_TOMO_SETTINGS, StateTomography,
                                           ProcessTomography, default_rotations,
                                           state_tomography_programs, process_tomography_programs,
                                           _SDP_SOLVER)
from pyquil.quil import Program
from pyquil.gates import CNOT, H, CZ
from referenceqvm.api import SyncConnection
from referenceqvm.gates import gate_matrix

BAD_1Q_READOUT = np.array([[.9, .15],
                           [.1, .85]])
BAD_2Q_READOUT = np.kron(BAD_1Q_READOUT, BAD_1Q_READOUT)


@pytest.fixture
def cxn():
    custom_gateset = gate_matrix.copy()
    custom_gateset.update({k: v.data.toarray() for k, v in TOMOGRAPHY_GATES.items()})
    return SyncConnection(gate_set=custom_gateset)


def ghz_circuit(N):
    return Program([H(0)] + [CNOT(j, j+1) for j in range(N-1)])


# methods for generating simulated counts
def probs_from_rho(rho, pi_basis):
    """
    Extract the probabilities of measuring some projective outcomes encapsulated in `pi_basis`.
    """
    return np.array([(pi_j * rho).tr().real for pi_j in pi_basis.ops])


def test_state_tomography(cxn):

    prog = Program([H(0),
                    H(1),
                    CZ(0, 1),
                    H(1)])

    nq = len(prog.extract_qubits())
    d = 2 ** nq

    tomo_channels = list(default_rotations(*range(nq)))

    tomo_seq = list(state_tomography_programs(prog))
    nsamples = 3000

    np.random.seed(234213)
    state_prep_hists = [make_histogram(sample_bad_readout(p, 3*nsamples, BAD_2Q_READOUT, cxn), d)
                        for p in basis_state_preps(*range(nq))]
    assignment_probs = estimate_assignment_probs(state_prep_hists)

    histograms = np.zeros((len(tomo_seq), d))

    for jj, p in enumerate(tomo_seq):
        histograms[jj] = make_histogram(sample_bad_readout(p, nsamples, BAD_2Q_READOUT, cxn), d)

    povm = make_diagonal_povm(POVM_PI_BASIS ** nq, assignment_probs)
    channel_ops = [quil_to_operator(tomo_q) for tomo_q in tomo_channels]

    for settings in [
        DEFAULT_STATE_TOMO_SETTINGS,
        DEFAULT_STATE_TOMO_SETTINGS._replace(constraints={'unit_trace', 'positive'})]:
        state_tomo = StateTomography.estimate_from_ssr(histograms, povm, channel_ops,
                                                       settings)


        cxn.run(prog)
        state = qt.Qobj(cxn.wf, dims=[[2, 2], [1, 1]])
        rho_ideal = state * state.dag()

        assert abs(1-state_tomo.fidelity(rho_ideal)) < 1e-2

    with patch("grove.benchmarking.utils.state_histogram"), \
         patch("grove.benchmarking.tomography.plt"):
        state_tomo.plot()


def test_SDP_SOLVER():
    assert _SDP_SOLVER.is_functional()


def test_process_tomography(cxn):

    prog = Program([H(1),
                    CZ(0, 1),
                    H(1)])

    nq = len(prog.extract_qubits())
    d = 2 ** nq

    tomo_channels = list(default_rotations(*range(nq)))

    tomo_seq = list(process_tomography_programs(prog))
    nsamples = 3000

    np.random.seed(2342134)
    state_prep_hists = [make_histogram(sample_bad_readout(p, 3*nsamples, BAD_2Q_READOUT, cxn), d)
                        for p in basis_state_preps(*range(nq))]
    assignment_probs = estimate_assignment_probs(state_prep_hists)

    histograms = np.zeros((len(tomo_seq), d))

    for jj, p in enumerate(tomo_seq):
        histograms[jj] = make_histogram(sample_bad_readout(p, nsamples, BAD_2Q_READOUT, cxn), d)

    histograms = histograms.reshape((len(tomo_channels), len(tomo_channels), d))

    povm = make_diagonal_povm(POVM_PI_BASIS ** nq, assignment_probs)
    channel_ops = [quil_to_operator(tomo_q) for tomo_q in tomo_channels]
    U_ideal = qt.cnot()
    for settings in [
        DEFAULT_PROCESS_TOMO_SETTINGS,
        DEFAULT_PROCESS_TOMO_SETTINGS._replace(constraints={'trace_preserving'}),
        DEFAULT_PROCESS_TOMO_SETTINGS._replace(constraints={'trace_preserving', 'cpositive'}),
    ]:

        process_tomo = ProcessTomography.estimate_from_ssr(histograms, povm, channel_ops, channel_ops,
                                                           settings)

        assert abs(1-process_tomo.avg_gate_fidelity(U_ideal)) < 1e-2

    assert abs(1-process_tomo.avg_gate_fidelity(qt.to_super(U_ideal))) < 1e-2

    with patch("grove.benchmarking.utils.plot_pauli_transfer_matrix"), \
         patch("grove.benchmarking.tomography.plt") as mplt:
        mplt.subplots.return_value = Mock(), Mock()
        process_tomo.plot()

