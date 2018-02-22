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

import logging

import numpy as np
import matplotlib.pyplot as plt
from pyquil.quilbase import Pragma
from scipy.sparse import csr_matrix, coo_matrix
from pyquil.quil import Program

import grove.tomography.operator_utils
from grove.tomography.tomography import TomographyBase, TomographySettings, DEFAULT_SOLVER_KWARGS
from grove.tomography import tomography
import grove.tomography.utils as ut
import grove.tomography.operator_utils as o_ut

_log = logging.getLogger(__name__)

qt = ut.import_qutip()
cvxpy = ut.import_cvxpy()


UNIT_TRACE = 'unit_trace'
POSITIVE = 'positive'
DEFAULT_STATE_TOMO_SETTINGS = TomographySettings(
    constraints={UNIT_TRACE},
    solver_kwargs=DEFAULT_SOLVER_KWARGS
)


def _prepare_c_jk_m(readout_povm, pauli_basis, channel_ops):
    """
    Prepare the coefficient matrix for state tomography. This function uses sparse matrices
    for much greater efficiency.
    The coefficient matrix is defined as:

    .. math::

            C_{(jk)m} = \tr{\Pi_{s_j} \Lambda_k(P_m)} = \sum_{r}\pi_{jr}(\mathcal{R}_{k})_{rm}

    where :math:`\Lambda_k(\cdot)` is the quantum map corresponding to the k-th pre-measurement
    channel, i.e., :math:`\Lambda_k(\rho) = E_k \rho E_k^\dagger` where :math:`E_k` is the k-th
    channel operator. This map can also be represented via its transfer matrix
    :math:`\mathcal{R}_{k}`. In that case one also requires the overlap between the (generalized)
    Pauli basis ops and the projection operators
    :math:`\pi_{jl}:=\sbraket{\Pi_j}{P_l} = \tr{\Pi_j P_l}`.
    See the grove documentation on tomography for detailed information.

    :param DiagonalPOVM readout_povm: The POVM corresponding to the readout plus classifier.
    :param OperatorBasis pauli_basis: The (generalized) Pauli basis employed in the estimation.
    :param list channel_ops: The pre-measurement channel operators as `qutip.Qobj`
    :return: The coefficient matrix necessary to set up the binomial state tomography problem.
    :rtype: scipy.sparse.csr_matrix
    """
    channel_transfer_matrices = [pauli_basis.transfer_matrix(qt.to_super(ek)) for ek in channel_ops]

    # This bit could be more efficient but does not run super long and is thus preserved for
    #  readability.
    pi_jr = csr_matrix(
        [pauli_basis.project_op(n_j).toarray().ravel()
         for n_j in readout_povm.ops])

    # Dict used for constructing our sparse matrix, keys are tuples (row_index, col_index), values
    # are the non-zero elements of the final matrix.
    c_jk_m_elms = {}

    # This explicitly exploits the sparsity of all operators involved
    for k in range(len(channel_ops)):
        pi_jr__rk_rm = (pi_jr * channel_transfer_matrices[k]).tocoo()
        for (j, m, val) in ut.izip(pi_jr__rk_rm.row, pi_jr__rk_rm.col, pi_jr__rk_rm.data):
            # The multi-index (j,k) is enumerated in column-major ordering (like Fortran arrays)
            c_jk_m_elms[(j + k * readout_povm.pi_basis.dim, m)] = val.real

    # create sparse matrix from COO-format (see scipy.sparse docs)
    _keys, _values = ut.izip(*c_jk_m_elms.items())
    _rows, _cols = ut.izip(*_keys)
    c_jk_m = coo_matrix((list(_values), (list(_rows), list(_cols))),
                        shape=(readout_povm.pi_basis.dim * len(channel_ops),
                               pauli_basis.dim)).tocsr()
    return c_jk_m


class StateTomography(TomographyBase):
    """
    A StateTomography object encapsulates the result of quantum state estimation from tomographic
    data. It provides convenience functions for visualization and computing state fidelities.
    """
    __tomography_type__ = "STATE"

    @staticmethod
    def estimate_from_ssr(histograms, readout_povm, channel_ops, settings):
        """
        Estimate a density matrix from single shot histograms obtained by measuring bitstrings in
        the Z-eigenbasis after application of given channel operators.

        :param numpy.ndarray histograms: The single shot histograms, `shape=(n_channels, dim)`.
        :param DiagognalPOVM readout_povm: The POVM corresponding to the readout plus classifier.
        :param list channel_ops: The tomography measurement channels as `qutip.Qobj`'s.
        :param TomographySettings settings: The solver and estimation settings.
        :return: The generated StateTomography object.
        :rtype: StateTomography
        """
        nqc = len(channel_ops[0].dims[0])
        pauli_basis = grove.tomography.operator_utils.PAULI_BASIS ** nqc
        pi_basis = readout_povm.pi_basis

        if not histograms.shape[1] == pi_basis.dim:  # pragma no coverage
            raise ValueError("Currently tomography is only implemented for two-level systems.")

        # prepare the log-likelihood function parameters, see documentation
        n_kj = np.asarray(histograms)
        c_jk_m = _prepare_c_jk_m(readout_povm, pauli_basis, channel_ops)
        rho_m = cvxpy.Variable(pauli_basis.dim)

        p_jk = c_jk_m * rho_m
        obj = -n_kj.ravel() * cvxpy.log(p_jk)

        p_jk_mat = cvxpy.reshape(p_jk, pi_basis.dim, len(channel_ops))  # cvxpy has col-major order

        # Default constraints:
        # MLE must describe valid probability distribution
        # i.e., for each k, p_jk must sum to one and be element-wise non-negative:
        # 1. \sum_j p_jk == 1  for all k
        # 2. p_jk >= 0         for all j, k
        # where p_jk = \sum_m c_jk_m rho_m
        constraints = [
            p_jk >= 0,
            np.matrix(np.ones((1, pi_basis.dim))) * p_jk_mat == 1,
        ]

        rho_m_real_imag = sum((rm * o_ut.to_realimag(Pm)
                               for (rm, Pm) in ut.izip(rho_m, pauli_basis.ops)), 0)

        if POSITIVE in settings.constraints:
            if tomography._SDP_SOLVER.is_functional():
                constraints.append(rho_m_real_imag >> 0)
            else:  # pragma no coverage
                _log.warning("No convex solver capable of semi-definite problems installed.\n"
                             "Dropping the positivity constraint on the density matrix.")

        if UNIT_TRACE in settings.constraints:
            # this assumes that the first element of the Pauli basis is always proportional to
            # the identity
            constraints.append(rho_m[0, 0] == 1. / pauli_basis.ops[0].tr().real)

        prob = cvxpy.Problem(cvxpy.Minimize(obj), constraints)

        _log.info("Starting convex solver")
        prob.solve(solver=tomography.SOLVER, **settings.solver_kwargs)
        if prob.status != cvxpy.OPTIMAL:  # pragma no coverage
            _log.warning("Problem did not converge to optimal solution. "
                         "Solver settings: {}".format(settings.solver_kwargs))

        return StateTomography(np.array(rho_m.value).ravel(), pauli_basis, settings)

    def __init__(self, rho_coeffs, pauli_basis, settings):
        """
        Construct a StateTomography to encapsulate the result of estimating the quantum state from
        a quantum tomography measurement.

        :param numpy.ndarray r_est: The estimated quantum state represented in a given (generalized)
        Pauli basis.
        :param OperatorBasis pauli_basis: The employed (generalized) Pauli basis.
        :param TomographySettings settings: The settings used to estimate the state.
        """
        self.rho_coeffs = rho_coeffs
        self.pauli_basis = pauli_basis
        self.rho_est = sum((r_m * p_m for r_m, p_m in ut.izip(rho_coeffs, pauli_basis.ops)))
        self.settings = settings

    def fidelity(self, other):
        """
        Compute the quantum state fidelity of the estimated state with another state.

        :param qutip.Qobj other: The other quantum state.
        :return: The fidelity, a real number between 0 and 1.
        :rtype: float
        """
        return qt.fidelity(self.rho_est, other)

    def plot_state_histogram(self, ax):
        """
        Visualize the complex matrix elements of the estimated state.

        :param matplotlib.Axes ax: A matplotlib Axes object to plot into.
        """
        title = "Estimated state"
        nqc = int(round(np.log2(self.rho_est.data.shape[0])))
        labels = ut.basis_labels(nqc)
        return ut.state_histogram(self.rho_est, ax, title)

    def plot(self):
        """
        Visualize the state.

        :return: The generated figure.
        :rtype: matplotlib.Figure
        """
        width = 10
        # The pleasing golden ratio.
        height = width / 1.618
        f = plt.figure(figsize=(width, height))
        ax = f.add_subplot(111, projection="3d")

        self.plot_state_histogram(ax)
        return f


def state_tomography_programs(state_prep, qubits=None,
                              rotation_generator=tomography.default_rotations):
    """
    Yield tomographic sequences that prepare a state with Quil program `state_prep` and then append
    tomographic rotations on the specified `qubits`. If `qubits is None`, it assumes all qubits in
    the program should be tomographically rotated.

    :param Program state_prep: The program to prepare the state to be tomographed.
    :param list|NoneType qubits: A list of Qubits or Numbers, to perform the tomography on. If
    `None`, performs it on all in state_prep.
    :param generator rotation_generator: A generator that yields tomography rotations to perform.
    :return: Program for state tomography.
    :rtype: Program
    """
    if qubits is None:
        qubits = state_prep.get_qubits()
    for tomography_program in rotation_generator(*qubits):
        state_tomography_program = Program(Pragma("PRESERVE_BLOCK"))
        state_tomography_program.inst(state_prep)
        state_tomography_program.inst(tomography_program)
        state_tomography_program.inst(Pragma("END_PRESERVE_BLOCK"))
        yield state_tomography_program


def do_state_tomography(preparation_program, nsamples, cxn, qubits=None, use_run=False):
    """
    Method to perform both a QPU and QVM state tomography, and use the latter as
    as reference to calculate the fidelity of the former.

    :param Program preparation_program: Program to execute.
    :param int nsamples: Number of samples to take for the program.
    :param QVMConnection|QPUConnection cxn: Connection on which to run the program.
    :param list qubits: List of qubits for the program.
    to use in the tomography analysis.
    :param bool use_run: If ``True``, use append measurements on all qubits and use ``cxn.run``
        instead of ``cxn.run_and_measure``.
    :return: The state tomogram.
    :rtype: StateTomography
    """
    return tomography._do_tomography(preparation_program, nsamples, cxn, qubits,
                                     tomography.MAX_QUBITS_STATE_TOMO,
                                     StateTomography, state_tomography_programs,
                                     DEFAULT_STATE_TOMO_SETTINGS, use_run=use_run)
