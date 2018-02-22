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
from scipy.sparse import (vstack as spvstack, csr_matrix, kron as spkron)
from pyquil.quil import Program

import grove.tomography.operator_utils
from grove.tomography.tomography import TomographyBase, TomographySettings, DEFAULT_SOLVER_KWARGS
import grove.tomography.state_tomography as state_tomography
from grove.tomography import tomography
import grove.tomography.utils as ut
import grove.tomography.operator_utils as o_ut

_log = logging.getLogger(__name__)


qt = ut.import_qutip()
cvxpy = ut.import_cvxpy()


TRACE_PRESERVING = 'trace_preserving'
COMPLETELY_POSITIVE = 'cpositive'
DEFAULT_PROCESS_TOMO_SETTINGS = TomographySettings(
    constraints=set([TRACE_PRESERVING]),
    solver_kwargs=DEFAULT_SOLVER_KWARGS
)


def _prepare_b_jkl_mn(readout_povm, pauli_basis, pre_channel_ops, post_channel_ops, rho0):
    """
    Prepare the coefficient matrix for process tomography. This function uses sparse matrices
    for much greater efficiency. The coefficient matrix is defined as:

    .. math::

            B_{(jkl)(mn)}=\sum_{r,q}\pi_{jr}(\mathcal{R}_{k})_{rm} (\mathcal{R}_{l})_{nq} (\rho_0)_q

    where :math:`\mathcal{R}_{k}` is the transfer matrix of the quantum map corresponding to the
    k-th pre-measurement channel, while :math:`\mathcal{R}_{l}` is the transfer matrix of the l-th
    state preparation process. We also require the overlap
    between the (generalized) Pauli basis ops and the projection operators
    :math:`\pi_{jl}:=\sbraket{\Pi_j}{P_l} = \tr{\Pi_j P_l}`.

    See the grove documentation on tomography for detailed information.

    :param DiagonalPOVM readout_povm: The POVM corresponding to the readout plus classifier.
    :param OperatorBasis pauli_basis: The (generalized) Pauli basis employed in the estimation.
    :param list pre_channel_ops: The state preparation channel operators as `qutip.Qobj`
    :param list post_channel_ops: The pre-measurement (post circuit) channel operators as `qutip.Qobj`
    :param qutip.Qobj rho0: The initial state as a density matrix.
    :return: The coefficient matrix necessary to set up the binomial state tomography problem.
    :rtype: scipy.sparse.csr_matrix
    """
    c_jk_m = state_tomography._prepare_c_jk_m(readout_povm, pauli_basis, post_channel_ops)
    pre_channel_transfer_matrices = [pauli_basis.transfer_matrix(qt.to_super(ek))
                                     for ek in pre_channel_ops]
    rho0_q = pauli_basis.project_op(rho0)

    # These next lines hide some very serious (sparse-)matrix index magic,
    # basically we exploit the same index math as in `qutip.sprepost()`
    # i.e., if a matrix X is linearly mapped `X -> A.dot(X).dot(B)`
    # then this can be rewritten as
    #           `np.kron(B.T, A).dot(X.T.ravel()).reshape((B.shape[1], A.shape[0])).T`
    # The extra matrix transpose operations are necessary because numpy by default
    # uses row-major storage, whereas these operations are conventionally defined for column-major
    # storage.
    d_ln = spvstack([(rlnq * rho0_q).T for rlnq in pre_channel_transfer_matrices]).tocoo()
    b_jkl_mn = spkron(d_ln, c_jk_m).real
    return b_jkl_mn


class ProcessTomography(TomographyBase):
    """
    A ProcessTomography object encapsulates the result of quantum process estimation from
    tomographic data. It provides convenience functions for visualization, computing process
    fidelities and inter-conversion between different numerical representation of quantum processes.
    """
    __tomography_type__ = "PROCESS"

    @staticmethod
    def estimate_from_ssr(histograms, readout_povm, pre_channel_ops, post_channel_ops, settings):
        """
        Estimate a quantum process from single shot histograms obtained by preparing specific input
        states and measuring bitstrings in the Z-eigenbasis after application of given channel
        operators.

        :param numpy.ndarray histograms: The single shot histograms.
        :param DiagonalPOVM readout_povm: The POVM corresponding to readout plus classifier.
        :param list pre_channel_ops: The input state preparation channels as `qutip.Qobj`'s.
        :param list post_channel_ops: The tomography post-process channels as `qutip.Qobj`'s.
        :param TomographySettings settings: The solver and estimation settings.
        :return: The ProcessTomography object and results from the the given data.
        :rtype: ProcessTomography
        """
        nqc = len(pre_channel_ops[0].dims[0])
        pauli_basis = grove.tomography.operator_utils.PAULI_BASIS ** nqc
        pi_basis = readout_povm.pi_basis

        if not histograms.shape[-1] == pi_basis.dim:  # pragma no coverage
            raise ValueError("Currently tomography is only implemented for two-level systems")

        rho0 = grove.tomography.operator_utils.n_qubit_ground_state(nqc)

        n_lkj = np.asarray(histograms)

        b_jkl_mn = _prepare_b_jkl_mn(readout_povm, pauli_basis, pre_channel_ops,
                                     post_channel_ops, rho0)

        r_mn = cvxpy.Variable(pauli_basis.dim ** 2)
        p_jkl = b_jkl_mn.real * r_mn
        obj = -np.matrix(n_lkj.ravel()) * cvxpy.log(p_jkl)

        # cvxpy has col-major order and we collapse k and l onto single dimension
        p_jkl_mat = cvxpy.reshape(p_jkl, pi_basis.dim, len(pre_channel_ops) * len(post_channel_ops))

        # Default constraints:
        # MLE must describe valid probability distribution
        # i.e., for each k and l, p_jkl must sum to one and be element-wise non-negative:
        # 1. \sum_j p_jkl == 1  for all k, l
        # 2. p_jkl >= 0         for all j, k, l
        # where p_jkl = \sum_m b_jkl_mn r_mn
        constraints = [p_jkl >= 0,
                       np.matrix(np.ones((1, pi_basis.dim))) * p_jkl_mat == 1]
        r_mn_mat = cvxpy.reshape(r_mn, pauli_basis.dim, pauli_basis.dim)
        super_pauli_basis = pauli_basis.super_basis()
        choi_real_imag = sum((r_mn_mat[jj, kk] * o_ut.to_realimag(
            super_pauli_basis.ops[jj + kk * pauli_basis.dim])
                              for jj in range(pauli_basis.dim)
                              for kk in range(pauli_basis.dim)), 0)

        if COMPLETELY_POSITIVE in settings.constraints:
            if tomography._SDP_SOLVER.is_functional():
                constraints.append(choi_real_imag >> 0)
            else:  # pragma no coverage
                _log.warning("No convex solver capable of semi-definite problems installed.\n"
                             "Dropping the complete positivity constraint on the process")

        if TRACE_PRESERVING in settings.constraints:
            constraints.append(r_mn_mat[0, 0] == 1)
            constraints.append(r_mn_mat[0, 1:] == 0)

        prob = cvxpy.Problem(cvxpy.Minimize(obj), constraints)
        _ = prob.solve(solver=tomography.SOLVER, **settings.solver_kwargs)

        r_mn_est = r_mn.value.reshape((pauli_basis.dim, pauli_basis.dim)).transpose()
        return ProcessTomography(r_mn_est, pauli_basis, settings)

    def __init__(self, r_est, pauli_basis, settings):
        """
        Construct a ProcessTomography to encapsulate the result of estimating a quantum process
        from a quantum tomography measurement.

        :param numpy.ndarray r_est: The estimated quantum process represented as a Pauli transfer
         matrix.
        :param OperatorBasis pauli_basis: The employed (generalized) Pauli basis.
        :param TomographySettings settings: The settings used to estimate the process.
        """
        self.r_est = r_est
        self.sop = pauli_basis.super_from_tm(r_est)
        self.pauli_basis = pauli_basis
        self.settings = settings

    def process_fidelity(self, reference_unitary):
        """
        Compute the quantum process fidelity of the estimated state with respect to a unitary
        process. For non-sparse reference_unitary, this implementation this will be expensive in
        higher dimensions.

        :param (qutip.Qobj|matrix-like) reference_unitary: A unitary operator that induces a process
         as ``rho -> other*rho*other.dag()``, can also be a superoperator or Pauli-transfer matrix.
        :return: The process fidelity, a real number between 0 and 1.
        :rtype: float
        """
        if isinstance(reference_unitary, qt.Qobj):
            if not reference_unitary.issuper or reference_unitary.superrep != "super":
                sother = qt.to_super(reference_unitary)
            else:
                sother = reference_unitary
            tm_other = self.pauli_basis.transfer_matrix(sother)
        else:
            tm_other = csr_matrix(reference_unitary)
        dimension = self.pauli_basis.ops[0].shape[0]
        return np.trace(tm_other.T * self.r_est).real / dimension ** 2

    def avg_gate_fidelity(self, reference_unitary):
        """
        Compute the average gate fidelity of the estimated process with respect to a unitary
         process. See `Chow et al., 2012, <https://doi.org/10.1103/PhysRevLett.109.060501>`_

        :param (qutip.Qobj|matrix-like) reference_unitary: A unitary operator that induces a process
         as `rho -> other*rho*other.dag()`, alternatively a superoperator or Pauli-transfer matrix.
        :return: The average gate fidelity, a real number between 1/(d+1) and 1, where d is the
        Hilbert space dimension.
        :rtype: float
        """
        process_fidelity = self.process_fidelity(reference_unitary)
        dimension = self.pauli_basis.ops[0].shape[0]
        return (dimension * process_fidelity + 1.0) / (dimension + 1.0)

    def to_super(self):
        """
        Compute the standard superoperator representation of the estimated process.

        :return: The process as a superoperator.
        :rytpe: qutip.Qobj
        """
        return self.sop

    def to_choi(self):
        """
        Compute the choi matrix representation of the estimated process.

        :return: The process as a choi-matrix.
        :rytpe: qutip.Qobj
        """
        return qt.to_choi(self.sop)

    def to_chi(self):
        """
        Compute the chi process matrix representation of the estimated process.

        :return: The process as a chi-matrix.
        :rytpe: qutip.Qobj
        """
        return qt.to_chi(self.sop)

    def to_kraus(self):
        """
        Compute the Kraus operator representation of the estimated process.

        :return: The process as a list of Kraus operators.
        :rytpe: List[np.array]
        """
        return [k.data.toarray() for k in qt.to_kraus(self.sop)]

    def plot_pauli_transfer_matrix(self, ax):
        """
        Plot the elements of the Pauli transfer matrix.

        :param matplotlib.Axes ax: A matplotlib Axes object to plot into.
        """
        title = "Estimated process"
        ut.plot_pauli_transfer_matrix(self.r_est, ax, self.pauli_basis.labels, title)

    def plot(self):
        """
        Visualize the process.

        :return: The generated figure.
        :rtype: matplotlib.Figure
        """
        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 8))
        self.plot_pauli_transfer_matrix(ax1)
        return fig


def process_tomography_programs(process, qubits=None,
                                pre_rotation_generator=tomography.default_rotations,
                                post_rotation_generator=tomography.default_rotations):
    """
    Generator that yields tomographic sequences that wrap a process encoded by a QUIL program `proc`
    in tomographic rotations on the specified `qubits`.

    If `qubits is None`, it assumes all qubits in the program should be
    tomographically rotated.

    :param Program process: A Quil program
    :param list|NoneType qubits: The specific qubits for which to generate the tomographic sequences
    :param pre_rotation_generator: A generator that yields tomographic pre-rotations to perform.
    :param post_rotation_generator: A generator that yields tomographic post-rotations to perform.
    :return: Program for process tomography.
    :rtype: Program
    """
    if qubits is None:
        qubits = process.get_qubits()
    for tomographic_pre_rotation in pre_rotation_generator(*qubits):
        for tomography_post_rotation in post_rotation_generator(*qubits):
            process_tomography_program = Program(Pragma("PRESERVE_BLOCK"))
            process_tomography_program.inst(tomographic_pre_rotation)
            process_tomography_program.inst(process)
            process_tomography_program.inst(tomography_post_rotation)
            process_tomography_program.inst(Pragma("END_PRESERVE_BLOCK"))

            yield process_tomography_program


def do_process_tomography(process, nsamples, cxn, qubits=None, use_run=False):
    """
    Method to perform a process tomography.

    :param Program process: Process to execute.
    :param int nsamples: Number of samples to take for the program.
    :param QVMConnection|QPUConnection cxn: Connection on which to run the program.
    :param list qubits: List of qubits for the program.
    to use in the tomography analysis.
    :param bool use_run: If ``True``, use append measurements on all qubits and use ``cxn.run``
        instead of ``cxn.run_and_measure``.
    :return: The process tomogram
    :rtype: ProcessTomography
    """
    return tomography._do_tomography(process, nsamples, cxn, qubits,
                                     tomography.MAX_QUBITS_PROCESS_TOMO,
                                     ProcessTomography, process_tomography_programs,
                                     DEFAULT_PROCESS_TOMO_SETTINGS, use_run=use_run)
