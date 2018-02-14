import logging
from time import sleep

import numpy as np
from pyquil.api import QVMConnection, QPUConnection
from pyquil.api.errors import DeviceRetuningError
from pyquil.device import ISA
from pyquil.gates import I, RX, CZ
from pyquil.noise import NoiseModel, KrausModel
from pyquil.quil import Program

import grove.tomography.operator_utils as o_ut
import grove.tomography.utils as ut
from grove.tomography.process_tomography import (process_tomography_programs, ProcessTomography,
                                                 DEFAULT_PROCESS_TOMO_SETTINGS, COMPLETELY_POSITIVE,
                                                 TRACE_PRESERVING)
from grove.tomography.tomography import default_channel_ops
from grove.tomography.utils import run_in_parallel

qt = ut.import_qutip()

_log = logging.getLogger(__name__)
#
# # Qubit 3 not tunable on Acorn.
# ACORN_QUBITS = [q for q in range(20) if q not in [3]]
#
# ACORN_GROUPS = [
#     [(0, 5), (10, 15), (1, 6), (11, 16), (2, 7), (12, 17), (13, 18), (4, 9), (14, 19)],
#     [(0, 6), (10, 16), (1, 7), (11, 17), (2, 8), (12, 18), (13, 19)],
#     [(5, 10), (6, 11), (7, 12), (8, 13), (9, 14)]
# ]
#
# AGAVE_QUBITS = [q for q in range(8) if q not in []]
#
#
# AGAVE_GROUPS = [
#     [(0, 1), (2, 3), (4, 5), (6, 7)],
#     [(0, 7), (1, 2), (3, 4), (5, 6)],
# ]
#
if qt:
    # generate ideal targets
    QCZ = qt.cphase(np.pi)
    QRX90 = (-np.pi * .25j * qt.sigmax()).expm()
    QI = qt.qeye(2)
else:  # pragma no coverage
    QCZ = None
    QRX90 = None
    QI = None

QPU_TOMO_SETTINGS = DEFAULT_PROCESS_TOMO_SETTINGS._replace(
 constraints={COMPLETELY_POSITIVE, TRACE_PRESERVING}
)


def estimate(isa, nsamples, cxn, edge_groups, shuffle=True, max_retries=100,
             settings=QPU_TOMO_SETTINGS, retune_sleep=60.):
    """
    Generate a ``NoiseModel`` for a QPU specified by its ``isa`` by sampling from a connection
    ``cxn``.

    .. note::

        Currently we only support 'CZ' type edges and 'Xhalves' type qubits.

    :param ISA isa: The instruction set architecture of the QPU.
    :param int nsamples: The number of samples to use in each measurement.
    :param QPUConnection|QVMConnection cxn: The connection to the virtual or actual quantum
    processor.
    :param bool shuffle: If True (default), the tomographic sequences between different qubit
        groups are randomized in order to avoid introducing bias, e.g., by having identity gate
        pre/post rotations occur on all qubits simultaneously.
    :param Sequence[List[Tuple]] edge_groups: List of lists of tuples ``[[(i, j), ...], ...]``
        such that ``i`` and ``j`` denote the targets of a 2q gate.  Gates within an edge group are
        executed simultaneously during noise model characterization.
    :param int max_retries: How often to restart tomographies that are interrupted by the QPU
        retuning.
    :param float retune_sleep: How long to wait before the next retry after a DeviceRetuningError.
    :param TomographySettings settings: The settings for the MLE tomography.
    :return: The estimated NoiseModel
    """

    qubits = []
    for q in isa.qubits:
        if q.dead:
            continue
        if q.type != "Xhalves":
            raise ValueError("Do not currently support qubit types other than 'Xhalves'.")
        qubits.append(q.id)

    all_edges = set()
    for e in isa.edges:
        if e.dead:
            continue
        if e.type != "CZ":
            raise ValueError("Do not currently support edges other than 'CZ'.")
        all_edges.add(tuple(sorted(e)))

    edge_group_sets = [{tuple(sorted(e)) for e in group} for group in edge_groups]
    given_edges = reduce(lambda s1, s2: s1.union(s2), edge_group_sets, set())
    missing_edges = all_edges - given_edges

    if missing_edges:
        raise ValueError("Incomplete specification of edge_groups. Missing edges: {}".format(
            missing_edges)
        )

    def _execute_tomos(gates, targets):
        trials = 0
        while True:
            try:
                return parallel_process_tomographies(gates, targets, nsamples, cxn, shuffle=shuffle,
                                                     settings=settings)
            except DeviceRetuningError as e:
                _log.info("Device retuning, sleeping for 60 seconds.")
                sleep(retune_sleep)
                trials += 1
                if trials >= max_retries:  # pragma no coverage
                    _log.error("Did not complete tomography of:\n\t {}".format(
                        "\n\t".join(map(str, [g(ts) for g, ts in zip(gates, targets)]))))
                    raise e

    # Identities
    i_tomos, _, i_assignment_probs  = _execute_tomos([I] * len(qubits), qubits)
    i_kraus = [KrausModel("I", (), q, t.to_kraus(), t.avg_gate_fidelity(QI))
               for q, t in zip(isa.qubits, i_tomos)]

    # RX(pi/2)
    rx90_tomos, _, rx90_assignment_probs = _execute_tomos([RX(np.pi/2)] * len(qubits), qubits)
    rx90_kraus = [KrausModel("RX", (np.pi/2,), q, t.to_kraus(), t.avg_gate_fidelity(QRX90))
                  for q, t in zip(isa.qubits, rx90_tomos)]

    # combine assignment probs through averaging
    assignment_probs = {q: .5 * (p1 + p2)
                        for q, p1, p2 in zip(isa.qubits, i_assignment_probs, rx90_assignment_probs)}

    cz_tomo_dict = {}
    for group in edge_groups:
        tomos = _execute_tomos([CZ] * len(group), group)[0]
        for edge, tomo in zip(group, tomos):
            cz_tomo_dict[edge] = tomo

    cz_tomos = [cz_tomo_dict[e] for e in isa.edges]
    cz_kraus = [KrausModel("CZ", (), e, t.to_kraus(), t.avg_gate_fidelity(QCZ))
                for e, t in zip(isa.edges, cz_tomos)]

    return NoiseModel(i_kraus + rx90_kraus + cz_kraus, assignment_probs)


def parallel_sample_assignment_probs(qubit_groups, nsamples, cxn, shuffle=True):
    """
    Sample the assignment probabilities of qubits using nsamples per measurement, and then compute
    the estimated assignment probability matrix. See the docstring for estimate_assignment_probs for
    more information.

    :param Sequence[tuple|list] qubit_groups: The qubit targets for each parallel tomography. All
        groups must contain the same number of qubits and they must be disjoint.
    :param int nsamples: The number of samples to use in each measurement.
    :param QPUConnection|QVMConnection cxn: The connection to the virtual or actual quantum
        processor.
    :param bool shuffle: If True (default), the tomographic sequences between different qubit
        groups are randomized in order to avoid introducing bias, e.g., by having identity gate
        pre/post rotations occur on all qubits simultaneously.
    :return: A list of assignment probability matrices for each qubit group.
    :rtype: List[numpy.ndarray]
    """
    n_qubits_per_group = [len(c) for c in qubit_groups]
    if not all(n_qubits == n_qubits_per_group[0] for n_qubits in n_qubits_per_group):
        raise ValueError("Can only handle groups of equal size.")

    programs = [list(ut.basis_state_preps(*sorted(c))) for c in qubit_groups]
    all_hists = run_in_parallel(programs, nsamples, cxn, shuffle=shuffle)
    return [ut.estimate_assignment_probs(hists) for hists in all_hists]


def parallel_process_tomographies(gates, targets, nsamples, cxn, shuffle=True,
                                  settings=DEFAULT_PROCESS_TOMO_SETTINGS):
    """
    Execute process tomographies of the same pyquil ``gate`` on several disjoint ``qubit_groups``
    simultaneously.

    :param Sequence[Gate] gate: The gates under investigation, one for each element of qubit_groups.
        (these can be any function that takes qubit indices as argument and outputs a Quil program.)
    :param Sequence[tuple|list] targets: The qubit targets for each parallel tomography. All
        groups must contain the same number of qubits and they must be disjoint.
    :param int nsamples: The number of samples
    :param QPUConnection|QVMConnection cxn: The connection to the virtual or actual quantum
        processor.
    :param bool shuffle: If True (default), the tomographic sequences between different qubit
        groups are randomized in order to avoid introducing bias, e.g., by having identity gate
        pre/post rotations occur on all qubits simultaneously.
    :param TomographySettings settings: The settings for the MLE tomography.
    :return: A tuple ``(qpts, qpt_histograms, assignment_probs)`` where ``qpts`` is a list of
        ``ProcessTomography`` objects, one for each qubit group, ``qpt_histograms`` is a sequence of
        2d-arrays containing the individual QPT measurement histograms for each group, and
        ``assignment_probs`` is a list of assignment probability matrices.
    """

    n_qubits_per_group = [len(c) for c in targets]
    n_qubits = n_qubits_per_group[0]
    if not all(_n_qubits == n_qubits for _n_qubits in n_qubits_per_group):
        raise ValueError("Can only handle groups of equal size.")

    # generate assignment probs
    assignment_probs_per_group = parallel_sample_assignment_probs(targets, nsamples, cxn,
                                                                  shuffle=shuffle)
    programs = []
    for g, c in zip(gates, targets):
        programs.append(list(process_tomography_programs(Program(g(*sorted(c))), sorted(c))))

    tomo_hists_per_group = run_in_parallel(programs, nsamples, cxn, shuffle=shuffle)
    results_per_group = []

    # same for all groups
    channel_ops = list(default_channel_ops(n_qubits))

    for assignment_probs, hists in zip(assignment_probs_per_group, tomo_hists_per_group):
        povm = o_ut.make_diagonal_povm(o_ut.POVM_PI_BASIS ** n_qubits, assignment_probs)
        hists = hists.reshape((len(channel_ops), len(channel_ops), 2 ** n_qubits))
        results_per_group.append(ProcessTomography.estimate_from_ssr(hists, povm, channel_ops,
                                                                     channel_ops, settings))

    return results_per_group, tomo_hists_per_group, assignment_probs_per_group
