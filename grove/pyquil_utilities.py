"""Module containing utilities that will eventually be implemented in pyquil."""

from pyquil.quilbase import ACTION_INSTANTIATE_QUBIT
from pyquil.quilbase import Gate, Qubit
from pyquil.resource_manager import merge_resource_managers, ResourceManager


def prog_equality(prog1, prog2):
    """Tests to see if two programs are equal based on their out methods

    :param prog1: The first Program to compare
    :param prog2: The second program to compare
    """
    return prog1.out() == prog2.out()


def synthesize_programs(*progs):
    """Synthesizes programs together, paying attention to shared qubits. This breaks if synthesize
     is called after, very fragile.

    :param progs: List of Programs.
    :return: None
    :rtype: NoneType"""
    def get_allocated_qubits(*progs):
        qubits_to_progs = {}
        for prog in progs:
            for inst in prog:
                if inst[0] == ACTION_INSTANTIATE_QUBIT:
                    qubits_to_progs[inst[1]] = [prog]
        return qubits_to_progs
    qubits_to_progs = get_allocated_qubits(*progs)
    for prog in progs:
        for inst in prog:
            if isinstance(inst[1], Gate):
                for arg in inst[1].arguments:
                    if isinstance(arg, Qubit):
                        qubits_to_progs[arg].append(prog)

    equiv_classes = list(qubits_to_progs.values())
    repeat = True
    while repeat:
        repeat = False
        for i, equiv_class in enumerate(equiv_classes):
            for j, other_equiv_class in enumerate(equiv_classes):
                if j <= i:
                    continue
                if set(equiv_class).intersection(set(other_equiv_class)):
                    equiv_classes[i] = set(equiv_class).union(other_equiv_class)
                    equiv_classes.remove(other_equiv_class)
                    repeat = True

    for equiv_class in equiv_classes:
        rm = ResourceManager()
        for prog in equiv_class:
            rm = merge_resource_managers(rm, prog.resource_manager)
        for prog in equiv_class:
            prog.resource_manager = rm
            prog.actions = [inst for inst in prog if inst[0] != ACTION_INSTANTIATE_QUBIT]
        for qubit in rm.live_qubits:
            for prog in progs:
                prog.actions = [(ACTION_INSTANTIATE_QUBIT, qubit)] + prog.actions
