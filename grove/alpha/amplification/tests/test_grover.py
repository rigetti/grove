
if __name__ == "__main__":
    from pyquil.api import SyncConnection
    import sys

    try:
        target = sys.argv[1]
    except IndexError:
        raise ValueError("Enter a target bitstring for Grover's Algorithm.")

    grover_program = pq.Program()
    qubits = range(len(target))
    oracle = basis_selector_oracle(target, qubits)
    grover_program += grover(oracle, qubits)

    cxn = SyncConnection()
    mem = cxn.run_and_measure(grover_program, qubits)
    print(mem)
