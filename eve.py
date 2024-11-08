from qiskit import *
import random

BATCHSIZE = 100

randBases = []

for i in range(BATCHSIZE):
    randBases.append(random.randint(0, 1)) # 0 is Z base and 1 is X base

def eve():
    qubits = []
    for i in range(BATCHSIZE):
        q = QuantumRegister(1)
        c = ClassicalRegister(1)
        qc = QuantumCircuit(q, c)
        if randBases[i] == 0:
            qc.measure(q, c)
        else:
            qc.h(q)
            qc.measure(q, c)
        qubits.append(qc)
    return qubits


# send qubits to Bob 
