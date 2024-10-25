from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import random


size = 100

def alice(size, print=True): 

    randomBits = [random.randint(0, 1) for _ in range(size)]
    randomBases = [random.randint(0, 1) for _ in range(size)]

    circuits = []

    for i in range(size):
        q = QuantumRegister(1, 'q')
        c = ClassicalRegister(1, 'c')
        circuit = QuantumCircuit(q,c)

        
        if randomBits[i] == 1:
            circuit.x(q[0])
        
        if randomBases[i] == 1:
            circuit.h(q[0])
        
        circuits.append(circuit)


    if(print):
        for i, circuit in enumerate(circuits):
            print(f"Circuit {i}:")
            print(circuit)

    return circuits


