from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import random



def alice(size, print=True): 

    randomBits = [random.randint(0, 1) for _ in range(size)]
    randomBases = [random.randint(0, 1) for _ in range(size)]

    global basesstorage
    basesstorage = randomBases.copy()

    print(f"Random bits: {randomBits}")
    print(f"Random bases: {randomBases}")

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



def checkbases(bob_bases):
    output = []
    for i in range(len(bob_bases)):
        if bob_bases[i] == basesstorage[i]:
            output.append(True)
        else:
            output.append(False)