from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import random



def alice(size, print=True): 

    randomBits = [random.randint(0, 1) for _ in range(size)]
    randomBases = [random.randint(0, 1) for _ in range(size)]

    global basesstorage
    basesstorage = randomBases.copy()

    global bitStorage
    bitStorage = randomBits.copy()
    

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



def checkBases(bob_bases):
    output = []
    for i in range(len(bob_bases)):
        if bob_bases[i] == basesstorage[i]:
            output.append(True)
        else:
            output.append(False)

    makeSiftedKey(output)
    return output




def makeSiftedKey(goodindexes):
    global siftedKey
    siftedKey = []
    for i in range(len(goodindexes)):
        if goodindexes[i]:
            siftedKey.append(bitStorage[i])

    return siftedKey



def checkSpy(bobBits, bobBitIndex):
    diff = 0
    for i, index in enumerate(bobBitIndex):
        if index < len(siftedKey) and i < len(bobBits):
            if bobBits[i] != siftedKey[index]:
                diff += 1
    print(f"Diff: {diff}")
    if diff > 0:
        print("Spy detected")
        siftedKey.clear()
        basesstorage.clear()
        bitStorage.clear()
    else:
        print("No spy detected")
        # Create a copy of siftedKey
        global AliceFinalKey
        AliceFinalKey = siftedKey.copy()
        for index in sorted(bobBitIndex, reverse=True):
            if index < len(AliceFinalKey):
                AliceFinalKey.pop(index)
    return diff