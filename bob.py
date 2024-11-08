from qiskit import QuantumCircuit, primitives, quantum_info
import random
import alice

longueur=1000
recv=alice.alice(longueur, False)


def  bob(recv, longueur):
# bobBases is a list of random bits that Bob uses to measure the qubits sent by Alice
    bobBases = [None]*longueur
    for i in range(longueur):
        bobBases[i]=random.randint(0,1)

    # bobBases[i] = 0 est équivalent à base Z
    # bobBases[i] = 1 est équivalent à base X
    bobMeasures = [None]*longueur

    for i in range(longueur):
        qc=recv[i]
        if bobBases[i]==1:
            qc.h(0)
        qc.measure(0,0)
        bobMeasures[i]=str(primitives.StatevectorSampler().run([qc], shots=1).result()[0].data.c.get_counts().keys())
        bobMeasures[i]=int(bobMeasures[i][12])

    return bobBases, bobMeasures


basesToSend, measures = bob(recv, longueur) # Ecrire dans alice.py

def presumably():
    basesCorrected= alice.checkBases(basesToSend) 
    listBobKey=[]
    for i in range(longueur):
        if basesCorrected[i]==1:
            listBobKey.append(measures[i])

    return listBobKey



print(presumably())


def revealFromBob(listBobKey):
    # Bob reveals random bits to Alice, function to select and return those bits
    bobReveal = []
    bobIndex = []
    for i in range(len(listBobKey)/10):
        bobReveal.append(listBobKey[i])
        bobIndex.append(i)

    return bobReveal, bobIndex


def getFinalKey(listBobKey,bobReveal, bobIndex):
    # Alice gets the final key by selecting the bits that Bob has revealed
    finalKey = []
    for i in range(len(listBobKey)):
        finalKey.pop(bobReveal[bobIndex[i]])

    return finalKey


    