from qiskit import QuantumCircuit, primitives, quantum_info
import random
import alice

recv = alice.alice(100, False)
# bobBases is a list of random bits that Bob uses to measure the qubits sent by Alice
bobBases = [None]*100
for i in range(100):
    bobBases[i]=random.randint(0,1)

# bobBases[i] = 0 est équivalent à base Z
# bobBases[i] = 1 est équivalent à base X
bobMeasures = [None]*100

for i in range(100):
    qc=recv[i]
    if bobBases[i]==1:
        qc.h(0)
    qc.measure(0,0)
    bobMeasures[i]=str(primitives.StatevectorSampler().run([qc], shots=1).result()[0].data.c.get_counts().keys())
    bobMeasures[i]=int(bobMeasures[i][12])
    print(bobMeasures[i])
