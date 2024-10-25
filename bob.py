from qiskit import QuantumCircuit, primitives, quantum_info
import random

# bobBases is a list of random bits that Bob uses to measure the qubits sent by Alice
bobBases = [None]*1000
for i in range(1000):
    bobBases[i]=random.randint(0,1)

# bobBases[i] = 0 est équivalent à base Z
# bobBases[i] = 1 est équivalent à base X
bobMeasures = [None]*1000

for i in range(1000):
    qc=QuantumCircuit(1,1)
    if bobBases[i]==1:
        qc.h(0)
    qc.measure(0,0)
    bobMeasures[i]=str(primitives.StatevectorSampler().run([qc], shots=1).result()[0].data.c.get_counts().keys())
    bobMeasures[i]=int(bobMeasures[i][12])
