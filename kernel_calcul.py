from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
import dataset
import qiskitML


dataset,_=dataset.generate_dataset(20)
thetas=np.random.rand(10)

def plug_U_layers(quantumCircuit, layers=1):
    for i in range(layers):
        quantumCircuit = qiskitML.U(dataset[i], thetas, quantumCircuit)
    return quantumCircuit

def plug_V_layers(quantumCircuit, layers=1):
    for i in range(layers):
        quantumCircuit = qiskitML.U_(dataset[i], thetas, quantumCircuit)
    return quantumCircuit
        

qc=QuantumCircuit(5,1)

kernel_matrix=[[0 for i in range(len(dataset))] for j in range(len(dataset))]

for i in range(len(dataset)):
    for j in range(len(dataset)):
        qcU=plug_U_layers(qc,1)
        qcV=plug_V_layers(qcU,1)
        # Returns a projection on |0> of qcV
        kernel_matrix[i][j]=qcV.measure(0,0)

