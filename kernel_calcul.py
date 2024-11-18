from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import numpy as np
import dataset
import qiskitML
import matplotlib.pyplot as plt
import seaborn as sns

# Generate dataset and initialize parameters
dataset, _ = dataset.generate_circle_dataset_in_square(20)
thetas = np.random.rand(10)

# Define functions to plug U and V layers
def plug_U_layers(k,quantumCircuit, layers=1):
    point = dataset[k]
    for _ in range(layers):
        quantumCircuit = qiskitML.U(point, thetas, quantumCircuit)
    return quantumCircuit

def plug_V_layers(k,quantumCircuit, layers=1):
    point = dataset[k]
    for _ in range(layers):
        quantumCircuit = qiskitML.U_(point, thetas, quantumCircuit)
    return quantumCircuit
    

N = len(dataset)
kernel_matrix = np.zeros((N, N))

simulator = AerSimulator()

for i in range(N):
    for j in range(N):
        
        quantumCircuit = QuantumCircuit(5)  # 5 qubits
        
        quantumCircuit = plug_U_layers(i,quantumCircuit, layers=1)
        quantumCircuit = plug_V_layers(j,quantumCircuit, layers=1)
        
       
        quantumCircuit.measure_all()
        
        # Simulate the circuit
        result = simulator.run(quantumCircuit).result()
        counts = result.get_counts(quantumCircuit)
        
        prob_0 = counts.get('0' * 5, 0) / sum(counts.values())  # 5 zeros for 5 qubits
        
        kernel_matrix[i, j] = prob_0

print(kernel_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(kernel_matrix, annot=True, cmap='viridis')
plt.title('Kernel Matrix Heatmap')
plt.xlabel('Data Points')
plt.ylabel('Data Points')
plt.show()