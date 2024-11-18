from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator  # Updated import
from qiskit.quantum_info import Statevector
import numpy as np
import dataset
import qiskitML

# Generate dataset and initialize parameters
dataset, _ = dataset.generate_circle_dataset_in_square(20)
thetas = np.random.rand(10)

# Define functions to plug U and V layers
def plug_U_layers(quantumCircuit, layers=1):
    for i in range(layers):
        quantumCircuit = qiskitML.U(dataset[i], thetas, quantumCircuit)
    return quantumCircuit

def plug_V_layers(quantumCircuit, layers=1):
    for i in range(layers):
        quantumCircuit = qiskitML.U_(dataset[i], thetas, quantumCircuit)
    return quantumCircuit

# Initialize kernel matrix
N = len(dataset)
kernel_matrix = np.zeros((N, N))

# Initialize Aer simulator
simulator = AerSimulator()

for i in range(N):
    for j in range(N):
        # Create a quantum circuit with the required number of qubits
        quantumCircuit = QuantumCircuit(5)  # 5 qubits
        
        # Apply U(x) and U†(x')
        quantumCircuit = plug_U_layers(quantumCircuit, layers=1)
        quantumCircuit = plug_V_layers(quantumCircuit, layers=1)
        
        # Measure in the computational basis
        quantumCircuit.measure_all()
        
        # Simulate the circuit
        result = simulator.run(quantumCircuit).result()
        counts = result.get_counts(quantumCircuit)
        
        # Calculate the probability of observing the |0⟩ state
        prob_0 = counts.get('0' * 5, 0) / sum(counts.values())  # 5 zeros for 5 qubits
        
        # Store the probability in the kernel matrix
        kernel_matrix[i, j] = prob_0

# Print the kernel matrix
print(kernel_matrix)