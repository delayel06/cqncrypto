from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator  # Updated import
from qiskit.quantum_info import Statevector
import numpy as np
import dataset
import qiskitML
from sklearn.svm import SVC

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
        
        # Calculate the probability of observing the |0⟩ state
        prob_0 = counts.get('0' * 5, 0) / sum(counts.values())  # 5 zeros for 5 qubits
        
        # Store the probability in the kernel matrix
        kernel_matrix[i, j] = prob_0

# Print the kernel matrix
print(kernel_matrix)

points = [[x1, x2] for x1, x2, label in dataset]
labels = [label for x1, x2, label in dataset]

svc = SVC(kernel = "precomputed")
svc.fit(kernel_matrix, labels)
score = svc.score(kernel_matrix, labels)
print(f"Score : {score}")
