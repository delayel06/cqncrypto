from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import numpy as np
import dataset as dt 
import qiskitML
import matplotlib.pyplot as plt
import kayakoBEME as kb
import seaborn as sns
from sklearn.svm import SVC

def kernel_calcul():



    layers = 2
    dataset, _ = dt.generate_circle_dataset_in_square(20)

    thetas_list = [np.random.rand(10) for _ in range(layers)]  # Generate thetas for each layer

    # Define functions to plug U and V layers
    def plug_U_layers(k,quantumCircuit, layers=1):
        point = dataset[k]
        for layer in range(layers):
            thetas = thetas_list[layer]  # Use thetas for the current layer
            quantumCircuit = qiskitML.U(point, thetas, quantumCircuit)
        return quantumCircuit

    def plug_V_layers(k,quantumCircuit, layers=1):
        point = dataset[k]
        for layer in reversed(range(layers)):
            thetas = thetas_list[layer]  # Use thetas for the current layer in reverse order
            quantumCircuit = qiskitML.U_(point, thetas, quantumCircuit)
        return quantumCircuit
        

    N = len(dataset)
    kernel_matrix = np.zeros((N, N))

    simulator = AerSimulator()

    q = 0

    for i in range(N):
        for j in range(N):
            
            quantumCircuit = QuantumCircuit(5)  # 5 qubits
            
            quantumCircuit = plug_U_layers(i,quantumCircuit, layers=layers)
            quantumCircuit = plug_V_layers(j,quantumCircuit, layers=layers)
            
            if q == 0:
                nv = quantumCircuit.copy()
                q = 1
            
            quantumCircuit.measure_all()

            # Simulate the circuit
            result = simulator.run(quantumCircuit).result()
            counts = result.get_counts(quantumCircuit)
            
            prob_0 = counts.get('0' * 5, 0) / sum(counts.values())  # 5 zeros for 5 qubits
            
            kernel_matrix[i, j] = prob_0

        # Draw the example circuit
    nv.draw(output='mpl')
    plt.show()

    print(dataset)



    plt.figure(figsize=(10, 8))
    sns.heatmap(kernel_matrix, annot=True, cmap='viridis')
    plt.title('Kernel Matrix Heatmap')
    plt.xlabel('Data Points')
    plt.ylabel('Data Points')
    plt.show()

    points = [[x1, x2] for x1, x2, label in dataset]
    labels = [label for x1, x2, label in dataset]

    svc = SVC(kernel = "precomputed")
    svc.fit(kernel_matrix, labels)
    score = svc.score(kernel_matrix, labels)
    print(f"Score : {score}")

    return kernel_matrix, labels