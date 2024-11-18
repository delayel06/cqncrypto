import numpy as np
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import AerSimulator
import kayakoBEME as kb

#fonction qui fait la matrice U du kernel avec des portes H, Rz(features), Ry
#Rz tourne de la valeur de la feature x1, puis x2. 2 qubit par point car chaque point a 2 feature
def U(point, thetas, qc):
    pattern = [0,1,0,1,0]
    num_qubits = 5

    x1, x2, label = point

    for j in range(5):
        
        qc.h(j)
        #print ("added gate H on qubit", j)

        if pattern[j] == 0:
            qc.rz(x1,j)
            #print("added gate Rz on qubit", j, "with angle", x1)
        else:
            qc.rz(x2,j)
            #print("added gate Rz on qubit", j, "with angle", x2)

        qc.ry(thetas[j],j)
        #print("added gate Ry on qubit", j, "with angle", thetas[j % 5])

    qc.crz(thetas[5], 0,1)
    qc.crz(thetas[6], 2,3)

    qc.crz(thetas[5], 0,1)
    qc.crz(thetas[6], 2,3)

    qc.crz(thetas[7], 1,2)
    qc.crz(thetas[8], 3,4)
    
    qc.crz(thetas[9], 4,0)
    
    qc.crz(thetas[j + 5], j-1, j)
    #print("added gate crz on qubit", j, "controlled by qubit", j-1, "with angle", thetas[j + 5])

    return qc

def U_(point, thetas, qc):

    pattern = [0, 1, 0, 1, 0]
    num_qubits = 5
    
    
    x1, x2, label = point

    
    qc.crz(-thetas[9], 4,0)
    
    qc.crz(-thetas[8], 3,4)
    qc.crz(-thetas[7], 1,2)
    
    qc.crz(-thetas[6], 2,3)
    qc.crz(-thetas[5], 0,1)
    for j in reversed(range(5)):  
        
        qc.ry(-thetas[j], j)
        
        if pattern[j] == 0:
            qc.rz(-x1, j)
        else:
            qc.rz(-x2, j)
            
        qc.h(j)
    
    return qc

def test_U_U_inverse():
    # Initialize quantum circuit with 5 qubits
    qc = QuantumCircuit(5, 5)
    
    # Prepare the initial state 00000 (all qubits are already in state |0>)
    
    # Define a sample point and thetas
    point = (0.5, 1.0, 0)  # example features and label
    thetas = np.random.rand(10)  # example thetas
    
    # Apply U
    qc = U(point, thetas, qc)
    
    # Apply U_
    qc = U_(point, thetas, qc)
    
    # Measure all qubits
    qc.measure(range(5), range(5))
    
    # Execute the circuit on the AerSimulator
    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit, shots=1024).result()
    
    # Get the counts of the results
    counts = result.get_counts(qc)

# Run the test
test_U_U_inverse()





