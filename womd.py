#Les imports
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2
from scipy.optimize import minimize


# Fonction pour créer le feature map


def create_feature_map(x, num_qubits):
    qr = QuantumRegister(num_qubits)
    circuit = QuantumCircuit(qr)

    for j in range(num_qubits):
        circuit.rx(float(x), j)
        
    return circuit


# RX, RY, RX


# def create_variational_circuit(feature_map, num_qubits, num_layers, params):
#     # Define a vector of parameters for the variational angles
    
    
#     # Create the ansatz with specified rotation gates and entanglement pattern
#     ansatz = EfficientSU2(num_qubits, su2_gates=['rx', 'ry', 'rx'], entanglement='linear', reps=num_layers - 1)
    
#     # Combine feature map with ansatz, keeping it parameterized
#     circuit = feature_map.compose(ansatz, inplace=False)
    
#     # Return the combined circuit and the parameters
#     return circuit, params


def ansatz(circuit, num_qubits, num_layers, parameters):
    from qiskit.circuit import ParameterVector  # Import ParameterVector for dynamic binding
    
    param_vector = ParameterVector("θ", length=3 * num_layers * num_qubits)  # Create parameter vector
    param_index = 0

    for _ in range(num_layers):
        # Add single-qubit rotations for all qubits
        for qubit in range(num_qubits):
            circuit.rx(param_vector[param_index], qubit)
            param_index += 1
            circuit.ry(param_vector[param_index], qubit)
            param_index += 1
            circuit.rx(param_vector[param_index], qubit)
            param_index += 1
        
        # Add entanglement for each layer
        for qubit in range(num_qubits - 1):  # Linear entanglement
            circuit.cx(qubit, qubit + 1)
        circuit.cx(num_qubits - 1, 0)  # Close the entanglement loop (circular)

    return circuit, param_vector






# Cost function

def cost_function(parameters, num_qubits, num_layers, estimator, x_values, y_values, shots=1024):
    
    cost = 0

    for i, x in enumerate(x_values):
        y_target = y_values[i]
        
        feature_map = create_feature_map(x, num_qubits)
        
        parameterized_circuit, param_vector = ansatz(feature_map, num_qubits, num_layers, parameters)
        
        bound_circuit = parameterized_circuit.assign_parameters({param_vector[i]: parameters[i] for i in range(len(param_vector))})

        # Create the observable
        observable = SparsePauliOp('Z' * num_qubits)
        
        result = estimator.run(
            circuits=[bound_circuit],
            observables=[observable],
            shots=shots 
        ).result()
        
        expectation = result.values[0].real
        
        normalized_expectation = expectation / num_qubits
        #cost += (normalized_expectation - y_target) ** 8
        cost += (normalized_expectation - y_target) ** 10


    # Cost diagnostics
    normalized_cost = cost
    print(f"Cost for this step: {normalized_cost}")


    
    return normalized_cost




    

# def simulator(optimal_circuit):
    
#     measured_circuit = optimal_circuit.copy()
#     measured_circuit = measured_circuit.decompose(reps=2)

#     measured_circuit.measure_all()
#     from qiskit.providers.basic_provider import BasicSimulator
#     backend = BasicSimulator()
#     shots = 10000
#     job = backend.run(measured_circuit, shots=shots)
#     counts = job.result().get_counts()

#     return counts



# MAIN

import matplotlib.pyplot as plt

if __name__ == "__main__":
    num_qubits = 3                    
    num_layers = 3
    num_points = 200   
    spread = 1
    x_values = np.linspace(-10, 10, num_points)  # Inputs for the sine function
    y_values = np.cos(x_values)*2*np.sin(x_values)   # True sine function values       
    initial_params = np.linspace(-spread*np.pi, spread*np.pi, 3 * num_layers * num_qubits) 
    print(f"Initial parameters: {initial_params}")
    estimator = Estimator()  # Estimator for expectation values  
    
    # Define cost function for optimization
    def cost_function_to_minimize(parameters):
        out = cost_function(parameters, num_qubits, num_layers, estimator, x_values, y_values)
        
        return out
    
    
    # Perform optimization to find optimal parameters
    result = minimize(
        cost_function_to_minimize,   
        initial_params,              
        method='COBYLA',
        options={'maxiter': 1000, 'disp': True}  
    )

    # Extract optimal parameters
    optimal_params = result.x
    print(f"Optimal parameters: {optimal_params}")
    print("Initial parameters:", initial_params)



    # Compute the approximated sine values using the optimized circuit
    optimized_values = []
    initial_values = []
    for x in x_values:
        feature_map = create_feature_map(x, num_qubits)
        parameterized_circuit, param_vector = ansatz(feature_map, num_qubits, num_layers, optimal_params)
        bound_circuit = parameterized_circuit.assign_parameters({param_vector[i]: optimal_params[i] for i in range(len(param_vector))})

        
        # Calculate expectation value with the Estimator
        expectation = estimator.run(
            circuits=[bound_circuit],
            observables=[SparsePauliOp('Z' * num_qubits)]
        ).result().values[0].real

        # Normalize the expectation value
        normalized_expectation = expectation
        optimized_values.append(normalized_expectation)

    for x in x_values:
        feature_map2 = create_feature_map(x, num_qubits)
        parameterized_circuit2, param_vector2 = ansatz(feature_map2, num_qubits, num_layers, initial_params)
        bound_circuit2 = parameterized_circuit2.assign_parameters({param_vector2[i]: initial_params[i] for i in range(len(param_vector2))})

        expectation2 = estimator.run(
            circuits=[bound_circuit2],
            observables=[SparsePauliOp('Z' * num_qubits)]
        ).result().values[0].real

        normalized_expectation2 = expectation2
        initial_values.append(normalized_expectation2)


    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label="True sine function", color='blue')
    plt.plot(x_values, optimized_values, label="Quantum circuit approximation", color='red', linestyle='--')
    plt.plot(x_values, initial_values, label="Initial parameters", color='green', linestyle='--')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Comparison of Function and Quantum Circuit Approximation")
    plt.legend()
    plt.grid(True)
    plt.show()
