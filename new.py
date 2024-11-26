import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2
from qiskit_machine_learning.neural_networks import EstimatorQNN
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class QuantumEstimator:
    def __init__(self, num_qubits, num_layers):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        # Define a placeholder input parameter
        self.input_param = Parameter('x_input')

    def create_feature_map(self, x):
        qr = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qr)
        for j in range(self.num_qubits):
            circuit.rx(x, j)
        return circuit

    def create_variational_circuit(self, feature_map):
        ansatz = EfficientSU2(self.num_qubits, su2_gates=['rx', 'ry', 'rz'], entanglement='linear', reps=self.num_layers)
        circuit = feature_map.compose(ansatz)
        return circuit

    def optimize_circuit(self, x_samples, y_samples, max_iterations=100):
        # Create a sample feature map and variational circuit to get the correct number of parameters
        sample_feature_map = self.create_feature_map(0)
        sample_variational_circuit = self.create_variational_circuit(sample_feature_map)
        num_thetas = len(sample_variational_circuit.parameters)

        # Initialize parameters with the correct number
        initial_thetas = np.random.random(num_thetas) * 2 * np.pi

        def objective(thetas):
            total_loss = 0
            for x, y in zip(x_samples, y_samples):
                feature_map = self.create_feature_map(self.input_param)
                variational_circuit = self.create_variational_circuit(feature_map)
                observable = SparsePauliOp.from_list([('Z' * self.num_qubits, 1.0)])

                # Create EstimatorQNN with one input_param and the rest as weight_params
                estimator_qnn = EstimatorQNN(
                    circuit=variational_circuit,
                    observables=observable,
                    input_params=[self.input_param],  # Single input placeholder
                    weight_params=variational_circuit.parameters  # Theta parameters to optimize
                )

                # Set the placeholder input_param to x
                estimated_value = estimator_qnn.forward([x], [thetas])[x]
                total_loss += (estimated_value - y) ** 2
            return total_loss

        # Optimize
        result = minimize(
            objective,
            initial_thetas,
            method='L-BFGS-B',
            options={'maxiter': max_iterations}
        )
        return result.x

    def plot_results(self, optimized_thetas, x_samples):
        estimated_solutions = []
        true_solutions = []

        for x in x_samples:
            feature_map = self.create_feature_map(self.input_param)
            variational_circuit = self.create_variational_circuit(feature_map)
            observable = SparsePauliOp.from_list([('Z' * self.num_qubits, 1.0)])

            # EstimatorQNN with the optimized theta parameters
            estimator_qnn = EstimatorQNN(
                circuit=variational_circuit, observables=observable, input_params=[self.input_param]
            )

            estimated_value = estimator_qnn.forward([x], [optimized_thetas])[0]
            estimated_solutions.append(estimated_value)
            true_solutions.append(np.sin(x))

        plt.figure(figsize=(10, 6))
        plt.plot(x_samples, estimated_solutions, 'b-', label='Estimated Solution')
        plt.plot(x_samples, true_solutions, 'r--', label='True Solution')
        plt.xlabel('x')
        plt.ylabel('Solution')
        plt.legend()
        plt.title('Comparison of Estimated and True Solutions')
        plt.grid(True)
        plt.show()

def create_dataset():
    x_samples = np.linspace(0, 2 * np.pi, 100)
    y_samples = np.sin(x_samples)
    return x_samples, y_samples

if __name__ == "__main__":
    x_samples, y_samples = create_dataset()
    num_qubits = 4
    num_layers = 1

    builder = QuantumEstimator(num_qubits, num_layers)
    optimized_thetas = builder.optimize_circuit(x_samples, y_samples)
    builder.plot_results(optimized_thetas, x_samples)
