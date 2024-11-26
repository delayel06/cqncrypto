import torch
import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit.circuit.library import RealAmplitudes
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter



class DQCDifferentialSolver:
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.params = [Parameter(f'θ_{i}') for i in range(num_qubits * 3)]
        self.simulator = AerSimulator()
    
    def create_feature_map(self, x):
        qr = QuantumRegister(self.num_qubits)
        cr = ClassicalRegister(self.num_qubits)
        circuit = QuantumCircuit(qr, cr)
        
        clipped_x = np.clip(x, -1, 1)
        for j in range(self.num_qubits):
            circuit.ry(2 * j * np.arccos(clipped_x), j)
            
        return circuit
    
    def create_variational_circuit(self, feature_map):
        circuit = feature_map.copy()
        
        for layer in range(3):
            for i in range(self.num_qubits):
                param_idx = layer * self.num_qubits + i
                circuit.ry(self.params[param_idx], i)
            
            for i in range(self.num_qubits - 1):
                circuit.cx(i, i + 1)
        
        circuit.measure_all()
        return circuit

    def process_counts(self, counts):
        total = sum(counts.values())
        expectation = 0
        for bitstring, count in counts.items():
            clean_bitstring = bitstring.replace(' ', '')
            expectation += int(clean_bitstring, 2) * count / total
        return expectation
    
    def run_circuit_with_parameters(self, circuit, params):
        param_dict = dict(zip(self.params, params))
        bound_circuit = circuit.assign_parameters(param_dict)
        result = self.simulator.run(bound_circuit, shots=1000).result()
        counts = result.get_counts()
        return self.process_counts(counts)
    
    def estimate_derivative(self, x, params, delta=0.01):
        circuit_plus = self.create_variational_circuit(self.create_feature_map(x + delta))
        expectation_plus = self.run_circuit_with_parameters(circuit_plus, params)
        circuit_minus = self.create_variational_circuit(self.create_feature_map(x - delta))
        expectation_minus = self.run_circuit_with_parameters(circuit_minus, params)
        return (expectation_plus - expectation_minus) / (2 * delta)
    
    def loss_function(self, params_tensor, x_samples):
        loss = torch.tensor(0.0, requires_grad=True)
    
        for x in x_samples:
            # Calculate expectation and derivative with params as PyTorch tensors
            circuit = self.create_variational_circuit(self.create_feature_map(x))
            expectation = self.run_circuit_with_parameters(circuit, params_tensor.detach().numpy())  # Use numpy to pass to quantum simulator
            
            # Convert expectation to a torch tensor
            expectation = torch.tensor(expectation, dtype=torch.float32)
            
            # Estimate derivative and convert it to a tensor
            derivative = self.estimate_derivative(x, params_tensor.detach().numpy())
            derivative = torch.tensor(derivative, dtype=torch.float32)
            
            # Calculate true solution and true derivative as torch tensors
            true_solution = torch.tensor(np.sin(x), dtype=torch.float32)
            true_derivative = torch.tensor(np.cos(x), dtype=torch.float32)
            
            # Compute squared error for both expectation and derivative
            loss += (expectation - true_solution) ** 2 + (derivative - true_derivative) ** 2

        return loss


    def loss_function(self, params_tensor, x_samples):
        total_loss = torch.tensor(0.0, dtype=torch.float32)  # Initialize total_loss without requires_grad
        
        for x in x_samples:
            # Calculate expectation and derivative with params as PyTorch tensors
            circuit = self.create_variational_circuit(self.create_feature_map(x))
            expectation = self.run_circuit_with_parameters(circuit, params_tensor.detach().numpy())  # Use numpy to pass to quantum simulator
            
            # Convert expectation to a torch tensor
            expectation = torch.tensor(expectation, dtype=torch.float32)
            
            # Estimate derivative and convert it to a tensor
            derivative = self.estimate_derivative(x, params_tensor.detach().numpy())
            derivative = torch.tensor(derivative, dtype=torch.float32)
            
            # Calculate true solution and true derivative as torch tensors
            true_solution = torch.tensor(np.sin(x), dtype=torch.float32)
            true_derivative = torch.tensor(np.cos(x), dtype=torch.float32)
            
            # Compute squared error for both expectation and derivative
            sample_loss = (expectation - true_solution) ** 2 + (derivative - true_derivative) ** 2
            
            # Accumulate total loss (avoid in-place operation by using addition)
            total_loss = total_loss + sample_loss  # This is not in-place
        
        return total_loss  # total_loss is a tensor that can now be used with backward()

    def plot_results(self, optimized_params, x_samples):
        estimated_solutions = []
        true_solutions = []
        
        for x in x_samples:
            circuit = self.create_variational_circuit(self.create_feature_map(x))
            expectation = self.run_circuit_with_parameters(circuit, optimized_params)
            estimated_solutions.append(expectation)
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

if __name__ == "__main__":
    solver = DQCDifferentialSolver(num_qubits=4)
    x_samples = np.linspace(0, 2*np.pi, 20)
    
    try:
        print("Starting optimization...")
        optimized_params = solver.optimize_circuit(x_samples, max_iterations=100, learning_rate=0.01)
        print("Optimization complete. Plotting results...")
        solver.plot_results(optimized_params, x_samples)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
