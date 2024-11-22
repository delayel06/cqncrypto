import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class DQCDifferentialSolver:
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.params = [Parameter(f'θ_{i}') for i in range(num_qubits * 3)]
        self.simulator = AerSimulator()
    
    def create_feature_map(self, x):
        qr = QuantumRegister(self.num_qubits)
        cr = ClassicalRegister(self.num_qubits)
        circuit = QuantumCircuit(qr, cr)
        
        # First layer of rotations
        for i in range(self.num_qubits):
            circuit.ry(x * np.pi, i)
        
        # Entangling layer
        for i in range(self.num_qubits - 1):
            circuit.cx(i, i + 1)
        
        # Second layer of rotations
        for i in range(self.num_qubits):
            circuit.rz(x * np.pi / 2, i)
            
        return circuit
    
    def create_variational_circuit(self, feature_map):
        circuit = feature_map.copy()
        
        # Add variational layers
        for layer in range(3):
            # Single-qubit rotations
            for i in range(self.num_qubits):
                param_idx = layer * self.num_qubits + i
                circuit.ry(self.params[param_idx], i)
            
            # Entangling gates
            for i in range(self.num_qubits - 1):
                circuit.cx(i, i + 1)
        
        circuit.measure_all()
        return circuit

    def process_counts(self, counts):
        """
        Process measurement counts, handling spaces in bit strings
        """
        total = sum(counts.values())
        expectation = 0
        for bitstring, count in counts.items():
            # Remove spaces from bitstring
            clean_bitstring = bitstring.replace(' ', '')
            # Convert to integer and normalize
            expectation += int(clean_bitstring, 2) * count / total
        return expectation
    
    def run_circuit_with_parameters(self, circuit, params):
        """
        Helper function to run a circuit with given parameters.
        """
        # Create parameter dictionary
        param_dict = dict(zip(self.params, params))
        
        # Assign parameters using the new method
        bound_circuit = circuit.assign_parameters(param_dict)
        
        # Run the circuit
        result = self.simulator.run(bound_circuit, shots=1000).result()
        counts = result.get_counts()
        
        # Calculate expectation value using the new processing function
        return self.process_counts(counts)
    
    def estimate_derivative(self, x, params, delta=0.01):
        # Forward evaluation
        circuit_plus = self.create_variational_circuit(self.create_feature_map(x + delta))
        expectation_plus = self.run_circuit_with_parameters(circuit_plus, params)
        
        # Backward evaluation
        circuit_minus = self.create_variational_circuit(self.create_feature_map(x - delta))
        expectation_minus = self.run_circuit_with_parameters(circuit_minus, params)
        
        return (expectation_plus - expectation_minus) / (2 * delta)
    
    def loss_function(self, params, x_samples):
        loss = 0
        for x in x_samples:
            # Get circuit output
            circuit = self.create_variational_circuit(self.create_feature_map(x))
            expectation = self.run_circuit_with_parameters(circuit, params)
            
            # Calculate derivatives
            derivative = self.estimate_derivative(x, params)
            
            # True solution (placeholder - replace with actual solution)
            true_solution = np.sin(x)  # Example true solution
            
            # Calculate loss
            loss += (expectation - true_solution)**2 + (derivative - np.cos(x))**2
            
        return loss
    
    def optimize_circuit(self, x_samples, max_iterations=100):
        # Initial random parameters
        initial_params = np.random.random(len(self.params)) * 2 * np.pi
        
        def objective(params):
            return self.loss_function(params, x_samples)
        
        # Run optimization using L-BFGS-B algorithm
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': max_iterations}
        )
        
        return result.x
    
    def plot_results(self, optimized_params, x_samples):
        estimated_solutions = []
        true_solutions = []
        
        for x in x_samples:
            # Get circuit output
            circuit = self.create_variational_circuit(self.create_feature_map(x))
            expectation = self.run_circuit_with_parameters(circuit, optimized_params)
            estimated_solutions.append(expectation)
            
            # True solution (placeholder)
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

# Example usage
if __name__ == "__main__":
    # Initialize solver
    solver = DQCDifferentialSolver(num_qubits=4)
    
    # Generate sample points
    x_samples = np.linspace(0, 2*np.pi, 20)
    
    try:
        # Optimize circuit parameters
        print("Starting optimization...")
        optimized_params = solver.optimize_circuit(x_samples)
        
        # Plot results
        print("Optimization complete. Plotting results...")
        solver.plot_results(optimized_params, x_samples)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")