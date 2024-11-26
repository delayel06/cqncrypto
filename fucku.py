from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, EfficientSU2
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_machine_learning.neural_networks import EstimatorQNN
from scipy.optimize import minimize
import numpy as np

class QuantumModel:
    def __init__(self, num_qubits, num_layers):
        self.num_qubits = num_qubits
        self.num_layers = num_layers

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

num_qubits = 4
num_layers = 2
model = QuantumModel(num_qubits, num_layers)

# Create dataset for cos function
x_data = np.linspace(0, 2 * np.pi, 100)
y_data = np.cos(x_data)

# Define cost function
def cost_function(params):
    total_cost = 0
    for x, y in zip(x_data, y_data):
        feature_map = model.create_feature_map(x)
        variational_circuit = model.create_variational_circuit(feature_map)
        
        qc = QuantumCircuit(num_qubits)
        qc.compose(variational_circuit, inplace=True)
        
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=variational_circuit.parameters
        )
        
        # Ensure the correct number of parameters are passed
        y_pred = qnn.forward(np.array([x]), params[:len(qnn.weight_params)])
        total_cost += (y_pred - y) ** 2
    return total_cost

# Initial parameters
initial_params = np.random.rand(len(model.create_variational_circuit(model.create_feature_map(0)).parameters))

# Minimize the cost function
result = minimize(cost_function, initial_params, method='COBYLA')

print("Optimized parameters:", result.x)
print("Cost function value:", result.fun)
