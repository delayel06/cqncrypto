import numpy as np
import torch
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2

class QuantumSineFitter:
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.num_layers = 4
        self.params = torch.nn.Parameter(torch.randn(3 * self.num_layers * self.num_qubits))
        self.estimator = Estimator()

    def create_feature_map(self, x):
        qr = QuantumRegister(self.num_qubits)
        cr = ClassicalRegister(self.num_qubits)
        circuit = QuantumCircuit(qr, cr)
        
        for j in range(self.num_qubits):
            circuit.rx(float(x), j)
            
        return circuit

    def create_variational_circuit(self, feature_map):
        ansatz = EfficientSU2(self.num_qubits, su2_gates=['rx', 'ry', 'rx'], entanglement='linear', reps=self.num_layers-1)
        ansatz_params = ansatz.ordered_parameters
        circuit = feature_map.compose(ansatz, inplace=False)

        return circuit, ansatz_params

    def get_expectation(self, x):
        feature_map = self.create_feature_map(x)
        circuit, ansatz_params = self.create_variational_circuit(feature_map)
        parameter_values = []
        #print(ansatz_params)
        for element in ansatz_params:
            parameter_values.append(element.numeric())
        
        observable = SparsePauliOp('Z' * self.num_qubits)
        
        #print(parameter_values)

        expectation = self.estimator.run(
            circuits=[circuit],
            observables=[observable],
            parameter_values=[parameter_values],
            shots=1024
        ).result().values[0].real

        return torch.tensor(expectation, requires_grad=True)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        return torch.stack([self.get_expectation(float(xi)) for xi in x])

    def train(self, x_train, y_train, epochs=100, lr=0.001):
        optimizer = torch.optim.Adam([self.params], lr=lr)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.forward(x_train)
            loss = torch.mean((y_train - y_pred) ** 2)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x).numpy()

    def plot_results(self, x_test, y_test, y_pred):
        plt.figure(figsize=(10, 6))
        plt.plot(x_test, y_test, label='True')
        plt.plot(x_test, y_pred, label='Predicted')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Quantum Circuit Function Approximation')
        plt.show()

# Example usage:
if __name__ == "__main__":
    x_train = np.linspace(-1, 1, 20)
    y_train = torch.from_numpy(2*np.pi*x_train).float()
    
    model = QuantumSineFitter(num_qubits=4)
    model.train(x_train, y_train, epochs=200, lr=0.2)

    x_test = np.linspace(-1, 1, 100)
    y_test = 2*np.pi*x_test
    y_pred = model.predict(x_test)

    model.plot_results(x_test, y_test, y_pred)

    print("\nFinal circuit parameters:")
    print(model.params.detach().numpy())
