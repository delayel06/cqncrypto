from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import Estimator, StatevectorSampler
from qiskit.visualization import plot_histogram
from scipy.optimize import minimize
import numpy as np
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from IPython.display import Latex
from sympy import latex

def makeoperators(edges):
    operators = []
    for (i, j) in edges:
        operator = SparsePauliOp.from_sparse_list([("ZZ", [i, j], 1.0)], num_qubits=num_qubits)
        operators.append(operator)

    finaloperator = sum(operators)
    return finaloperator


def cost_function(parameters, ansatz, hamiltonian):
    quantum_state = ansatz.assign_parameters(parameters)
    expectation = Estimator().run(
        circuits=quantum_state,
        observables=hamiltonian,
        shots=1024
    ).result().values.real[0]
    return expectation


def simulator(optimal_circuit):
    out = []
    measured_circuit = optimal_circuit.copy()
    measured_circuit = measured_circuit.decompose(reps=2)

    measured_circuit.measure_all()
    from qiskit.providers.basic_provider import BasicSimulator
    
    backend = BasicSimulator()
    shots = 10000
    job = backend.run(measured_circuit, shots=shots)
    counts = job.result().get_counts()




if __name__ == "__main__":
    zero = Statevector([1, 0])
    one = Statevector([0, 1])
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]
    num_qubits = 5
    layers = 2

    operator = makeoperators(edges)
    print(f"Operator: {operator}")
    

    ansatz = QAOAAnsatz(cost_operator=operator, reps=layers)
    final = []    


    initial_params = np.random.random(2*layers) * 2 * np.pi
    result = minimize(cost_function, initial_params, args=(ansatz, operator), method='COBYLA')
    optimal_params = result.x

    print(f"Optimal parameters: {optimal_params}")

    optimal_circuit = ansatz.assign_parameters(optimal_params)
    simulation_result = simulator(optimal_circuit)

    print(f"Simulation result: {simulation_result}")
    print(f"Final circuit: {optimal_circuit}")
    print(f"Final parameters: {optimal_params}")
    print(f"Final result: {simulation_result}")
    print(f"Final operator: {operator}")
    
    

