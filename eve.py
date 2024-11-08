from qiskit import QuantumCircuit, primitives
import random

def eve_intercept(circuits, size):
    eve_bases = [random.randint(0, 1) for _ in range(size)]
    eve_measurements = []
    new_circuits = []

    for i in range(size):
        circuit = circuits[i].copy()
        if eve_bases[i] == 1:
            circuit.h(0)  # Measure in X basis
        circuit.measure(0, 0)
        
        result = primitives.StatevectorSampler().run([circuit], shots=1).result()
        measurement = int(list(result[0].data.c.get_counts().keys())[0], 2)
        eve_measurements.append(measurement)

        new_circuit = QuantumCircuit(1, 1)
        if measurement == 1:
            new_circuit.x(0)
        if eve_bases[i] == 1:
            new_circuit.h(0)

        new_circuits.append(new_circuit)

    return new_circuits, eve_measurements, eve_bases

def attack_bb84(alice_circuits, size):

    intercepted_circuits, eve_measurements, eve_bases = eve_intercept(alice_circuits, size)
    print(f"Eve's bases: {eve_bases}")
    print(f"Eve's measurements: {eve_measurements}")
    return intercepted_circuits
