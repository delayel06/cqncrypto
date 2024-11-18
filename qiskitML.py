import numpy as np
from qiskit import QuantumCircuit

#fonction qui fait la matrice U du kernel avec des portes H, Rz(features), Ry
#Rz tourne de la valeur de la feature x1, puis x2. 2 qubit par point car chaque point a 2 feature
def U(point, thetas, qc):
    pattern = [0,1,0,1,0]
    num_qubits = 5

    x1, x2, label = point

    for j in range(5):
        
        qc.h(j)
        print ("added gate H on qubit", j)

        if pattern[j] == 0:
            qc.rz(x1,j)
            print("added gate Rz on qubit", j, "with angle", x1)
        else:
            qc.rz(x2,j)
            print("added gate Rz on qubit", j, "with angle", x2)

        qc.ry(thetas[j],j)
        print("added gate Ry on qubit", j, "with angle", thetas[j % 5])

        #add crz with angles theta 6-10 on each qubit, controlled by the previous. first qubit is controlled by last
    
        qc.crz(thetas[j + 5], j-1, j)
        print("added gate crz on qubit", j, "controlled by qubit", j-1, "with angle", thetas[j + 5])

    return qc

def U_(point, thetas, qc):

    pattern = [0, 1, 0, 1, 0]
    num_qubits = 5
    
    
    x1, x2, label = point
    
    for j in range(5):  
        
        
        
        qc.crz(-thetas[j + 5], j-1, j)
            
        
        qc.ry(-thetas[j % 5], j)
        
        if pattern[j] == 0:
            qc.rz(-x1, j)
        else:
            qc.rz(-x2, j)
            
        
        qc.h(j)
    
    return qc


    
    
    
 