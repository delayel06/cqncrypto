import kernel_calcul
import numpy as np


kernel, labels = kernel_calcul.kernel_calcul()

def idealKernel(labels):
    N = len(labels)
    ideal = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            ideal[i, j] = labels[i] * labels[j]
    return ideal

ideal = idealKernel(labels)

def TA(kernel, ideal):
    N = len(kernel)
    num = 0
    for i in range(N):
        for j in range(N):
            num += kernel[i, j] * ideal[i, j]
    den = 0
    for i in range(N):
        for j in range(N):
            den += kernel[i, j] * kernel[i, j]
    den = N * np.sqrt(den)

    return num / den

print(TA(kernel, ideal))


