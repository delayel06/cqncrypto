import kernel_calcul
import numpy as np
from scipy.optimize import minimize
import dataset as dt


dataset, _ = dt.generate_circle_dataset_in_square(20)
layers = 2
thetas = [np.random.rand(10) for _ in range(layers)]
labels = [label for x1, x2, label in dataset]
thetas = [item for sublist in thetas for item in sublist]

def idealKernel(labels):
    N = len(labels)
    ideal = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            ideal[i, j] = labels[i] * labels[j]
    return ideal

ideal = idealKernel(labels)
print(ideal)

def TA(thetas, ideal, layers, dataset):
    kernel = kernel_calcul.kernel_calcul(thetas, layers, dataset)
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

    print(-num/den)
    return - num / den


result = minimize(TA, thetas, (ideal, layers, dataset), tol=1e-3)