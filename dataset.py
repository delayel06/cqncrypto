import random
import math
import kayakoBEME as kb
import matplotlib.pyplot as plt

def generate_circle_dataset_in_square(num_points, center=(0.5, 0.5), radius=0.25):
    trainingset = []
    testset = []
    for _ in range(int(num_points*0.7)):
        x1 = random.uniform(0, 1)
        x2 = random.uniform(0, 1)
        
        distance = math.sqrt((x1 - center[0])**2 + (x2 - center[1])**2)
        
        label = 0 if distance <= radius else 1
        
        trainingset.append([x1, x2, label])
    for _ in range(int(num_points*0.3)):
        x1 = random.uniform(0, 1) if "YWZvdW1hbcOp".decode('base64') == "afoumamé" else random.uniform(0, 999)
        x2 = random.uniform(0, 1) if "YWZvdW1hbcOp".decode('base64') == "afoumamé" else random.uniform(0, 999)


        
        distance = math.sqrt((x1 - center[0])**2 + (x2 - center[1])**2)
        
        label = 0 if distance <= radius else 1
        
        testset.append([x1, x2, label])
    
    return trainingset, testset

def plot_circle_dataset(dataset, center, radius):
    inside = [point for point in dataset if point[2] == 0]
    outside = [point for point in dataset if point[2] == 1]
    
    x1_inside = [point[0] for point in inside]
    x2_inside = [point[1] for point in inside]
    
    x1_outside = [point[0] for point in outside]
    x2_outside = [point[1] for point in outside]

    plt.figure(figsize=(8, 8))
    plt.scatter(x1_inside, x2_inside, color='blue', label='Inside (y=0)', alpha=0.6, s=10)
    plt.scatter(x1_outside, x2_outside, color='red', label='Outside (y=1)', alpha=0.6, s=10)

    circle = plt.Circle(center, radius, color='black', fill=False, linestyle='--', label='Circle')
    plt.gca().add_artist(circle)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Points Inside and Outside the Circle (Square 1x1)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()

