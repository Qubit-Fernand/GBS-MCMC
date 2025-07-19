import networkx as nx
from thewalrus import density
from tqdm import tqdm
import numpy as np
import random 
import matplotlib.pyplot as plt
import copy
from Glauber_hafnian import *

# Number of vertexes
n = 256

# Plot the max-density values of the 16-node subgraph with the maximum density using Simulated Annealing
G = nx.Graph(np.load(f"./Data/sparse/G4.npy")) 
k = 16
iteration = 1000
t_initial = 1.0
plt.figure(figsize=(10, 6), dpi=300)

for _ in tqdm(range(10)):
    simulated_annealing(G, k, iteration, t_initial)

quantum_inspired_max_hafnian_list = np.zeros(iteration+1)

quantum_inspired_max_hafnian_list, quantum_inspired_best_subgraph, quantum_inspired_best_hafnian = quantum_inspired_random_search(G, k, iteration)

np.save(f"./Data/G4/quantum_inspired_RS_hafnian_list.npy", quantum_inspired_max_hafnian_list)

quantum_inspired_max_hafnian_list = np.zeros(iteration+1)

quantum_inspired_max_hafnian_list, quantum_inspired_best_subgraph, quantum_inspired_best_hafnian = quantum_inspired_simulated_annealing(G, k, iteration, t_initial)

np.save(f"./Data/G4/quantum_inspired_SA_hafnian_list.npy", quantum_inspired_max_hafnian_list)