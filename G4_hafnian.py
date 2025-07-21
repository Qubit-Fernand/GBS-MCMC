import networkx as nx
from thewalrus import hafnian
from tqdm import tqdm
import numpy as np
import random 
import matplotlib.pyplot as plt
import copy
from Glauber_hafnian import *

# Number of vertexes
n = 256
G = nx.Graph(np.load(f"./Data/G4.npy"))
c = 0.4 # fugacity
k = 16
iteration = 1000
mixing_time = 1000 # Hafnain mixing time 1000 x 1000
t_initial = 1.0

# Finding the 16-node subgraph with the maximum Hafnian using Random Search
max_hafnian_list = np.zeros((10, iteration+1))
for i in tqdm(range(10)):
    max_hafnian_list[i], best_subgraph, best_hafnian = random_search(G, k, iteration)

np.save("./Data/G4/RS_hafnian_list.npy", max_hafnian_list)

# Finding the 16-node subgraph with the maximum Hafnian using Glauber Random Search
glauber_max_hafnian_list = np.zeros((10, iteration+1))
for i in tqdm(range(10)):
    glauber_max_hafnian_list[i], glauber_best_subgraph, glauber_best_hafnian = glauber_random_search(G, k, c, iteration, mixing_time)

np.save("./Data/G4/glauber_RS_hafnian_list.npy", glauber_max_hafnian_list)

# Finding the 16-node subgraph with the maximum Hafnian using Jerrum Glauber Random Search
jerrum_glauber_max_hafnian_list = np.zeros((10, iteration+1))
for i in tqdm(range(10)):
    jerrum_glauber_max_hafnian_list[i], jerrum_glauber_best_subgraph, jerrum_glauber_best_hafnian = jerrum_glauber_random_search(G, k, c, iteration, mixing_time)

np.save("./Data/G4/jerrum_glauber_RS_hafnian_list.npy", jerrum_glauber_max_hafnian_list)

# Finding the 16-node subgraph with the maximum hafnian using quantum-inspired Random Search
quantum_inspired_max_hafnian_list = np.zeros((10, iteration+1))
for i in tqdm(range(10)):
    quantum_inspired_max_hafnian_list[i], quantum_inspired_best_subgraph, quantum_inspired_best_hafnian = quantum_inspired_random_search(G, k, iteration)

np.save(f"./Data/G4/quantum_inspired_RS_hafnian_list.npy", quantum_inspired_max_hafnian_list)

# Finding the 16-node subgraph with the maximum Hafnian using Double-Loop Glauber Random Search
double_loop_glauber_max_hafnian_list = np.zeros((10, iteration+1))
for i in tqdm(range(10)):
    double_loop_glauber_max_hafnian_list[i], double_loop_glauber_best_subgraph, double_loop_glauber_best_hafnian = double_loop_glauber_random_search(G, k, c, iteration, mixing_time)

np.save("./Data/G4/double_loop_glauber_RS_hafnian_list.npy", double_loop_glauber_max_hafnian_list)






# Finding the 16-node subgraph with the maximum Hafnian using Simulated Annealing
max_hafnian_list = np.zeros((10, iteration+1))
for i in tqdm(range(10)):
    max_hafnian_list[i], best_subgraph, best_hafnian = simulated_annealing(G, k, iteration, t_initial)

np.save("./Data/G4/SA_hafnian_list.npy", max_hafnian_list)

# Finding the 16-node subgraph with the maximum Hafnian using Glauber Simulated Annealing
glauber_max_hafnian_list = np.zeros((10, iteration+1))
for i in tqdm(range(10)):
    glauber_max_hafnian_list[i], glauber_best_subgraph, glauber_best_hafnian = glauber_simulated_annealing(G, k, c, iteration, mixing_time, t_initial)

np.save("./Data/G4/glauber_SA_hafnian_list.npy", glauber_max_hafnian_list)

# Finding the 16-node subgraph with the maximum Hafnian using Jerrum Glauber Simulated Annealing
jerrum_glauber_max_hafnian_list = np.zeros((10, iteration+1))
for i in tqdm(range(10)):
    jerrum_glauber_max_hafnian_list[i], jerrum_glauber_best_subgraph, jerrum_glauber_best_hafnian = jerrum_glauber_simulated_annealing(G, k, c, iteration, mixing_time, t_initial)

np.save("./Data/G4/jerrum_glauber_SA_hafnian_list.npy", jerrum_glauber_max_hafnian_list)

# Finding the 16-node subgraph with the maximum hafnian using quantum-inspired Simulated Annealing
quantum_inspired_max_hafnian_list = np.zeros((10, iteration+1))
for i in tqdm(range(10)):
    quantum_inspired_max_hafnian_list[i], quantum_inspired_best_subgraph, quantum_inspired_best_hafnian = quantum_inspired_simulated_annealing(G, k, iteration)

np.save(f"./Data/G4/quantum_inspired_SA_hafnian_list.npy", quantum_inspired_max_hafnian_list)

# Finding the 16-node subgraph with the maximum Hafnian using Double-Loop Glauber Simulated Annealing
double_loop_glauber_max_hafnian_list = np.zeros((10, iteration+1))
for i in tqdm(range(10)):
    double_loop_glauber_max_hafnian_list[i], double_loop_glauber_best_subgraph, double_loop_glauber_best_hafnian = double_loop_glauber_simulated_annealing(G, k, c, iteration, mixing_time, t_initial)

np.save("./Data/G4/double_loop_glauber_SA_hafnian_list.npy", double_loop_glauber_max_hafnian_list)