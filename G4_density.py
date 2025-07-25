import networkx as nx
from thewalrus import hafnian
from tqdm import tqdm
import numpy as np
import random 
import matplotlib.pyplot as plt
import copy
from Glauber_density import *

# Number of vertexes
n = 256
c = 0.8 # fugacity
k = 80
iteration = 1000
mixing_time = 1000 # density mixing time 1000 x 100, since k = 80 computationally expensive
t_initial = 1.0

G = nx.Graph(np.load(f"./Data/G4.npy"))

# Finding the 80-node subgraph with the maximum density using Random Search
max_density_list = np.zeros((10, iteration+1))
for i in tqdm(range(10)):
    max_density_list[i], best_subgraph, best_density = random_search(G, k, iteration)

np.save("./Data/G4/RS_density_list.npy", max_density_list)

# Finding the 80-node subgraph with the maximum density using Glauber Random Search
glauber_max_density_list = np.zeros((10, iteration+1))
for i in tqdm(range(10)):
    glauber_max_density_list[i], glauber_best_subgraph, glauber_best_density = glauber_random_search(G, k, c, iteration, mixing_time)

np.save("./Data/G4/glauber_RS_density_list.npy", glauber_max_density_list)


# Finding the 80-node subgraph with the maximum density using Jerrum Glauber Random Search
jerrum_glauber_max_density_list = np.zeros((10, iteration+1))
for i in tqdm(range(10)):
    jerrum_glauber_max_density_list[i], jerrum_glauber_best_subgraph, jerrum_glauber_best_density = jerrum_glauber_random_search(G, k, c, iteration, mixing_time)

np.save("./Data/G4/jerrum_glauber_RS_density_list.npy", jerrum_glauber_max_density_list)

# Finding the 80-node subgraph with the maximum density using Double-Loop Glauber Random Search
double_loop_glauber_max_density_list = np.zeros((10, iteration+1))
for i in tqdm(range(10)):
    double_loop_glauber_max_density_list[i], double_loop_glauber_best_subgraph, double_loop_glauber_best_density = double_loop_glauber_random_search(G, k, c, iteration, mixing_time)

np.save("./Data/G4/double_loop_glauber_RS_density_list.npy", double_loop_glauber_max_density_list)







# Finding the 80-node subgraph with the maximum density using Simulated Annealing
max_density_list = np.zeros((10, iteration+1))
for i in tqdm(range(10)):
    max_density_list[i], best_subgraph, best_density = simulated_annealing(G, k, iteration, t_initial)

np.save("./Data/G4/SA_density_list.npy", max_density_list)

# Finding the 80-node subgraph with the maximum density using Glauber Simulated Annealing
glauber_max_density_list = np.zeros((10, iteration+1))
for i in tqdm(range(10)):
    glauber_max_density_list[i], glauber_best_subgraph, glauber_best_density = glauber_simulated_annealing(G, k, c, iteration, mixing_time, t_initial)

np.save("./Data/G4/glauber_SA_density_list.npy", glauber_max_density_list)

# Finding the 80-node subgraph with the maximum density using Jerrum Glauber Simulated Annealing
jerrum_glauber_max_density_list = np.zeros((10, iteration+1))
for i in tqdm(range(10)):
    jerrum_glauber_max_density_list[i], jerrum_glauber_best_subgraph, jerrum_glauber_best_density = jerrum_glauber_simulated_annealing(G, k, c, iteration, mixing_time, t_initial)

np.save("./Data/G4/jerrum_glauber_SA_density_list.npy", jerrum_glauber_max_density_list)


# Finding the 80-node subgraph with the maximum density using Double-Loop Glauber Simulated Annealing
double_loop_glauber_max_density_list = np.zeros((10, iteration+1))
for i in tqdm(range(10)):
    double_loop_glauber_max_density_list[i], double_loop_glauber_best_subgraph, double_loop_glauber_best_density = double_loop_glauber_simulated_annealing(G, k, c, iteration, mixing_time, t_initial)

np.save("./Data/G4/double_loop_glauber_SA_density_list.npy", double_loop_glauber_max_density_list)