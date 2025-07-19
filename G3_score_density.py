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
G = nx.Graph(np.load(f"./Data/G3_sparse.npy"))

# Plot the hafnian score advantage versus click number
c = 0.4 # fugacity
iteration = 100
mixing_time = 1000
click_number_list = [16, 18, 20, 22, 24, 26, 28]

for k in click_number_list:
    best_density = np.zeros(10)
    double_loop_glauber_best_density = np.zeros(10)
    for i in tqdm(range(10)):
        _, _, best_density[i] = random_search(G, k, iteration)
        # print(f"Click number {k}: {best_density}")
        _, _, double_loop_glauber_best_density[i] = double_loop_glauber_random_search(G, k, c, iteration, mixing_time)
    score_advantage = double_loop_glauber_best_density / best_density
    np.save(f"./Data/G3/density_score_advantage_{k}.npy", score_advantage)