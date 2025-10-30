# Efficient Classical Sampling from Gaussian Boson Sampling Distributions on Unweighted Graphs

Code and data for our paper [Efficient Classical Sampling from Gaussian Boson Sampling Distributions on Unweighted Graphs](https://doi.org/10.1038/s41467-025-64442-7).

## 👋 Overview

Gaussian Boson Sampling (GBS) is a promising candidate for demonstrating quantum computational advantage and can be applied to solving graph-related problems. In this work, we propose Markov chain Monte Carlo-based algorithms to sample from GBS distributions on undirected, unweighted graphs. Numerically, we conduct experiments on various graphs with 256 vertices, and show that both the single-loop and double-loop Glauber dynamics improve the performance of original random search and simulated annealing algorithms for the max-Hafnian and densest k-subgraph problems up to 10x.

## 💽 Usage

1. `Glauber_hafnian.py` and `Glauber_density.py` contain the core functions and methods to be imported for Glauber dynamics and enhanced random search and simulate annealing algorithms.
2. The jupyter notebooks for each graph showcase the running pipeline: Generating graphs, running sampling with Glauber dynamics, then store the `.npy` data at `Data/` and finally plot `.pdf` at `Figure/`.
3. We also provide each notebook with corresponding python file in the same name, which can be directly running on high-performance computing platforms. You can first use `Graph.ipynb` to generate graphs and use `Plot.ipynb` to plot with output data.

## ✍️ Citation

If you find our work helpful, please use the following citations.

```
@article{zhang_efficient_2025,
	title = {Efficient classical sampling from {Gaussian} boson sampling distributions on unweighted graphs},
	volume = {16},
	issn = {2041-1723},
	url = {https://doi.org/10.1038/s41467-025-64442-7},
	doi = {10.1038/s41467-025-64442-7},
	number = {1},
	journal = {Nature Communications},
	author = {Zhang, Yexin and Zhou, Shuo and Wang, Xinzhao and Wang, Ziruo and Yang, Ziyi and Yang, Rui and Xue, Yecheng and Li, Tongyang},
	month = oct,
	year = {2025},
	pages = {9335},
    eprint={2505.02445},
}
```

## 🪪 License

MIT. Check `LICENSE`.
