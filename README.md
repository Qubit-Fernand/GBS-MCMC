# Efficient Classical Algorithms for Simulating GBS on Graphs

Code and data for our paper [Efficient Classical Algorithms for Simulating Gaussian Boson Sampling on Graphs](https://arxiv.org/abs/2505.02445).

## 👋 Overview
Gaussian Boson Sampling (GBS) is a promising candidate for demonstrating quantum computational advantage and can be applied to solving graph-related problems. In this work, we propose Markov chain Monte Carlo-based algorithms to simulate GBS on undirected, unweighted graphs. Numerically, we conduct experiments on various graphs with 256 vertices, and show that both the single-loop and double-loop Glauber dynamics improve the performance of original random search and simulated annealing algorithms for the max-Hafnian and densest k-subgraph problems up to 10x.

## 💽 Usage
1. `Glauber_hafnian.py` and `Glauber_density.py` contain the core functions and methods to be imported for Glauber dynamics and enhanced random search and simulate annealing algorithms.

2. The jupyter notebooks for each graph showcase the running pipeline: Generating graphs, running sampling with Glauber dynamics, then store the `.npy` data at `Data/` and finally plot at `Figure/`.

3. We also provide each notebook with corresponding python file in the same name, which can be directly running on high-performance computing platforms.

## ✍️ Citation
If you find our work helpful, please use the following citations.
```
@misc{zhang2025efficientclassicalalgorithmssimulating,
      title={Efficient Classical Algorithms for Simulating Gaussian Boson Sampling on Graphs}, 
      author={Yexin Zhang and Shuo Zhou and Xinzhao Wang and Ziruo Wang and Ziyi Yang and Rui Yang and Yecheng Xue and Tongyang Li},
      year={2025},
      eprint={2505.02445},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2505.02445}, 
}
```

## 🪪 License
MIT. Check `LICENSE`.
