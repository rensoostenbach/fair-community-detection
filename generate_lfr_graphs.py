import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
from utils import classify_graph

"""
Generate 1000 LFR graphs with set parameters, and classify whether they are suitable.
"""

SEEDS = list(range(1000))
PERCENTILE = {"size": 75, "density": 50}
fairness_graphs = []

for seed in SEEDS:
    try:
        G = LFR_benchmark_graph(
            n=1000, tau1=3, tau2=1.5, mu=0.2, min_degree=4, min_community=25, seed=seed
        )
        print(f"Graph created with seed: {seed}")
        fairness_graphs.append(classify_graph(G=G, percentile=PERCENTILE))
    except nx.ExceededMaxIterations:
        print(f"Failed to create graph with seed: {seed}")
        continue

size_fairness_graphs = [x[0] for x in fairness_graphs if x[0] is not None]
size_seeds = [i for i in range(len(fairness_graphs)) if fairness_graphs[i][0] != None]
density_fairness_graphs = [x[1] for x in fairness_graphs if x[1] is not None]
density_seeds = [
    i for i in range(len(fairness_graphs)) if fairness_graphs[i][1] != None
]

for graph, seed in zip(size_fairness_graphs, size_seeds):
    nx.write_gpickle(graph, f"data/labeled/lfr/size_graph_{seed}.pickle")

for graph, seed in zip(density_fairness_graphs, density_seeds):
    nx.write_gpickle(graph, f"data/labeled/lfr/density_graph_{seed}.pickle")

with open("data/labeled/lfr/size_seeds.txt", "w") as f:
    for line in size_seeds:
        f.write(f"{line}\n")

with open("data/labeled/lfr/density_seeds.txt", "w") as f:
    for line in density_seeds:
        f.write(f"{line}\n")
