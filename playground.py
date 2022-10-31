import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
from cdlib import algorithms
import numpy as np

from metrics.ground_truth import purity, inverse_purity
from metrics.own_metric import calculate_fairness_metrics
from utils import draw_graph, small_large_communities, lineplot_fairness, classify_graph


# The below plot shows that EMD is most often much higher than the other two
# This indicates two things:
# 1. Generally speaking, EMD simply overestimates (or the others underestimate)
# 2. More specifically, if one type (e.g. large comms) has very few (or no) misclassifications
#    EMD punishes this much more if the other type has relatively more misclassifications

# In the (rare) case that all metrics are quite low: We can really say this was a fair classification

N = 1000
TAU1 = 3
TAU2 = 1.5
MU = 0.3
SEEDS = list(range(20))
PERCENTILE = 75

fairness_graphs = []

for seed in SEEDS:

    try:
        G = LFR_benchmark_graph(
            n=N, tau1=TAU1, tau2=TAU2, mu=MU, min_degree=4, min_community=25, seed=seed
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

emd = []
f1 = []
acc = []

for G in size_fairness_graphs:
    G.remove_edges_from(nx.selfloop_edges(G))

    communities = {frozenset(G.nodes[v]["community"]) for v in G}
    gt_communities = list(communities)

    # CD part
    pred_coms = algorithms.louvain(g_original=G, randomize=0)

    #  Not important drawing stuff, just for myself
    # pos = nx.spring_layout(G)  # compute graph layout
    # draw_graph(G, pos=pos, communities=communities)

    (
        emd_fairness_score,
        f1_fairness_score,
        accuracy_fairness_score,
    ) = calculate_fairness_metrics(
        G=G,
        gt_communities=gt_communities,
        pred_communities=pred_coms.communities,
        fairness_type="size",
        percentile=PERCENTILE,
    )

    emd.append(emd_fairness_score)
    f1.append(f1_fairness_score)
    acc.append(accuracy_fairness_score)

lineplot_fairness(
    emd=emd,
    f1=f1,
    acc=acc,
    x_axis=size_seeds,
    xlabel="Seed chosen",
    title="Community size fairness score different seed \n(lines don't say anything but do show how they are correlated)",
)
print(np.array(emd) / np.array(f1))
print(np.array(emd) / np.array(acc))

emd = []
f1 = []
acc = []

for G in density_fairness_graphs:
    G.remove_edges_from(nx.selfloop_edges(G))

    communities = {frozenset(G.nodes[v]["community"]) for v in G}
    gt_communities = list(communities)

    # CD part
    pred_coms = algorithms.louvain(g_original=G, randomize=0)

    #  Not important drawing stuff, just for myself
    # pos = nx.spring_layout(G)  # compute graph layout
    # draw_graph(G, pos=pos, communities=communities)

    (
        emd_fairness_score,
        f1_fairness_score,
        accuracy_fairness_score,
    ) = calculate_fairness_metrics(
        G=G,
        gt_communities=gt_communities,
        pred_communities=pred_coms.communities,
        fairness_type="density",
        percentile=PERCENTILE,
    )

    emd.append(emd_fairness_score)
    f1.append(f1_fairness_score)
    acc.append(accuracy_fairness_score)

lineplot_fairness(
    emd=emd,
    f1=f1,
    acc=acc,
    x_axis=density_seeds,
    xlabel="Seed chosen",
    title="Density fairness score different seed \n(lines don't say anything but do show how they are correlated)",
)
print(np.array(emd) / np.array(f1))
print(np.array(emd) / np.array(acc))

# print(f"Purity: {purity(pred_coms=pred_coms.communities, real_coms=gt_communities)}")
# print(
#     f"Inverse purity: {inverse_purity(pred_coms=pred_coms.communities, real_coms=gt_communities)}"
# )
#
# print("debugger here")
