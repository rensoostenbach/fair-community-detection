import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
from cdlib import algorithms
import numpy as np

from metrics.ground_truth import purity, inverse_purity
from metrics.own_metric import calculate_fairness_metrics
from utils import draw_graph, small_large_communities, plot_fairness


# The below plot shows that EMD is most often much higher than the other two
# This indicates two things:
# 1. Generally speaking, EMD simply overestimates (or the others underestimate)
# 2. More specifically, if one type (e.g. large comms) has very few (or no) misclassifications
#    EMD punishes this much more if the other type has relatively more misclassifications

# In the (rare) case that all metrics are quite low: We can really say this was a fair classification

n = 1000
tau1 = 3
tau2 = 1.5
mu = 0.3
seeds = list(range(20))
emd = []
f1 = []
acc = []

for seed in seeds:

    try:
        G = LFR_benchmark_graph(
            n=n, tau1=tau1, tau2=tau2, mu=mu, min_degree=4, min_community=25, seed=seed
        )
        print(f"Graph created with seed: {seed}")
    except nx.ExceededMaxIterations:
        emd.append(np.nan)
        f1.append(np.nan)
        acc.append(np.nan)
        continue

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
        fairness_type="small_large",
        size_percentile=75,
    )

    emd.append(emd_fairness_score)
    f1.append(f1_fairness_score)
    acc.append(accuracy_fairness_score)

plot_fairness(
    emd=emd,
    f1=f1,
    acc=acc,
    x_axis=seeds,
    xlabel="Seed chosen",
    title="Fairness score different seed \n(lines don't say anything but do show how they are correlated)",
)

# print(f"Purity: {purity(pred_coms=pred_coms.communities, real_coms=gt_communities)}")
# print(
#     f"Inverse purity: {inverse_purity(pred_coms=pred_coms.communities, real_coms=gt_communities)}"
# )
#
# print("debugger here")
