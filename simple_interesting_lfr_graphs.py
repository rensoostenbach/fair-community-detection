import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
from cdlib import algorithms
import numpy as np

from metrics.ground_truth import purity, inverse_purity
from metrics.own_metric import calculate_fairness_metrics
from utils import (
    draw_graph,
    small_large_communities,
    lineplot_fairness,
    classify_graph,
    interesting_playground_graphs,
)


# The below plots shows that EMD is most often much lower than the other two
# This indicates two things:
# 1. Generally speaking, EMD simply underestimates (or the others overestimate)
# 2. More specifically, if one type (e.g. large comms) has very few (or no) misclassifications
#    EMD punishes this much more if the other type has relatively more misclassifications

# In the case that all metrics are quite high: We can really say this was a fair classification

N = 100
TAU1 = 3
TAU2 = 1.5
MU = 0.21
SEEDS = list(range(40))
PERCENTILE = 75

fairness_graphs = []

for seed in SEEDS:
    try:
        G = LFR_benchmark_graph(
            n=N, tau1=TAU1, tau2=TAU2, mu=MU, min_degree=3, min_community=4, seed=seed
        )
        print(f"Graph created with seed: {seed}")
        fairness_graphs.append(classify_graph(G=G, percentile=PERCENTILE))
    except nx.ExceededMaxIterations:
        print(f"Failed to create graph with seed: {seed}")
        continue

size_fairness_graphs = [x[0] for x in fairness_graphs if x[0] is not None]
density_fairness_graphs = [x[1] for x in fairness_graphs if x[1] is not None]

fairness_types = ["size", "density"]
for fairness_type in fairness_types:
    emd = []
    f1 = []
    acc = []

    for idx, G in enumerate(eval(f"{fairness_type}_fairness_graphs")):
        G.remove_edges_from(nx.selfloop_edges(G))

        communities = {frozenset(G.nodes[v]["community"]) for v in G}
        gt_communities = list(communities)

        # CD part
        pred_coms = algorithms.louvain(g_original=G)

        (
            emd_fairness_score,
            f1_fairness_score,
            accuracy_fairness_score,
            fractions_type1,
            fractions_type2,
            f1_type1,
            f1_type2,
            precision_type1,
            precision_type2,
            recall_type1,
            recall_type2,
            mapping_list,
            comm_types,
        ) = calculate_fairness_metrics(
            G=G,
            gt_communities=gt_communities,
            pred_communities=pred_coms.communities,
            fairness_type=fairness_type,
            percentile=PERCENTILE,
            interpret_results=True,
        )

        emd.append(emd_fairness_score)
        f1.append(f1_fairness_score)
        acc.append(accuracy_fairness_score)

        if (
            abs(f1_fairness_score - emd_fairness_score) < 0.2
            and emd_fairness_score > 0.7
        ):
            interesting_playground_graphs(
                fair_unfair="fair",
                fairness_type=fairness_type,
                G=G,
                idx=idx,
                communities=gt_communities,
                pred_coms=pred_coms.communities,
                emd=emd_fairness_score,
                f1=f1_fairness_score,
                acc=accuracy_fairness_score,
                frac_type1=fractions_type1,
                frac_type2=fractions_type2,
                f1_type1=f1_type1,
                f1_type2=f1_type2,
                precision_type1=precision_type1,
                precision_type2=precision_type2,
                recall_type1=recall_type1,
                recall_type2=recall_type2,
                comm_types=comm_types,
                mapping_list=mapping_list,
            )

        elif (
            f1_fairness_score - emd_fairness_score > 0.3 and emd_fairness_score < 0.5
        ) or accuracy_fairness_score < 0.6:
            interesting_playground_graphs(
                fair_unfair="unfair",
                fairness_type=fairness_type,
                G=G,
                idx=idx,
                communities=gt_communities,
                pred_coms=pred_coms.communities,
                emd=emd_fairness_score,
                f1=f1_fairness_score,
                acc=accuracy_fairness_score,
                frac_type1=fractions_type1,
                frac_type2=fractions_type2,
                f1_type1=f1_type1,
                f1_type2=f1_type2,
                precision_type1=precision_type1,
                precision_type2=precision_type2,
                recall_type1=recall_type1,
                recall_type2=recall_type2,
                comm_types=comm_types,
                mapping_list=mapping_list,
            )

        else:
            continue

    lineplot_fairness(
        emd=emd,
        f1=f1,
        acc=acc,
        x_axis=list(range(len(eval(f"{fairness_type}_fairness_graphs")))),
        xlabel="Generated graph number",
        noline=True,
        title=f"{fairness_type} fairness score for generated graphs",
        filename=f"lineplot_simple_interesting_lfr_graphs_{fairness_type}",
    )


# emd = []
# f1 = []
# acc = []
#
# for G in density_fairness_graphs:
#     G.remove_edges_from(nx.selfloop_edges(G))
#
#     communities = {frozenset(G.nodes[v]["community"]) for v in G}
#     gt_communities = list(communities)
#
#     # CD part
#     pred_coms = algorithms.louvain(g_original=G)
#
#     #  Not important drawing stuff, just for myself
#     # pos = nx.spring_layout(G)  # compute graph layout
#     # draw_graph(G, pos=pos, communities=communities)
#
#     (
#         emd_fairness_score,
#         f1_fairness_score,
#         accuracy_fairness_score,
#     ) = calculate_fairness_metrics(
#         G=G,
#         gt_communities=gt_communities,
#         pred_communities=pred_coms.communities,
#         fairness_type="density",
#         percentile=PERCENTILE,
#     )
#
#     emd.append(emd_fairness_score)
#     f1.append(f1_fairness_score)
#     acc.append(accuracy_fairness_score)
#
# lineplot_fairness(
#     emd=emd,
#     f1=f1,
#     acc=acc,
#     x_axis=list(range(len(density_fairness_graphs))),
#     xlabel="Generated graph number",
#     noline=True,
#     title="Density fairness score different seed",
# )
