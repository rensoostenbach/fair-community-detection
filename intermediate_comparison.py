import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from cdlib.algorithms import eigenvector, label_propagation, leiden, louvain, spinglass
from cdlib.evaluation import adjusted_rand_index, variation_of_information
from cdlib import NodeClustering

from metrics.own_metric import calculate_fairness_metrics
from utils import scatterplot_fairness

"""
This file will give us an indication of what we can expect in the rest of the research
in terms of the comparison between community detection methods in terms of fairness and accuracy.
"""

CD_METHODS = [eigenvector, label_propagation, leiden, louvain, spinglass]
PERCENTILE = 75

for fairness_type in ["density", "size"]:
    fairness_scores = {}
    evaluation_scores = {}
    with open(f"data/labeled/lfr/{fairness_type}_seeds.txt") as seeds_file:
        seeds = [line.rstrip() for line in seeds_file]
    # seeds = seeds[:50]  # First 50 for now, for speed purposes
    for cd_method in CD_METHODS:
        print(f"Starting with {cd_method.__name__}")
        emd = []
        f1 = []
        acc = []
        ari = []
        vi = []
        for seed in seeds:
            with open(
                f"data/labeled/lfr/{fairness_type}_graph_{seed}.pickle", "rb"
            ) as graph_file:
                G = pickle.load(graph_file)

                G.remove_edges_from(nx.selfloop_edges(G))

                communities = {frozenset(G.nodes[v]["community"]) for v in G}
                gt_communities = list(communities)
                cdlib_communities = NodeClustering(communities=gt_communities, graph=G)

                # CD part
                try:
                    pred_coms = cd_method(g_original=G)
                except Exception:  # In case of convergence error or something else
                    emd.append(None)
                    f1.append(None)
                    acc.append(None)
                    ari.append(None)
                    vi.append(None)
                    continue

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
                ari.append(adjusted_rand_index(cdlib_communities, pred_coms))
                vi.append(variation_of_information(cdlib_communities, pred_coms))

        fairness_scores[cd_method.__name__] = (emd, f1, acc)
        evaluation_scores[cd_method.__name__] = (ari, vi)

    # Now we make scatterplots of accuracy vs fairness
    for evaluation_metric in ["ARI", "VI"]:
        for fairness_metric in ["EMD", "F1", "FCC"]:
            scatterplot_fairness(
                fairness_scores=fairness_scores,
                evaluation_scores=evaluation_scores,
                fairness_metric=fairness_metric,
                evaluation_metric=evaluation_metric,
                title=f"{evaluation_metric} vs {fairness_metric} {fairness_type} Fairness",
                filename=f"Scatterplot_{fairness_type}_{evaluation_metric}_{fairness_metric}",
            )
