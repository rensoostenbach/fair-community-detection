import networkx as nx
import numpy as np
from cdlib import algorithms

from utils import draw_graph
from data.labeled.casestudy.synthetic_graphs import (
    varying_mu_values,
    varying_denseness,
    mislabel_nodes,
)
from metrics.own_metric import calculate_fairness_metrics


# Varying mu values for pre-specified community sizes
num_small = 30
num_large = 30
mus = np.linspace(0.01, 0.5, 15)
# graphs = varying_mu_values(num_nodes_G1=num_small, num_nodes_G2=num_large, mus=mus)


# Varying denseness between two communities
num_nodes = 15
densenesses_G1 = [0.7, 0.8, 0.9]
densenesses_G2 = [0.25, 0.35, 0.45]
density_cutoff = 0.5
inter_community_edges = 0.1
graphs = varying_denseness(
    num_nodes=num_nodes,
    densenesses_G1=densenesses_G1,
    densenesses_G2=densenesses_G2,
    inter_community_edges=inter_community_edges,
)

for G in graphs:
    communities = {frozenset(G.nodes[v]["community"]) for v in G}
    gt_communities = list(communities)

    # CD part
    # predicted_communities = algorithms.louvain(g_original=G)

    #  Not important drawing stuff, just for myself
    pos = nx.spring_layout(G)  # compute graph layout
    draw_graph(G, pos=pos, communities=communities)

    G_mislabeled = mislabel_nodes(G=G, num_nodes=3, where_to_mislabel="sparse", density_cutoff=density_cutoff)
    mislabeled_communities = {frozenset(G_mislabeled.nodes[v]["community"]) for v in G_mislabeled}

    #  Not important drawing stuff, just for myself
    draw_graph(G_mislabeled, pos=pos, communities=mislabeled_communities)

    calculate_fairness_metrics(
            G=G,
            gt_communities=gt_communities,
            pred_communities=list(mislabeled_communities),
            fairness_type="density",
            density_cutoff=density_cutoff
        )
